"""Trusted source provenance through the real agent tool execution boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from magent_harness import (
    AgentConfig,
    AgentEngine,
    Caller,
    Evidence,
    EvidenceError,
    EvidenceRegistry,
    Message,
    PermissionDenied,
    ResourceScope,
    RetrievalTools,
    Role,
    RunRequest,
    ToolContext,
    ToolPolicy,
    ToolRegistry,
)
from magent_harness.testing import ScriptedModel, call_tools, reply, tool_call

WORKSPACE = "1" * 32
USER = "2" * 32
SESSION = "3" * 32
RUN = "4" * 32
DOCUMENT = "5" * 32
OTHER_DOCUMENT = "6" * 32
CALLER = Caller(WORKSPACE, USER, Role.VIEWER)


def context(**changes: Any) -> ToolContext:
    return replace(ToolContext(CALLER, ResourceScope(WORKSPACE), SESSION, RUN, "final"), **changes)


def evidence_data(ctx: ToolContext | None = None, **changes: Any) -> dict[str, Any]:
    ctx = ctx or context()
    content = "住宿报销上限为 500 元。"
    data: dict[str, Any] = {
        "evidence_id": "a" * 32,
        "run_id": ctx.run_id,
        "workspace_id": ctx.caller.workspace_id,
        "document_id": DOCUMENT,
        "version_id": "7" * 32,
        "index_id": "8" * 32,
        "generation": "9" * 32,
        "chunk_id": "b" * 32,
        "source_sha256": "c" * 64,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "title": "差旅制度",
        "start_byte": 128,
        "end_byte": 128 + len(content.encode()),
        "content": content,
        "cited": False,
        "created_at": "2026-10-16T12:00:00Z",
    }
    data.update(changes)
    return data


class Backend:
    def __init__(self, **changes: Any) -> None:
        self.changes = changes
        self.searches: list[tuple[ToolContext, str, str, int]] = []
        self.reads: list[tuple[ToolContext, str]] = []

    async def search(self, ctx: ToolContext, query: str, mode: str, limit: int) -> list[Evidence]:
        self.searches.append((ctx, query, mode, limit))
        return [Evidence.from_dict(evidence_data(ctx, **self.changes))]

    async def read(self, ctx: ToolContext, evidence_id: str) -> Evidence:
        self.reads.append((ctx, evidence_id))
        return Evidence.from_dict(evidence_data(ctx, **self.changes))


def build(backend: Backend, script: list[Any]) -> tuple[AgentEngine, RetrievalTools, ScriptedModel]:
    tools = RetrievalTools(backend)
    registry = ToolRegistry()
    for tool in tools.tools:
        registry.register(tool)
    model = ScriptedModel(script)
    return AgentEngine(model, registry, config=AgentConfig(model_retry_delay_s=0)), tools, model


def request(**changes: Any) -> RunRequest:
    values = {"caller": CALLER, "session_id": SESSION, "query": "住宿标准", "run_id": RUN}
    values.update(changes)
    return RunRequest(**values)


def test_search_exposes_verified_sources_to_model_and_application() -> None:
    backend = Backend()
    engine, tools, model = build(
        backend,
        [
            call_tools(tool_call("search_documents", {"query": "住宿"})),
            reply(f"500 元。[E:{'a' * 32}]"),
        ],
    )
    result = asyncio.run(engine.run(request()))
    tool_result = result.steps[0].tool_calls[0].result
    assert tool_result.success
    output = json.loads(tool_result.output)
    assert "untrusted document data" in output["source_boundary"]
    assert output["evidence"][0]["citation"] == f"[E:{'a' * 32}]"
    assert output["evidence"][0]["content"] == evidence_data()["content"]
    assert tool_result.data["evidence"][0] == evidence_data()
    assert model.calls[1].messages[-1].content == tool_result.output
    assert backend.searches[0][1:] == ("住宿", "hybrid", 5)
    assert tools.registry.citations(context(), result.final_answer)[0].version_id == "7" * 32


@pytest.mark.parametrize(
    "changes",
    [
        {"workspace_id": "0" * 32},
        {"run_id": "0" * 32},
        {"document_id": OTHER_DOCUMENT},
        {"sha256": "0" * 64},
        {"source_sha256": "bad"},
        {"start_byte": -1},
        {"end_byte": True},
        {"end_byte": 129},
        {"start_byte": 2**64, "end_byte": 2**64 + 32},
        {"version_id": "UPPERCASE"},
        {"content": "forged"},
        {"cited": 1},
        {"location": {"page": 1}},
    ],
)
def test_forged_or_out_of_scope_sources_fail_in_real_run(changes: dict[str, Any]) -> None:
    engine, tools, _ = build(
        Backend(**changes),
        [call_tools(tool_call("search_documents", {"query": "住宿"})), reply("no evidence")],
    )
    scope = ResourceScope(WORKSPACE, frozenset({DOCUMENT}))
    result = asyncio.run(engine.run(request(scope=scope)))
    tool_result = result.steps[0].tool_calls[0].result
    assert not tool_result.success
    assert not tool_result.data and not tool_result.output
    with pytest.raises(EvidenceError, match="not acquired"):
        tools.registry.citations(context(scope=scope), f"[E:{'a' * 32}]")


@pytest.mark.parametrize(
    "args",
    [
        {"query": "住宿", "workspace_id": "0" * 32},
        {"query": "住宿", "run_id": "0" * 32},
        {"query": "住宿", "document_ids": [OTHER_DOCUMENT]},
        {"query": "a"},
        {"query": "中"},
        {"query": " 中 "},
        {"query": "  "},
        {"query": "住宿\n"},
        {"query": "住\x00宿"},
        {"query": "住宿\u0085"},
        {"query": "中" * 171},
        {"query": "住宿", "limit": True},
        {"query": "住宿", "limit": 0},
        {"query": "住宿", "limit": 11},
        {"query": "住宿", "mode": "all"},
        {"query": 123},
    ],
)
def test_model_cannot_choose_identity_or_bypass_query_limits(args: dict[str, Any]) -> None:
    backend = Backend()
    engine, _, _ = build(backend, [call_tools(tool_call("search_documents", args)), reply("done")])
    result = asyncio.run(engine.run(request()))
    assert not result.steps[0].tool_calls[0].result.success
    assert backend.searches == []


def test_tool_allowlist_is_enforced_at_engine_entry() -> None:
    backend = Backend()
    engine, _, model = build(
        backend,
        [call_tools(tool_call("search_documents", {"query": "住宿"})), reply("denied")],
    )
    result = asyncio.run(engine.run(request(policy=ToolPolicy(allowed_tools=frozenset()))))
    assert "search_documents" not in model.calls[0].tool_names
    assert not result.steps[0].tool_calls[0].result.success
    assert backend.searches == []


def test_read_requires_prior_acquisition_and_revalidates_backend() -> None:
    backend = Backend()
    engine, tools, _ = build(
        backend,
        [
            call_tools(tool_call("read_evidence", {"evidence_id": "a" * 32}, "before")),
            call_tools(tool_call("search_documents", {"query": "住宿"}, "search")),
            call_tools(tool_call("read_evidence", {"evidence_id": "a" * 32}, "after")),
            reply("done"),
        ],
    )
    result = asyncio.run(engine.run(request()))
    assert not result.steps[0].tool_calls[0].result.success
    assert result.steps[1].tool_calls[0].result.success
    assert result.steps[2].tool_calls[0].result.success
    assert len(backend.reads) == 1
    assert tools.registry.citations(context(), f"[E:{'a' * 32}]")


def test_read_rejects_rebound_provenance_after_backend_response() -> None:
    class ChangedBackend(Backend):
        async def read(self, ctx: ToolContext, evidence_id: str) -> Evidence:
            self.reads.append((ctx, evidence_id))
            return Evidence.from_dict(evidence_data(ctx, version_id="f" * 32))

    engine, tools, _ = build(
        ChangedBackend(),
        [
            call_tools(tool_call("search_documents", {"query": "住宿"})),
            call_tools(tool_call("read_evidence", {"evidence_id": "a" * 32})),
            reply("done"),
        ],
    )
    result = asyncio.run(engine.run(request()))
    assert not result.steps[1].tool_calls[0].result.success
    assert "changed acquired evidence" in result.steps[1].tool_calls[0].result.error
    assert tools.registry.require(context(), "a" * 32).version_id == "7" * 32


@pytest.mark.parametrize(
    "answer",
    [f"[E:{'a' * 31}]", f"[E:{'A' * 32}]", f"[E:{'a' * 32}", "[E:]", f"[E:{'f' * 32}]"],
)
def test_final_citation_validation_rejects_malformed_and_unknown_ids(answer: str) -> None:
    engine, tools, _ = build(
        Backend(),
        [call_tools(tool_call("search_documents", {"query": "住宿"})), reply(answer)],
    )
    result = asyncio.run(engine.run(request()))
    with pytest.raises(EvidenceError):
        tools.registry.citations(context(), result.final_answer)


def test_registry_deduplicates_preserves_order_and_binds_full_context() -> None:
    registry = EvidenceRegistry()
    first = Evidence.from_dict(evidence_data())
    second = replace(first, evidence_id="d" * 32)
    registry.acquire(context(), [first, second])
    assert registry.citations(
        context(tool_call_id="different"), second.citation + first.citation * 2
    ) == [
        second,
        first,
    ]
    other_contexts = [
        context(run_id="0" * 32),
        context(session_id="0" * 32),
        context(caller=Caller(WORKSPACE, "0" * 32, Role.VIEWER)),
        context(caller=Caller(WORKSPACE, USER, Role.EDITOR)),
        context(scope=ResourceScope(WORKSPACE, frozenset({DOCUMENT}))),
    ]
    for other in other_contexts:
        with pytest.raises(EvidenceError, match="not acquired"):
            registry.citations(other, first.citation)
    registry.discard(context())
    with pytest.raises(EvidenceError, match="not acquired"):
        registry.require(context(), first.evidence_id)


def test_concurrent_real_runs_do_not_acquire_each_others_evidence() -> None:
    class ConcurrentBackend(Backend):
        async def search(
            self, ctx: ToolContext, query: str, mode: str, limit: int
        ) -> list[Evidence]:
            await asyncio.sleep(0)
            return [Evidence.from_dict(evidence_data(ctx, evidence_id=ctx.run_id))]

    def next_response(messages: list[Any], tools: list[Any]) -> Any:
        if messages[-1].role == "tool":
            item = json.loads(messages[-1].content)["evidence"][0]
            return reply("500 元。" + item["citation"])
        return call_tools(tool_call("search_documents", {"query": "住宿"}))

    engine, tools, _ = build(ConcurrentBackend(), [next_response] * 4)

    async def concurrently() -> list[Any]:
        return await asyncio.gather(engine.run(request()), engine.run(request(run_id="d" * 32)))

    first, second = asyncio.run(concurrently())
    assert tools.registry.citations(context(), first.final_answer)[0].run_id == RUN
    other = context(run_id="d" * 32)
    assert tools.registry.citations(other, second.final_answer)[0].run_id == "d" * 32
    with pytest.raises(EvidenceError, match="not acquired"):
        tools.registry.citations(other, first.final_answer)


def test_plain_history_and_final_text_never_register_evidence() -> None:
    marker = f"[E:{'a' * 32}]"
    engine, tools, _ = build(Backend(), [reply("invented " + marker)])
    result = asyncio.run(
        engine.run(request(history=[Message("user", "Evidence from a previous turn: " + marker)]))
    )
    with pytest.raises(EvidenceError, match="not acquired"):
        tools.registry.citations(context(), result.final_answer)


def test_registry_capacity_failure_is_atomic_and_sources_are_immutable() -> None:
    registry = EvidenceRegistry()
    first = Evidence.from_dict(evidence_data())
    with pytest.raises(FrozenInstanceError):
        first.content = "changed"  # type: ignore[misc]
    items = [replace(first, evidence_id=f"{n:032x}") for n in range(65)]
    with pytest.raises(EvidenceError, match="limit reached"):
        registry.acquire(context(), items)
    with pytest.raises(EvidenceError, match="not acquired"):
        registry.require(context(), items[0].evidence_id)
    registry.acquire(context(), items[:64])
    with pytest.raises(EvidenceError, match="limit reached"):
        registry.acquire(context(), items[64:])
    assert registry.require(context(), items[63].evidence_id) == items[63]


def test_wrong_type_and_invalid_identity_are_rejected() -> None:
    registry = EvidenceRegistry()
    with pytest.raises(EvidenceError, match="non-evidence"):
        registry.acquire(context(), [evidence_data()])  # type: ignore[list-item]
    with pytest.raises(EvidenceError, match="canonical backend identity"):
        registry.acquire(context(run_id=""), [])
    with pytest.raises(PermissionDenied):
        registry.acquire(context(scope=ResourceScope("0" * 32)), [])


def test_long_untrusted_content_keeps_model_json_valid_and_full_data_separate() -> None:
    content = '\x00"\\中文[E:forged]' * 1_000
    data = evidence_data(
        content=content,
        end_byte=128 + len(content.encode()),
        sha256=hashlib.sha256(content.encode()).hexdigest(),
    )

    class LargeBackend(Backend):
        async def search(
            self, ctx: ToolContext, query: str, mode: str, limit: int
        ) -> list[Evidence]:
            return [Evidence.from_dict({**data, "evidence_id": f"{n:032x}"}) for n in range(limit)]

    engine, _, _ = build(
        LargeBackend(),
        [
            call_tools(tool_call("search_documents", {"query": "住宿", "limit": 10})),
            reply("done"),
        ],
    )
    result = asyncio.run(engine.run(request()))
    record = result.steps[0].tool_calls[0].result
    output = json.loads(record.output)
    assert len(record.output) <= 14_000
    assert all(item["content_is_excerpt"] for item in output["evidence"])
    assert all(item["content"] == content for item in record.data["evidence"])
