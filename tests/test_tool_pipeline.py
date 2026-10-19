"""Bounded execution, cancellation and canonical tool results at real entry points."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from support import EDITOR, build, run

from magent_harness import (
    AccessLevel,
    AgentConfig,
    AgentEngine,
    EventType,
    FunctionTool,
    JsonlRunStore,
    RecoveryUnsafe,
    Risk,
    RunRequest,
    RunStatus,
    ToolContext,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from magent_harness.testing import ScriptedModel, call_tools, reply, tool_call


def request() -> RunRequest:
    return RunRequest(EDITOR, "s1", "Use the tools", policy=ToolPolicy(AccessLevel.AUTO_WRITE))


@pytest.mark.parametrize("field", ["max_parallel_tools", "max_tool_calls_per_round"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_execution_limits_fail_at_configuration(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        AgentConfig(**{field: value})


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_tool_deadline_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="timeout"):
        ToolSpec("read", "Read", timeout_s=value)


def test_bounded_workers_complete_out_of_order_but_preserve_model_order() -> None:
    async def scenario() -> None:
        release_first = asyncio.Event()
        active = peak = 0
        workers: set[asyncio.Task[Any] | None] = set()
        completed: list[str] = []

        async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            nonlocal active, peak
            workers.add(asyncio.current_task())
            active += 1
            peak = max(peak, active)
            try:
                if ctx.tool_call_id == "c0":
                    await asyncio.wait_for(release_first.wait(), 1)
                if ctx.tool_call_id == "c19":
                    release_first.set()
                completed.append(ctx.tool_call_id)
                return ToolResult.ok(ctx.tool_call_id)
            finally:
                active -= 1

        registry = ToolRegistry()
        registry.register(FunctionTool(ToolSpec("read", "Read", concurrency_safe=True), read))
        h = build(
            [call_tools(*(tool_call("read", {}, f"c{i}") for i in range(20))), reply("done")],
            registry=registry,
            config=AgentConfig(max_parallel_tools=2),
        )
        result = await h.engine.run(request())
        assert result.status is RunStatus.COMPLETED
        assert peak == 2 and active == 0 and len(workers) == 2
        assert completed[0] == "c1" and completed[-1] == "c0"
        assert [r.id for r in result.steps[0].tool_calls] == [f"c{i}" for i in range(20)]
        assert [m.tool_call_id for m in h.model.calls[1].messages if m.role == "tool"] == [
            f"c{i}" for i in range(20)
        ]
        assert [e.data["tool_call_id"] for e in h.sink.of_type(EventType.TOOL_RESULT)] == completed

    run(scenario())


def test_cancel_joins_workers_and_never_starts_queued_calls() -> None:
    async def scenario() -> None:
        started: list[str] = []
        stopped: list[str] = []
        both_started = asyncio.Event()

        async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            started.append(ctx.tool_call_id)
            if len(started) == 2:
                both_started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            finally:
                stopped.append(ctx.tool_call_id)

        registry = ToolRegistry()
        registry.register(FunctionTool(ToolSpec("read", "Read", concurrency_safe=True), read))
        h = build(
            [call_tools(*(tool_call("read", {}, f"c{i}") for i in range(20)))],
            registry=registry,
            config=AgentConfig(max_parallel_tools=2),
        )
        task = asyncio.create_task(h.engine.run(request()))
        await asyncio.wait_for(both_started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert set(started) == set(stopped) == {"c0", "c1"}
        assert len(h.model.calls) == 1
        assert not h.sink.of_type(EventType.TOOL_RESULT)

    run(scenario())


def test_oversized_round_is_refused_before_any_tool_executes() -> None:
    h = build(
        [call_tools(*(tool_call("read_document", {"document_id": "d1"}) for _ in range(3)))],
        config=AgentConfig(max_tool_calls_per_round=2),
    )
    result = run(h.engine.run(request()))
    assert result.status is RunStatus.FAILED and "too many tool calls" in result.error
    assert not h.workspace.reads and not h.sink.of_type(EventType.TOOL_CALL)


def test_lower_round_limit_also_applies_to_approval_resume() -> None:
    h = build(
        [
            call_tools(
                tool_call("update_document", {"document_id": "d1", "content": "v2"}),
                tool_call("read_document", {"document_id": "d1"}),
            )
        ]
    )
    first = run(h.engine.run(RunRequest(EDITOR, "s1", "Update")))
    assert first.prompt is not None
    h.broker.resolve(EDITOR, first.prompt.id, {"approved": True})
    h.engine.config = AgentConfig(max_tool_calls_per_round=1)
    result = run(h.engine.resume(EDITOR, first.prompt.id))
    assert result.status is RunStatus.FAILED
    assert not h.workspace.writes and not h.workspace.reads


@pytest.mark.parametrize("failure", ["exception", "invalid_data", "timeout"])
def test_unknown_mutation_stops_even_without_a_run_store(failure: str) -> None:
    writes: list[str] = []

    async def write(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        writes.append(ctx.tool_call_id)
        if failure == "exception":
            raise RuntimeError("private backend details")
        if failure == "timeout":
            await asyncio.Event().wait()
        return ToolResult(True, "done", data={"invalid": object()})

    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            ToolSpec("write", "Write", risk=Risk.WRITE, concurrency_safe=True, timeout_s=0.01),
            write,
        )
    )
    h = build(
        [call_tools(tool_call("write", {}, "w1"), tool_call("write", {}, "w2")), reply("done")],
        registry=registry,
    )
    with pytest.raises(RecoveryUnsafe, match="outcome_unknown"):
        run(h.engine.run(request()))
    assert writes == ["w1"] and len(h.model.calls) == 1


@pytest.mark.parametrize(
    "data",
    [
        {"x": float("nan")},
        {1: "bad key"},
        {"x": object()},
        {"x": "x" * 500},
        {"outcome_unknown": "yes"},
    ],
)
def test_invalid_private_data_becomes_a_safe_read_failure(data: dict[Any, Any]) -> None:
    async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(True, "PRIVATE invalid result", data=data)

    registry = ToolRegistry(max_data_bytes=256)
    registry.register(FunctionTool(ToolSpec("read", "Read"), read))
    h = build([call_tools(tool_call("read")), reply("done")], registry=registry)
    result = run(h.engine.run(request())).steps[0].tool_calls[0].result
    assert not result.success and "invalid or oversized" in result.error
    assert "PRIVATE" not in str(h.model.calls[1].messages)


def test_results_are_detached_and_json_excerpts_remain_valid() -> None:
    source = {"evidence": [{"id": "trusted"}]}
    original = json.dumps({"content": "x" * 500, "tail": "important"})

    async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(True, original, "error" * 100, source)

    def finish(messages: Any, tools: Any) -> Any:
        source["evidence"][0]["id"] = "changed after tool completion"
        return reply("done")

    registry = ToolRegistry(max_output_chars=80, max_error_chars=20)
    registry.register(FunctionTool(ToolSpec("read", "Read"), read))
    h = build([call_tools(tool_call("read")), finish], registry=registry)
    result = run(h.engine.run(request())).steps[0].tool_calls[0].result
    excerpt = json.loads(result.output)
    assert excerpt["truncated"] and "important" in excerpt["excerpt"]
    assert excerpt["omitted_chars"] == len(original) - 80
    assert len(result.error) < 100
    assert result.data["evidence"][0]["id"] == "trusted"
    assert result.data["result_governance"]["error_omitted_chars"] == 480


def test_exception_text_never_reaches_model_events_or_logs(caplog: Any) -> None:
    async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("SECRET-TOKEN private document")

    registry = ToolRegistry()
    registry.register(FunctionTool(ToolSpec("read", "Read"), read))
    h = build([call_tools(tool_call("read")), reply("done")], registry=registry)
    result = run(h.engine.run(request()))
    assert "RuntimeError" in result.steps[0].tool_calls[0].result.error
    assert "SECRET-TOKEN" not in str((h.model.calls, h.sink.events, caplog.text))


@pytest.mark.parametrize("kind", ["cycle", "deep", "wide", "bad_unicode", "wrong_success"])
def test_result_structure_limits_fail_safely(kind: str) -> None:
    data: dict[str, Any] = {}
    if kind == "cycle":
        data["self"] = data
    elif kind == "deep":
        current = data
        for _ in range(40):
            current["next"] = {}
            current = current["next"]
    elif kind == "wide":
        data["items"] = [None] * 10_001
    elif kind == "bad_unicode":
        data["text"] = chr(0xD800)
    result = ToolResult(True, "ok", data=data)
    if kind == "wrong_success":
        result.success = "true"  # type: ignore[assignment]
    governed = ToolRegistry().govern_result(ToolSpec("write", "Write", risk=Risk.WRITE), result)
    assert not governed.success and governed.data["outcome_unknown"] is True


def test_approval_is_a_barrier_and_completed_reads_are_not_repeated() -> None:
    log: list[str] = []

    async def execute(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        log.append(ctx.tool_call_id)
        return ToolResult.ok(ctx.tool_call_id)

    registry = ToolRegistry()
    for name, risk in [("read", Risk.READ), ("write", Risk.WRITE)]:
        registry.register(
            FunctionTool(ToolSpec(name, name, risk=risk, concurrency_safe=True), execute)
        )
    h = build(
        [
            call_tools(
                tool_call("read", {}, "r1"),
                tool_call("read", {}, "r2"),
                tool_call("write", {}, "w1"),
                tool_call("read", {}, "r3"),
            ),
            reply("done"),
        ],
        registry=registry,
    )
    first = run(h.engine.run(RunRequest(EDITOR, "s1", "Update")))
    assert first.prompt is not None and log == ["r1", "r2"]
    h.broker.resolve(EDITOR, first.prompt.id, {"approved": True})
    second = run(h.engine.resume(EDITOR, first.prompt.id))
    assert second.status is RunStatus.COMPLETED and log == ["r1", "r2", "w1", "r3"]


def test_one_failed_read_does_not_cancel_successful_siblings() -> None:
    async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if ctx.tool_call_id == "bad":
            raise OSError("PRIVATE backend detail")
        return ToolResult.ok("evidence")

    registry = ToolRegistry()
    registry.register(FunctionTool(ToolSpec("read", "Read", concurrency_safe=True), read))
    h = build(
        [call_tools(tool_call("read", {}, "bad"), tool_call("read", {}, "good")), reply("done")],
        registry=registry,
    )
    records = run(h.engine.run(request())).steps[0].tool_calls
    assert not records[0].result.success and records[1].result.success
    assert records[1].result.output == "evidence"


def test_only_governed_results_are_persisted_and_recovered(tmp_path: Path) -> None:
    async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if ctx.tool_call_id == "invalid":
            return ToolResult(True, "PRIVATE_INVALID", data={"not_json": object()})
        return ToolResult.ok("HEAD" + "x" * 200 + "TAIL")

    registry = ToolRegistry(max_output_chars=16)
    registry.register(FunctionTool(ToolSpec("read", "Read", concurrency_safe=True), read))
    model = ScriptedModel(
        [
            call_tools(tool_call("read", {}, "invalid"), tool_call("read", {}, "large")),
            reply("done"),
        ]
    )
    engine = AgentEngine(model, registry, run_store=JsonlRunStore(tmp_path))
    first = run(engine.run(RunRequest(EDITOR, "s1", "Read", run_id="governed")))
    fresh_model = ScriptedModel([])
    fresh = AgentEngine(fresh_model, registry, run_store=JsonlRunStore(tmp_path))
    second = run(fresh.recover(EDITOR, "governed"))
    assert first.steps == second.steps and not fresh_model.calls
    records = second.steps[0].tool_calls
    assert not records[0].result.success and records[0].result.output == ""
    assert records[1].result.output.startswith("HEAD") and records[1].result.output.endswith("TAIL")
    for path in tmp_path.rglob("*.jsonl"):
        assert "PRIVATE_INVALID" not in path.read_text()


def test_fatal_worker_failure_stops_dequeuing_before_sibling_finishes() -> None:
    class FatalFailure(BaseException):
        pass

    async def scenario() -> None:
        sibling_started = asyncio.Event()
        release_sibling = asyncio.Event()
        started: list[str] = []
        stopped: list[str] = []

        async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            started.append(ctx.tool_call_id)
            if ctx.tool_call_id == "c0":
                await sibling_started.wait()
                release_sibling.set()
                raise FatalFailure()
            if ctx.tool_call_id == "c1":
                sibling_started.set()
                try:
                    await release_sibling.wait()
                finally:
                    stopped.append(ctx.tool_call_id)
            return ToolResult.ok("done")

        registry = ToolRegistry()
        registry.register(FunctionTool(ToolSpec("read", "Read", concurrency_safe=True), read))
        h = build(
            [call_tools(*(tool_call("read", {}, f"c{i}") for i in range(5)))],
            registry=registry,
            config=AgentConfig(max_parallel_tools=2),
        )
        with pytest.raises(FatalFailure):
            await h.engine.run(request())
        assert started == ["c0", "c1"] and stopped == ["c1"]

    run(scenario())
