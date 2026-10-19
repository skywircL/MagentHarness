"""Review regressions at the actual overflow, authority and journal boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from support import EDITOR, build, run

from magent_harness import (
    AgentConfig,
    AgentEngine,
    ContextOverflowError,
    FunctionTool,
    JsonlRunStore,
    MemoryKind,
    MemoryService,
    Message,
    ModelError,
    PersistenceError,
    PromptStatus,
    Risk,
    RunRequest,
    RunSnapshot,
    RunStatus,
    SnapshotError,
    TokenEstimator,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    search_memory_tool,
)
from magent_harness.runs import RunLog, RunState
from magent_harness.testing import ScriptedModel, call_tools, reply, tool_call

RUN_ID = "overflow-authority-review"
SESSION_ID = "overflow-session"


class ProcessCrash(BaseException):
    """Interrupt after a committed journal fact without ordinary cleanup."""


def memory_fixture() -> tuple[MemoryService, ToolRegistry, AgentConfig]:
    memory = MemoryService()
    for number in range(20):
        memory.remember(EDITOR, MemoryKind.FACT, f"cobalt {number} " + "文" * 450)
    registry = ToolRegistry()
    registry.register(search_memory_tool(memory))
    config = AgentConfig(interactive=False, system_prompt="small system")
    return memory, registry, config


def initial_script() -> list[Any]:
    return [
        call_tools(tool_call("search_memory", {"query": "cobalt", "limit": 20}, "search")),
        ContextOverflowError("provider reported an overflow"),
        reply("done"),
    ]


def assert_paired_search(messages: list[Message]) -> None:
    assistant = next(message for message in messages if message.tool_calls)
    assert [(call.id, call.name) for call in assistant.tool_calls] == [("search", "search_memory")]
    tools = [message for message in messages if message.role == "tool"]
    assert [(message.tool_call_id, message.name) for message in tools] == [
        ("search", "search_memory")
    ]
    assert "tool result trimmed" in tools[0].content


def test_recover_after_forced_context_commit_does_not_resend_the_rejected_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory, registry, config = memory_fixture()
    model = ScriptedModel(initial_script())
    store = JsonlRunStore(tmp_path)
    first = AgentEngine(
        model,
        registry,
        memory=memory,
        config=config,
        run_store=store,
        summarizer=ScriptedModel([reply("short summary")]),
    )
    request = RunRequest(EDITOR, SESSION_ID, "query cobalt", run_id=RUN_ID)
    journal_type = type(store.journal(EDITOR, RUN_ID))
    append = journal_type.append

    def crash_after_shrink(journal: Any, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        record = append(journal, kind, data)
        if kind == "context_committed" and any(
            message.get("role") == "tool" and "tool result trimmed" in message.get("content", "")
            for message in data["messages"]
        ):
            raise ProcessCrash("forced context committed before the retry")
        return record

    with monkeypatch.context() as patch:
        patch.setattr(journal_type, "append", crash_after_shrink)
        with pytest.raises(ProcessCrash):
            run(first.run(request))
    assert len(model.calls) == 2
    rejected = model.calls[-1]
    with store.journal(EDITOR, RUN_ID) as journal:
        log = RunLog(journal)
        assert log.state is not None
        assert isinstance(log.state.context_input_limit, int)
        assert log.state.context_input_limit > 0
    fresh_model = ScriptedModel([reply("recovered")])
    fresh = AgentEngine(
        fresh_model,
        registry,
        memory=memory,
        config=config,
        run_store=JsonlRunStore(tmp_path),
        summarizer=ScriptedModel([reply("short summary")]),
    )

    result = run(fresh.recover(EDITOR, RUN_ID, session_id=SESSION_ID))

    assert result.status is RunStatus.COMPLETED
    assert result.final_answer == "recovered"
    assert len(fresh_model.calls) == 1

    actual = fresh_model.calls[0]
    estimator = TokenEstimator()
    assert estimator.request(actual.messages, actual.tools) < (
        estimator.request(rejected.messages, rejected.tools) // 2
    )
    assert actual.messages != rejected.messages
    assert_paired_search(actual.messages)
    assert len(result.steps) == 1
    assert len(result.steps[0].tool_calls) == 1
    # Reopening the terminal journal performs no more provider requests.
    assert run(fresh.recover(EDITOR, RUN_ID, session_id=SESSION_ID)) == result
    assert len(fresh_model.calls) == 1


@pytest.mark.parametrize(
    "from_history", [False, True], ids=["current-run", "cross-session-history"]
)
def test_plain_approval_resume_keeps_the_previously_reduced_memory_input(
    from_history: bool,
) -> None:
    memory, registry, config = memory_fixture()
    writes: list[str] = []

    async def write(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        writes.append(ctx.tool_call_id)
        return ToolResult.ok("write committed")

    registry.register(FunctionTool(ToolSpec("write", "write", risk=Risk.WRITE), write))
    config = replace(config, interactive=True)
    history = []
    script = initial_script()[:2]
    if from_history:
        search = tool_call("search_memory", {"query": "cobalt", "limit": 20}, "old-search")
        history = [
            Message("user", "previous session cobalt request"),
            Message("assistant", tool_calls=(search,)),
            Message("tool", "old memory", tool_call_id=search.id, name=search.name),
            Message("assistant", "previous session request completed"),
        ]
        script = [ContextOverflowError("provider reported an overflow")]
    model = ScriptedModel([*script, call_tools(tool_call("write", {}, "approved-write"))])
    first = AgentEngine(
        model,
        registry,
        memory=memory,
        config=config,
        summarizer=ScriptedModel([ModelError("summary unavailable")]),
    )
    pending = run(first.run(RunRequest(EDITOR, SESSION_ID, "query cobalt", history=history))).prompt
    assert pending is not None
    assert writes == []
    assert len(model.calls) == (2 if from_history else 3)
    snapshot = RunSnapshot.decode(pending.snapshot)
    assert snapshot.context_input_limit is not None
    assert snapshot.context_input_limit > 0
    first.broker.resolve(EDITOR, pending.id, {"approved": True}, session_id=SESSION_ID)
    fresh_model = ScriptedModel([reply("resumed")])
    fresh = AgentEngine(
        fresh_model,
        registry,
        broker=first.broker,
        memory=memory,
        config=config,
        summarizer=ScriptedModel([reply("short summary")]),
    )

    result = run(fresh.resume(EDITOR, pending.id, session_id=SESSION_ID))

    assert result.status is RunStatus.COMPLETED
    assert result.final_answer == "resumed"
    assert writes == ["approved-write"]
    assert len(fresh_model.calls) == 1
    estimator = TokenEstimator()
    rejected, resumed = model.calls[0 if from_history else 1], fresh_model.calls[0]
    assert estimator.request(resumed.messages, resumed.tools) < (
        estimator.request(rejected.messages, rejected.tools) // 2
    )
    assert len(result.steps) == (1 if from_history else 2)
    if not from_history:
        assert len(result.steps[0].tool_calls[0].result.output) > 9_000


def suspended_snapshot() -> tuple[Any, Any, dict[str, Any]]:
    harness = build(
        [call_tools(tool_call("update_document", {"document_id": "d1", "content": "new"}))]
    )
    prompt = run(harness.engine.run(RunRequest(EDITOR, "metadata-session", "update d1"))).prompt
    assert prompt is not None
    return harness, prompt, json.loads(prompt.snapshot)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "100", {}, []])
def test_invalid_context_limit_is_rejected_before_an_approved_tool_runs(invalid: Any) -> None:
    harness, prompt, data = suspended_snapshot()
    data["context_input_limit"] = invalid
    malformed = json.dumps(data)
    with pytest.raises(SnapshotError, match="context_input_limit"):
        RunSnapshot.decode(malformed)
    with pytest.raises(SnapshotError, match="context_input_limit"):
        RunState.from_core({**data, "version": 1})
    harness.broker.store.transition(prompt.id, [PromptStatus.SUSPENDED], snapshot=malformed)
    harness.broker.resolve(EDITOR, prompt.id, {"approved": True})

    with pytest.raises(SnapshotError, match="context_input_limit"):
        run(harness.engine.resume(EDITOR, prompt.id, session_id="metadata-session"))

    assert harness.workspace.writes == []
    assert len(harness.model.calls) == 1


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_snapshot_without_a_context_limit_remains_readable(version: int) -> None:
    _, _, data = suspended_snapshot()
    data["version"] = version
    data.pop("context_input_limit", None)

    snapshot = RunSnapshot.decode(json.dumps(data))
    state = RunState.from_core({**data, "version": 1})

    assert snapshot.version == 2
    assert snapshot.context_input_limit is None
    assert state.context_input_limit is None
    assert RunSnapshot.decode(snapshot.encode()).context_input_limit is None
    assert RunState.from_core(state.core()).context_input_limit is None
    data["context_input_limit"] = 1234
    cached = RunSnapshot.decode(json.dumps(data))
    assert RunSnapshot.decode(cached.encode()).context_input_limit == 1234
    assert RunState.from_core({**data, "version": 1}).core()["context_input_limit"] == 1234


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "100", {}, []])
def test_invalid_context_commit_limit_does_not_modify_the_journal(
    tmp_path: Path, invalid: Any
) -> None:
    model = ScriptedModel([ProcessCrash()])
    store = JsonlRunStore(tmp_path)
    engine = AgentEngine(model, ToolRegistry(), run_store=store)
    with pytest.raises(ProcessCrash):
        run(engine.run(RunRequest(EDITOR, SESSION_ID, "current request", run_id=RUN_ID)))
    with store.journal(EDITOR, RUN_ID) as journal:
        log = RunLog(journal)
        assert log.state is not None
        before = log.state.core()
        count = len(journal.records)

        with pytest.raises(PersistenceError, match="context_committed"):
            log.append(
                "context_committed",
                {"messages": [], "context_input_limit": invalid},
            )

        assert log.state.core() == before
        assert len(journal.records) == count
        assert RunLog(journal).state.core() == before


def test_context_commit_atomically_preserves_the_optional_limit_cache(tmp_path: Path) -> None:
    model = ScriptedModel([ProcessCrash()])
    store = JsonlRunStore(tmp_path)
    engine = AgentEngine(model, ToolRegistry(), run_store=store)
    with pytest.raises(ProcessCrash):
        run(engine.run(RunRequest(EDITOR, SESSION_ID, "current request", run_id=RUN_ID)))
    with store.journal(EDITOR, RUN_ID) as journal:
        log = RunLog(journal)
        assert log.state is not None
        messages = log.state.core()["messages"]
        log.append("context_committed", {"messages": messages, "context_input_limit": 1234})
        assert log.state.context_input_limit == 1234
        assert log.state.core()["messages"] == messages
        log.append("context_committed", {"messages": messages})
        assert log.state.context_input_limit == 1234
        log.append("progress", {"state": log.state.core()})
        assert log.state.context_input_limit == 1234
        restored = RunLog(journal).state
        assert restored is not None
        assert restored.context_input_limit == 1234
        assert restored.messages == log.state.messages
