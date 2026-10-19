"""Durable recovery through real engine entries and interrupted journal commits."""

from __future__ import annotations

import asyncio
import json
import select
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from support import EDITOR, OTHER_WORKSPACE, FakeWorkspace, run

from magent_harness import (
    AccessLevel,
    AgentConfig,
    AgentEngine,
    Caller,
    ContextSettings,
    FunctionTool,
    JsonlRunStore,
    MemoryKind,
    MemoryService,
    Message,
    PersistenceError,
    PromptStatus,
    RecoveryUnsafe,
    ResourceScope,
    Role,
    RunAlreadyExists,
    RunNotFound,
    RunRequest,
    RunStatus,
    StoreBusy,
    ToolContext,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from magent_harness.messages import KIND_COMPACTION_SUMMARY
from magent_harness.model import FINISH_LENGTH, ChatResponse
from magent_harness.testing import ScriptedModel, call_tools, reply, tool_call

RUN_ID = "durable-run"
SESSION_ID = "durable-session"
AUTO = ToolPolicy(AccessLevel.AUTO_WRITE)


class ProcessCrash(BaseException):
    """Bypass ordinary tool/model error handling, as process death would."""


def request(**kwargs: Any) -> RunRequest:
    return RunRequest(
        EDITOR,
        SESSION_ID,
        "Read the documents and finish the task",
        run_id=RUN_ID,
        **kwargs,
    )


def engine(
    path: Path,
    script: list[Any],
    workspace: FakeWorkspace,
    **kwargs: Any,
) -> tuple[AgentEngine, ScriptedModel, JsonlRunStore]:
    model = ScriptedModel(script)
    store = JsonlRunStore(path)
    instance = AgentEngine(
        model,
        workspace.registry(),
        run_store=store,
        config=kwargs.pop("config", AgentConfig(model_retry_delay_s=0)),
        **kwargs,
    )
    return instance, model, store


def interrupt_append(
    monkeypatch: pytest.MonkeyPatch,
    store: JsonlRunStore,
    kind: str,
    *,
    after: bool = False,
) -> None:
    """Interrupt once at an actual persistence boundary, not an engine stub."""
    journal_type = type(store.journal(EDITOR, RUN_ID))
    original = journal_type.append
    armed = True

    def append(journal: Any, event_kind: str, data: dict[str, Any]) -> dict[str, Any]:
        nonlocal armed
        matches = armed and event_kind == kind
        if matches:
            armed = False
            if not after:
                raise ProcessCrash(kind)
        record = original(journal, event_kind, data)
        if matches:
            raise ProcessCrash(kind)
        return record

    monkeypatch.setattr(journal_type, "append", append)


def read_call(document_id: str = "d1", call_id: str = "r1") -> Any:
    return tool_call("read_document", {"document_id": document_id}, call_id)


def write_call() -> Any:
    return tool_call("update_document", {"document_id": "d1", "content": "v2"}, "w1")


def test_completed_run_reopens_without_model_or_tool_calls(tmp_path: Path) -> None:
    workspace = FakeWorkspace.default()
    first, _, _ = engine(tmp_path, [call_tools(read_call()), reply("durable answer")], workspace)
    expected = run(first.run(request()))

    fresh, model, _ = engine(tmp_path, [], workspace)
    actual = run(fresh.recover(EDITOR, RUN_ID, session_id=SESSION_ID))

    assert actual == expected
    assert actual.status is RunStatus.COMPLETED
    assert workspace.reads == ["d1"]
    assert model.calls == []
    with pytest.raises(RunAlreadyExists):
        run(fresh.run(request()))
    assert model.calls == []


def test_crash_before_plan_repeats_model_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    first, original_model, store = engine(tmp_path, [call_tools(read_call())], workspace)
    with monkeypatch.context() as patch:
        interrupt_append(patch, store, "planned")
        with pytest.raises(ProcessCrash, match="planned"):
            run(first.run(request()))
    assert len(original_model.calls) == 1
    assert workspace.reads == []

    fresh, model, _ = engine(tmp_path, [call_tools(read_call()), reply("finished")], workspace)
    result = run(fresh.recover(EDITOR, RUN_ID))
    assert result.final_answer == "finished"
    assert workspace.reads == ["d1"]
    assert len(model.calls) == 2


def test_persisted_plan_continues_without_regenerating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    first, _, store = engine(tmp_path, [call_tools(read_call(), content="Read d1")], workspace)
    with monkeypatch.context() as patch:
        interrupt_append(patch, store, "planned", after=True)
        with pytest.raises(ProcessCrash):
            run(first.run(request()))

    fresh, model, _ = engine(tmp_path, [reply("finished")], workspace)
    result = run(fresh.recover(EDITOR, RUN_ID))
    assert workspace.reads == ["d1"]
    assert len(model.calls) == 1
    assert result.steps[0].thought == "Read d1"
    assert result.steps[0].tool_calls[0].id == "r1"
    assert model.calls[0].messages[-1].tool_call_id == "r1"


def test_started_read_without_result_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    first, _, store = engine(tmp_path, [call_tools(read_call())], workspace)
    with monkeypatch.context() as patch:
        # The read actually returned, but its result never committed.
        interrupt_append(patch, store, "tool_finished")
        with pytest.raises(ProcessCrash):
            run(first.run(request()))
    assert workspace.reads == ["d1"]

    fresh, model, _ = engine(tmp_path, [reply("finished")], workspace)
    result = run(fresh.recover(EDITOR, RUN_ID))
    assert result.status is RunStatus.COMPLETED
    assert workspace.reads == ["d1", "d1"]
    assert len(model.calls) == 1
    assert len(result.steps[0].tool_calls) == 1


@pytest.mark.parametrize(
    ("kind", "after", "expected_writes"),
    [("tool_started", True, []), ("tool_finished", False, [("d1", "v2")])],
)
def test_unknown_write_outcome_is_never_reexecuted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    after: bool,
    expected_writes: list[tuple[str, str]],
) -> None:
    workspace = FakeWorkspace.default()
    first, _, store = engine(tmp_path, [call_tools(write_call())], workspace)
    with monkeypatch.context() as patch:
        interrupt_append(patch, store, kind, after=after)
        with pytest.raises(ProcessCrash):
            run(first.run(request(policy=AUTO)))
    assert workspace.writes == expected_writes

    fresh, model, _ = engine(tmp_path, [], workspace)
    with pytest.raises(RecoveryUnsafe):
        run(fresh.recover(EDITOR, RUN_ID))
    assert workspace.writes == expected_writes
    assert model.calls == []


def test_committed_tool_result_is_reused_before_round_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    first, _, store = engine(tmp_path, [call_tools(write_call())], workspace)
    with monkeypatch.context() as patch:
        interrupt_append(patch, store, "tool_finished", after=True)
        with pytest.raises(ProcessCrash):
            run(first.run(request(policy=AUTO)))
    assert workspace.writes == [("d1", "v2")]

    fresh, model, _ = engine(tmp_path, [reply("saved once")], workspace)
    result = run(fresh.recover(EDITOR, RUN_ID))
    assert result.final_answer == "saved once"
    assert workspace.writes == [("d1", "v2")]
    assert len(result.steps[0].tool_calls) == 1
    assert result.steps[0].tool_calls[0].result.success
    assert model.calls[0].messages[-1].tool_call_id == "w1"


def test_parallel_results_recover_in_model_call_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    completion_order: list[str] = []
    second_finished = asyncio.Event()

    async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        document_id = args["document_id"]
        if document_id == "d1":
            await second_finished.wait()
        else:
            second_finished.set()
        completion_order.append(document_id)
        return ToolResult.ok(document_id)

    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            ToolSpec(
                "read_document",
                "Read a document.",
                {
                    "type": "object",
                    "properties": {"document_id": {"type": "string"}},
                    "required": ["document_id"],
                },
                concurrency_safe=True,
            ),
            read,
        )
    )
    first, _, store = engine(tmp_path, [call_tools(read_call(), read_call("d2", "r2"))], workspace)
    first.registry = registry
    with monkeypatch.context() as patch:
        interrupt_append(patch, store, "round_committed")
        with pytest.raises(ProcessCrash):
            run(first.run(request()))
    assert completion_order == ["d2", "d1"]

    fresh, model, _ = engine(tmp_path, [reply("ordered")], workspace)
    fresh.registry = registry
    result = run(fresh.recover(EDITOR, RUN_ID))
    assert completion_order == ["d2", "d1"]
    assert [record.id for record in result.steps[0].tool_calls] == ["r1", "r2"]
    tools = [message for message in model.calls[0].messages if message.role == "tool"]
    assert [message.tool_call_id for message in tools] == ["r1", "r2"]
    assert [message.content for message in tools] == ["d1", "d2"]


def test_human_approval_survives_new_store_and_engine(tmp_path: Path) -> None:
    workspace = FakeWorkspace.default()
    first, _, _ = engine(tmp_path, [call_tools(write_call())], workspace)
    suspended = run(first.run(request()))
    assert suspended.prompt is not None
    prompt_id = suspended.prompt.id
    assert workspace.writes == []

    fresh, _, _ = engine(tmp_path, [reply("approved and saved")], workspace)
    prompt = fresh.broker.resolve(EDITOR, prompt_id, {"approved": True}, session_id=SESSION_ID)
    assert prompt.status is PromptStatus.ANSWERED
    result = run(fresh.resume(EDITOR, prompt_id, session_id=SESSION_ID))
    assert result.final_answer == "approved and saved"
    assert workspace.writes == [("d1", "v2")]

    reader, model, _ = engine(tmp_path, [], workspace)
    assert run(reader.recover(EDITOR, RUN_ID)) == result
    assert model.calls == []


def test_persisted_approval_rechecks_current_role(tmp_path: Path) -> None:
    workspace = FakeWorkspace.default()
    first, _, _ = engine(tmp_path, [call_tools(write_call())], workspace)
    suspended = run(first.run(request()))
    assert suspended.prompt is not None

    fresh, model, _ = engine(tmp_path, [reply("permission was withdrawn")], workspace)
    fresh.broker.resolve(EDITOR, suspended.prompt.id, {"approved": True})
    demoted = Caller(EDITOR.workspace_id, EDITOR.user_id, Role.VIEWER)
    result = run(fresh.resume(demoted, suspended.prompt.id))

    assert workspace.writes == []
    assert "not permitted" in result.steps[0].tool_calls[0].result.error
    assert "update_document" not in model.calls[0].tool_names
    stored_prompt = fresh.broker.store.get(suspended.prompt.id)
    assert stored_prompt is not None and stored_prompt.status is PromptStatus.FAILED


def test_persisted_approval_does_not_restore_forgotten_memory(tmp_path: Path) -> None:
    workspace = FakeWorkspace.default()
    memory = MemoryService()
    forgotten = memory.remember(EDITOR, MemoryKind.PREFERENCE, "Use the unique cobalt format")
    first, original_model, _ = engine(
        tmp_path, [call_tools(write_call())], workspace, memory=memory
    )
    suspended = run(first.run(request()))
    assert suspended.prompt is not None
    assert "unique cobalt" in original_model.calls[0].messages[0].content
    assert memory.forget(EDITOR, forgotten.id)

    fresh, model, _ = engine(tmp_path, [reply("approved")], workspace, memory=memory)
    fresh.broker.resolve(EDITOR, suspended.prompt.id, {"approved": True})
    run(fresh.resume(EDITOR, suspended.prompt.id))

    assert workspace.writes == [("d1", "v2")]
    assert all("unique cobalt" not in message.content for message in model.calls[0].messages)


@pytest.mark.parametrize(
    "caller",
    [Caller(EDITOR.workspace_id, "mallory", Role.OWNER), OTHER_WORKSPACE],
)
def test_recovery_hides_runs_from_other_identities(tmp_path: Path, caller: Caller) -> None:
    workspace = FakeWorkspace.default()
    first, _, _ = engine(tmp_path, [reply("private answer")], workspace)
    run(first.run(request()))
    fresh, model, _ = engine(tmp_path, [], workspace)
    with pytest.raises(RunNotFound):
        run(fresh.recover(caller, RUN_ID))
    assert model.calls == []


def test_recovery_rejects_wrong_session(tmp_path: Path) -> None:
    workspace = FakeWorkspace.default()
    first, _, _ = engine(tmp_path, [reply("private answer")], workspace)
    run(first.run(request()))
    fresh, model, _ = engine(tmp_path, [], workspace)
    with pytest.raises(RunNotFound):
        run(fresh.recover(EDITOR, RUN_ID, session_id="another-session"))
    assert model.calls == []


def test_recovery_keeps_original_document_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    first, _, store = engine(tmp_path, [call_tools(read_call(), read_call("d2", "r2"))], workspace)
    with monkeypatch.context() as patch:
        interrupt_append(patch, store, "planned", after=True)
        with pytest.raises(ProcessCrash):
            run(first.run(request(scope=ResourceScope("ws1", frozenset({"d1"})))))

    fresh, _, _ = engine(tmp_path, [reply("only d1 was accessible")], workspace)
    result = run(fresh.recover(EDITOR, RUN_ID))
    assert workspace.reads == ["d1"]
    assert result.steps[0].tool_calls[0].result.success
    assert "scope" in result.steps[0].tool_calls[1].result.error


def test_recovery_reevaluates_current_role_before_planned_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    first, _, store = engine(tmp_path, [call_tools(write_call())], workspace)
    with monkeypatch.context() as patch:
        interrupt_append(patch, store, "planned", after=True)
        with pytest.raises(ProcessCrash):
            run(first.run(request(policy=AUTO)))

    fresh, model, _ = engine(tmp_path, [reply("permission was withdrawn")], workspace)
    demoted = Caller(EDITOR.workspace_id, EDITOR.user_id, Role.VIEWER)
    result = run(fresh.recover(demoted, RUN_ID))
    assert workspace.writes == []
    assert "not permitted" in result.steps[0].tool_calls[0].result.error
    assert "update_document" not in model.calls[0].tool_names


def test_recovery_does_not_reinject_forgotten_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    memory = MemoryService()
    forgotten = memory.remember(EDITOR, MemoryKind.PREFERENCE, "Use the unique cobalt format")
    first, original_model, store = engine(
        tmp_path, [call_tools(read_call())], workspace, memory=memory
    )
    with monkeypatch.context() as patch:
        interrupt_append(patch, store, "planned", after=True)
        with pytest.raises(ProcessCrash):
            run(first.run(request()))
    assert "unique cobalt" in original_model.calls[0].messages[0].content
    assert memory.forget(EDITOR, forgotten.id)

    fresh, model, _ = engine(tmp_path, [reply("finished")], workspace, memory=memory)
    run(fresh.recover(EDITOR, RUN_ID))
    assert all("unique cobalt" not in message.content for message in model.calls[0].messages)


@pytest.mark.parametrize("after", [False, True])
def test_compaction_commit_is_recoverable_on_both_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after: bool
) -> None:
    workspace = FakeWorkspace.default()
    config = AgentConfig(
        context=ContextSettings(
            max_context_tokens=2_000, reserve_tokens=500, keep_recent_tokens=400
        ),
        model_retry_delay_s=0,
    )
    summarizer = ScriptedModel([reply("Durable summary: inspect the roadmap next")])
    first, _, store = engine(tmp_path, [], workspace, config=config, summarizer=summarizer)
    history = [Message("user" if i % 2 == 0 else "assistant", "history " * 500) for i in range(4)]
    with monkeypatch.context() as patch:
        interrupt_append(patch, store, "context_committed", after=after)
        with pytest.raises(ProcessCrash):
            run(first.run(request(history=history)))
    assert len(summarizer.calls) == 1

    retry_summary = ScriptedModel([reply("Durable summary: inspect the roadmap next")])
    fresh, model, _ = engine(
        tmp_path,
        [call_tools(read_call()), reply("continued reading")],
        workspace,
        config=config,
        summarizer=retry_summary,
    )
    result = run(fresh.recover(EDITOR, RUN_ID))
    assert result.final_answer == "continued reading"
    assert workspace.reads == ["d1"]
    assert len(retry_summary.calls) == (0 if after else 1)
    summaries = [
        message for message in model.calls[0].messages if message.kind == KIND_COMPACTION_SUMMARY
    ]
    assert len(summaries) == 1
    assert "inspect the roadmap" in summaries[0].content
    assert "Read the documents and finish the task" in "\n".join(
        message.content for message in model.calls[0].messages
    )


def test_actual_process_exit_recovers_in_another_python_process(tmp_path: Path) -> None:
    child = textwrap.dedent(
        """
        import asyncio
        import os
        import sys
        from pathlib import Path
        from magent_harness import AgentEngine, Caller, JsonlRunStore, Role, RunRequest

        class InterruptedModel:
            async def complete(self, messages, tools, *, max_tokens=None):
                Path(sys.argv[1], 'model-entered').write_text('yes')
                os._exit(73)

        caller = Caller('ws1', 'alice', Role.EDITOR)
        engine = AgentEngine(InterruptedModel(), run_store=JsonlRunStore(Path(sys.argv[1])))
        asyncio.run(engine.run(RunRequest(
            caller, 'durable-session', 'Continue after the process dies', run_id='durable-run'
        )))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 73, completed.stderr
    assert (tmp_path / "model-entered").read_text() == "yes"

    workspace = FakeWorkspace.default()
    fresh, model, _ = engine(tmp_path, [reply("resumed after process exit")], workspace)
    result = run(fresh.recover(EDITOR, RUN_ID, session_id=SESSION_ID))
    assert result.final_answer == "resumed after process exit"
    assert result.status is RunStatus.COMPLETED
    assert len(model.calls) == 1
    assert model.calls[0].messages[-1].content == "Continue after the process dies"


def test_active_engine_excludes_recovery_in_another_process(tmp_path: Path) -> None:
    child = textwrap.dedent(
        """
        import asyncio
        import sys
        from pathlib import Path
        from magent_harness import AgentEngine, Caller, JsonlRunStore, Role, RunRequest

        class WaitingModel:
            async def complete(self, messages, tools, *, max_tokens=None):
                print('model-awaiting', flush=True)
                await asyncio.Event().wait()

        caller = Caller('ws1', 'alice', Role.EDITOR)
        engine = AgentEngine(WaitingModel(), run_store=JsonlRunStore(Path(sys.argv[1])))
        asyncio.run(engine.run(RunRequest(
            caller, 'durable-session', 'Keep running', run_id='durable-run'
        )))
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 10)
        assert ready, "child never entered the model await"
        assert process.stdout.readline().strip() == "model-awaiting"
        workspace = FakeWorkspace.default()
        fresh, model, _ = engine(tmp_path, [reply("recovered after exit")], workspace)
        with pytest.raises(StoreBusy):
            run(fresh.recover(EDITOR, RUN_ID))
        assert model.calls == []
        assert workspace.reads == []
    finally:
        process.kill()
        process.communicate(timeout=10)

    # Process death releases the lock even though no Python finally block ran.
    result = run(fresh.recover(EDITOR, RUN_ID))
    assert result.final_answer == "recovered after exit"
    assert len(model.calls) == 1


def test_committed_middle_line_corruption_prevents_any_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    first, _, store = engine(tmp_path, [call_tools(read_call())], workspace)
    with monkeypatch.context() as patch:
        interrupt_append(patch, store, "tool_started", after=True)
        with pytest.raises(ProcessCrash):
            run(first.run(request()))
    assert workspace.reads == []
    path = store.journal(EDITOR, RUN_ID).path
    lines = path.read_bytes().splitlines(keepends=True)
    assert len(lines) >= 3
    lines[1] = b'{"corrupted-committed-line":true}\n'
    corrupted = b"".join(lines)
    path.write_bytes(corrupted)

    fresh, model, _ = engine(tmp_path, [], workspace)
    with pytest.raises(PersistenceError):
        run(fresh.recover(EDITOR, RUN_ID))
    assert workspace.reads == []
    assert model.calls == []
    assert path.read_bytes() == corrupted


@pytest.mark.parametrize("damage", ["missing", "broken", "forged"])
def test_checkpoint_cannot_override_authoritative_run_facts(tmp_path: Path, damage: str) -> None:
    workspace = FakeWorkspace.default()
    first, _, store = engine(tmp_path, [call_tools(write_call()), reply("real answer")], workspace)
    expected = run(first.run(request(policy=AUTO)))
    journal = store.journal(EDITOR, RUN_ID)
    original = json.loads(journal.checkpoint_path.read_text())
    authority = journal.path.read_bytes()
    if damage == "missing":
        journal.checkpoint_path.unlink()
    elif damage == "broken":
        journal.checkpoint_path.write_text("{torn checkpoint")
    else:
        forged = json.loads(json.dumps(original))
        forged["data"]["answer"] = "forged cached answer"
        # The forged projection deliberately retains the genuine sequence/hash.
        assert forged["seq"] == original["seq"]
        assert forged["digest"] == original["digest"]
        journal.checkpoint_path.write_text(json.dumps(forged))

    fresh, model, _ = engine(tmp_path, [], workspace)
    actual = run(fresh.recover(EDITOR, RUN_ID))
    assert actual == expected
    assert workspace.writes == [("d1", "v2")]
    assert model.calls == []
    assert journal.path.read_bytes() == authority
    assert json.loads(journal.checkpoint_path.read_text()) == original


def test_truncated_tool_plan_never_executes_after_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    response = ChatResponse(tool_calls=[write_call()], finish_reason=FINISH_LENGTH)
    first, _, store = engine(tmp_path, [response], workspace)
    with monkeypatch.context() as patch:
        interrupt_append(patch, store, "planned", after=True)
        with pytest.raises(ProcessCrash):
            run(first.run(request(policy=AUTO)))

    fresh, model, _ = engine(tmp_path, [reply("the write was not executed")], workspace)
    result = run(fresh.recover(EDITOR, RUN_ID))
    assert workspace.writes == []
    assert result.status is RunStatus.COMPLETED
    record = result.steps[0].tool_calls[0]
    assert not record.result.success
    assert "cut off" in record.result.error
    assert len(model.calls) == 1


def test_suspension_wal_recreates_missing_prompt_with_the_original_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    first, _, store = engine(tmp_path, [call_tools(write_call())], workspace)
    with monkeypatch.context() as patch:
        # The run knows the prompt; PromptStore.create has not happened yet.
        interrupt_append(patch, store, "suspended", after=True)
        with pytest.raises(ProcessCrash):
            run(first.run(request()))
    with store.journal(EDITOR, RUN_ID) as journal:
        saved = journal.records[-1]["data"]["prompt"]
    assert first.broker.store.get(saved["id"]) is None

    fresh, model, _ = engine(tmp_path, [reply("saved")], workspace)
    result = run(fresh.recover(EDITOR, RUN_ID))
    assert result.status is RunStatus.SUSPENDED
    assert result.prompt is not None and result.prompt.id == saved["id"]
    assert fresh.broker.store.get(saved["id"]) == result.prompt
    assert model.calls == []
    fresh.broker.resolve(EDITOR, saved["id"], {"approved": True})
    resumed = run(fresh.resume(EDITOR, saved["id"]))
    assert resumed.final_answer == "saved"
    assert workspace.writes == [("d1", "v2")]


@pytest.mark.parametrize("after_claim", [False, True])
def test_answered_prompt_recovers_around_claim_before_tool_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_claim: bool
) -> None:
    workspace = FakeWorkspace.default()
    first, _, store = engine(tmp_path, [call_tools(write_call())], workspace)
    result = run(first.run(request()))
    assert result.prompt is not None
    prompt_id = result.prompt.id
    first.broker.resolve(EDITOR, prompt_id, {"approved": True})

    with monkeypatch.context() as patch:
        if after_claim:
            interrupt_append(patch, store, "tool_started")
        else:
            interrupt_append(patch, store, "resume_requested", after=True)
        with pytest.raises(ProcessCrash):
            run(first.resume(EDITOR, prompt_id))
    stored = first.broker.store.get(prompt_id)
    assert stored is not None
    assert stored.status is (PromptStatus.EXECUTING if after_claim else PromptStatus.ANSWERED)
    assert workspace.writes == []

    fresh, model, _ = engine(tmp_path, [reply("saved after claim recovery")], workspace)
    recovered = run(fresh.recover(EDITOR, RUN_ID))
    assert recovered.final_answer == "saved after claim recovery"
    assert workspace.writes == [("d1", "v2")]
    assert len(model.calls) == 1
    stored = fresh.broker.store.get(prompt_id)
    assert stored is not None and stored.status is PromptStatus.DONE


def test_committed_approved_write_closes_prompt_without_reexecuting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    first, _, store = engine(tmp_path, [call_tools(write_call())], workspace)
    result = run(first.run(request()))
    assert result.prompt is not None
    prompt_id = result.prompt.id
    first.broker.resolve(EDITOR, prompt_id, {"approved": True})
    with monkeypatch.context() as patch:
        interrupt_append(patch, store, "tool_finished", after=True)
        with pytest.raises(ProcessCrash):
            run(first.resume(EDITOR, prompt_id))
    assert workspace.writes == [("d1", "v2")]
    prompt = first.broker.store.get(prompt_id)
    assert prompt is not None and prompt.status is PromptStatus.EXECUTING

    fresh, model, _ = engine(tmp_path, [reply("saved exactly once")], workspace)
    recovered = run(fresh.recover(EDITOR, RUN_ID))
    assert recovered.final_answer == "saved exactly once"
    assert workspace.writes == [("d1", "v2")]
    assert len(model.calls) == 1
    prompt = fresh.broker.store.get(prompt_id)
    assert prompt is not None and prompt.status is PromptStatus.DONE


def test_multiple_approvals_in_one_round_survive_restarts(tmp_path: Path) -> None:
    workspace = FakeWorkspace.default()
    second_call = tool_call("update_document", {"document_id": "d2", "content": "v3"}, "w2")
    first, _, _ = engine(tmp_path, [call_tools(write_call(), second_call)], workspace)
    suspended = run(first.run(request()))
    assert suspended.prompt is not None and suspended.prompt.tool_call_id == "w1"

    second, second_model, _ = engine(tmp_path, [], workspace)
    second.broker.resolve(EDITOR, suspended.prompt.id, {"approved": True})
    suspended_again = run(second.recover(EDITOR, RUN_ID))
    assert suspended_again.prompt is not None
    assert suspended_again.prompt.tool_call_id == "w2"
    assert suspended_again.prompt.id != suspended.prompt.id
    assert workspace.writes == [("d1", "v2")]
    assert second_model.calls == []

    third, third_model, _ = engine(tmp_path, [reply("both saved")], workspace)
    third.broker.resolve(EDITOR, suspended_again.prompt.id, {"approved": True})
    completed = run(third.recover(EDITOR, RUN_ID))
    assert completed.final_answer == "both saved"
    assert workspace.writes == [("d1", "v2"), ("d2", "v3")]
    assert [record.id for record in completed.steps[0].tool_calls] == ["w1", "w2"]
    assert len(third_model.calls) == 1
