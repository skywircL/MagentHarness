"""Recovery remains safe across refusal commits and parallel persistence failures."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from support import EDITOR, FakeWorkspace, run

from magent_harness import (
    AccessLevel,
    AgentConfig,
    AgentEngine,
    Caller,
    FunctionTool,
    JsonlRunStore,
    PersistenceError,
    PromptStatus,
    RecoveryUnsafe,
    Risk,
    Role,
    RunRequest,
    RunStatus,
    SnapshotError,
    ToolContext,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from magent_harness.file_store import FileJournal
from magent_harness.testing import ScriptedModel, call_tools, reply, tool_call


class ProcessCrash(BaseException):
    pass


def _engine(path: Path, script: list[Any], registry: ToolRegistry) -> AgentEngine:
    return AgentEngine(
        ScriptedModel(script),
        registry,
        run_store=JsonlRunStore(path),
        config=AgentConfig(model_retry_delay_s=0),
    )


@pytest.mark.parametrize("refusal", ["role_revoked", "invalid_edited_arguments"])
def test_refused_approval_recovers_after_prompt_failed_before_round_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    workspace = FakeWorkspace.default()
    first = _engine(
        tmp_path,
        [call_tools(tool_call("update_document", {"document_id": "d1", "content": "new"}, "w1"))],
        workspace.registry(),
    )
    prompt = run(first.run(RunRequest(EDITOR, "s1", "Update", run_id="run"))).prompt
    assert prompt is not None
    response: dict[str, Any] = {"approved": True}
    caller = EDITOR
    if refusal == "role_revoked":
        caller = Caller(EDITOR.workspace_id, EDITOR.user_id, Role.VIEWER)
    else:
        response["modified_args"] = {"document_id": 7}
    first.broker.resolve(EDITOR, prompt.id, response)
    original_fail = first.broker.fail

    def fail_then_crash(prompt_id: str, reason: str) -> None:
        original_fail(prompt_id, reason)
        raise ProcessCrash("after prompt failure")

    with monkeypatch.context() as patch:
        patch.setattr(first.broker, "fail", fail_then_crash)
        with pytest.raises(ProcessCrash):
            run(first.resume(caller, prompt.id))

    fresh = _engine(tmp_path, [reply("The write was refused")], workspace.registry())
    result = run(fresh.recover(caller, "run"))
    assert result.status is RunStatus.COMPLETED
    assert workspace.writes == []
    assert not result.steps[0].tool_calls[0].result.success
    stored = fresh.broker.store.get(prompt.id)
    assert stored is not None and stored.status is PromptStatus.FAILED


def test_parallel_persistence_failure_finishes_sibling_tasks_before_releasing_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        slow_started, slow_stopped, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            if args["document_id"] == "slow":
                slow_started.set()
                try:
                    await release.wait()
                finally:
                    slow_stopped.set()
            else:
                await slow_started.wait()
            return ToolResult.ok(args["document_id"])

        registry = ToolRegistry()
        registry.register(
            FunctionTool(
                ToolSpec(
                    "read_document",
                    "Read",
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
        first = _engine(
            tmp_path,
            [
                call_tools(
                    tool_call("read_document", {"document_id": "slow"}, "slow"),
                    tool_call("read_document", {"document_id": "fast"}, "fast"),
                )
            ],
            registry,
        )
        original_append = FileJournal.append

        def fail_fast_result(
            journal: FileJournal, kind: str, data: dict[str, Any]
        ) -> dict[str, Any]:
            if kind == "tool_finished" and data["record"]["id"] == "fast":
                raise PersistenceError("disk unavailable")
            return original_append(journal, kind, data)

        with monkeypatch.context() as patch:
            patch.setattr(FileJournal, "append", fail_fast_result)
            with pytest.raises(PersistenceError):
                await first.run(RunRequest(EDITOR, "s1", "Read", run_id="run"))
            lingering = not slow_stopped.is_set()
            release.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        assert not lingering, "parallel tools outlived the run's exclusive journal lease"

    run(scenario())


@pytest.mark.parametrize("durable_import", [False, True])
def test_approval_snapshot_cannot_change_the_arguments_the_user_approved(
    tmp_path: Path, durable_import: bool
) -> None:
    workspace = FakeWorkspace.default()
    first = AgentEngine(
        ScriptedModel(
            [
                call_tools(
                    tool_call("update_document", {"document_id": "d1", "content": "new"}, "w1")
                )
            ]
        ),
        workspace.registry(),
    )
    prompt = run(first.run(RunRequest(EDITOR, "s1", "Update d1", run_id="run"))).prompt
    assert prompt is not None
    changed = json.loads(prompt.snapshot)
    changed["pending"]["tool_calls"][0]["arguments"] = json.dumps(
        {"document_id": "d2", "content": "changed without approval"}
    )
    first.broker.store.transition(prompt.id, [PromptStatus.SUSPENDED], snapshot=json.dumps(changed))
    first.broker.resolve(EDITOR, prompt.id, {"approved": True})
    fresh = AgentEngine(
        ScriptedModel([reply("done")]),
        workspace.registry(),
        first.broker,
        run_store=JsonlRunStore(tmp_path) if durable_import else None,
    )
    with pytest.raises(SnapshotError):
        run(fresh.resume(EDITOR, prompt.id))
    assert workspace.writes == []


@pytest.mark.parametrize("failure", ["exception", "timeout"])
@pytest.mark.parametrize("approved", [False, True])
def test_write_outcome_unknown_stops_model_and_refuses_recovery(
    tmp_path: Path, failure: str, approved: bool
) -> None:
    writes: list[str] = []

    async def write(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        writes.append(ctx.tool_call_id)
        if failure == "exception":
            raise RuntimeError("connection lost after the remote commit")
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            ToolSpec("write", "Write", risk=Risk.WRITE, timeout_s=0.01),
            write,
        )
    )
    first = _engine(tmp_path, [call_tools(tool_call("write", {}, "w1")), reply("done")], registry)
    if approved:
        prompt = run(first.run(RunRequest(EDITOR, "s1", "Write", run_id="run"))).prompt
        assert prompt is not None
        first.broker.resolve(EDITOR, prompt.id, {"approved": True})
        with pytest.raises(RecoveryUnsafe):
            run(first.resume(EDITOR, prompt.id))
    else:
        with pytest.raises(RecoveryUnsafe):
            run(
                first.run(
                    RunRequest(
                        EDITOR,
                        "s1",
                        "Write",
                        run_id="run",
                        policy=ToolPolicy(AccessLevel.AUTO_WRITE),
                    )
                )
            )
    assert writes == ["w1"]
    assert isinstance(first.model, ScriptedModel) and len(first.model.calls) == 1

    fresh = _engine(tmp_path, [reply("must not request another write")], registry)
    with pytest.raises(RecoveryUnsafe):
        run(fresh.recover(EDITOR, "run"))
    assert writes == ["w1"]
    assert isinstance(fresh.model, ScriptedModel) and fresh.model.calls == []


def test_legacy_prompt_import_keeps_its_answer_when_creation_commit_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = FakeWorkspace.default()
    first = AgentEngine(
        ScriptedModel(
            [
                call_tools(
                    tool_call("update_document", {"document_id": "d1", "content": "new"}, "w1")
                )
            ]
        ),
        workspace.registry(),
    )
    prompt = run(first.run(RunRequest(EDITOR, "s1", "Update", run_id="run"))).prompt
    assert prompt is not None
    first.broker.resolve(EDITOR, prompt.id, {"approved": True})
    importer = AgentEngine(
        ScriptedModel([]), workspace.registry(), first.broker, run_store=JsonlRunStore(tmp_path)
    )
    original_append = FileJournal.append

    def crash_after_creation(
        journal: FileJournal, kind: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        original_append(journal, kind, data)
        if kind == "created":
            raise ProcessCrash("after imported run creation")
        raise AssertionError("the import should crash at creation")

    with monkeypatch.context() as patch:
        patch.setattr(FileJournal, "append", crash_after_creation)
        with pytest.raises(ProcessCrash):
            run(importer.resume(EDITOR, prompt.id))
    assert workspace.writes == []

    fresh = AgentEngine(
        ScriptedModel([reply("saved once")]),
        workspace.registry(),
        first.broker,
        run_store=JsonlRunStore(tmp_path),
    )
    result = run(fresh.recover(EDITOR, "run"))
    assert result.status is RunStatus.COMPLETED
    assert workspace.writes == [("d1", "new")]
    stored = fresh.broker.store.get(prompt.id)
    assert stored is not None and stored.status is PromptStatus.DONE
    assert fresh.broker.store.list_by_status(PromptStatus.SUSPENDED) == []


@pytest.mark.parametrize("approved_value", [False, True])
def test_approval_argument_matching_preserves_nested_json_types(approved_value: bool) -> None:
    executed: list[dict[str, Any]] = []

    async def configure(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        executed.append(args)
        return ToolResult.ok("configured")

    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            ToolSpec(
                "configure",
                "Configure",
                {
                    "type": "object",
                    "properties": {"settings": {"type": "object"}},
                    "required": ["settings"],
                },
                risk=Risk.WRITE,
            ),
            configure,
        )
    )
    instance = AgentEngine(
        ScriptedModel(
            [
                call_tools(tool_call("configure", {"settings": {"enabled": approved_value}}, "c1")),
                reply("done"),
            ]
        ),
        registry,
    )
    prompt = run(instance.run(RunRequest(EDITOR, "s1", "Configure"))).prompt
    assert prompt is not None
    changed = json.loads(prompt.snapshot)
    changed["pending"]["tool_calls"][0]["arguments"] = json.dumps(
        {"settings": {"enabled": int(approved_value)}}
    )
    instance.broker.store.transition(
        prompt.id, [PromptStatus.SUSPENDED], snapshot=json.dumps(changed)
    )
    instance.broker.resolve(EDITOR, prompt.id, {"approved": True})
    with pytest.raises(SnapshotError):
        run(instance.resume(EDITOR, prompt.id))
    assert executed == []
