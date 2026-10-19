"""Application-owned stores and explicit recovery streams keep backend authority."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from support import EDITOR, FakeWorkspace, run

from magent_harness import (
    AgentEngine,
    JsonlRunStore,
    RunJournal,
    RunNotFound,
    RunRequest,
    stream_recover,
)
from magent_harness.testing import ScriptedModel, call_tools, reply, tool_call


class ApplicationStore:
    """An application supplies its own execution lease and human-answer store."""

    def __init__(self, path: Path) -> None:
        self.storage = JsonlRunStore(path)
        self.prompts = self.storage.prompts
        self.leases = 0

    @contextmanager
    def journal(self, caller, run_id: str) -> Iterator[RunJournal]:
        self.leases += 1
        with self.storage.journal(caller, run_id) as journal:
            yield journal


class ProcessDied(BaseException):
    pass


def test_application_store_recovery_stream_reuses_completed_tools(tmp_path: Path) -> None:
    workspace = FakeWorkspace.default()
    store = ApplicationStore(tmp_path)
    initial = AgentEngine(
        ScriptedModel(
            [call_tools(tool_call("read_document", {"document_id": "d1"}, "r1")), ProcessDied()]
        ),
        workspace.registry(),
        run_store=store,
    )
    with pytest.raises(ProcessDied):
        run(initial.run(RunRequest(EDITOR, "s1", "Read d1", run_id="run1")))
    model = ScriptedModel([reply("recovered")])
    restored = AgentEngine(model, workspace.registry(), run_store=store)

    async def consume():
        async with stream_recover(restored, EDITOR, "run1", session_id="s1") as stream:
            frames = [frame async for frame in stream]
            return frames, await stream.result()

    frames, result = run(consume())
    names = [frame.split(b"\n", 1)[0] for frame in frames]
    assert b"event: run_resumed" in names
    assert b"event: final_answer" in names
    assert result.final_answer == "recovered"
    assert workspace.reads == ["d1"]
    assert len(model.calls) == 1 and store.leases == 2
    assert json.loads(frames[-1].split(b"data: ", 1)[1])["type"] == "run_completed"


def test_recovery_stream_can_return_cached_terminal_without_new_events(tmp_path: Path) -> None:
    store = ApplicationStore(tmp_path)
    first = AgentEngine(ScriptedModel([reply("already saved")]), run_store=store)
    expected = run(first.run(RunRequest(EDITOR, "s1", "hello", run_id="run1")))
    model = ScriptedModel([])
    restored = AgentEngine(model, run_store=store)

    async def consume():
        async with stream_recover(restored, EDITOR, "run1", session_id="s1") as stream:
            assert [frame async for frame in stream] == []
            return await stream.result()

    assert run(consume()) == expected
    assert model.calls == []


def test_recovery_stream_rejects_another_session_before_model_calls(tmp_path: Path) -> None:
    store = ApplicationStore(tmp_path)
    first = AgentEngine(ScriptedModel([ProcessDied()]), run_store=store)
    with pytest.raises(ProcessDied):
        run(first.run(RunRequest(EDITOR, "s1", "hello", run_id="run1")))
    model = ScriptedModel([])
    restored = AgentEngine(model, run_store=store)

    async def consume():
        async with stream_recover(restored, EDITOR, "run1", session_id="other") as stream:
            return [frame async for frame in stream]

    with pytest.raises(RunNotFound):
        run(consume())
    assert model.calls == []
