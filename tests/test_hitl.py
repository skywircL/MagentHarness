"""Human-in-the-loop: approvals and questions suspend the run; answers resume it."""

from __future__ import annotations

import json

import pytest
from support import EDITOR, OTHER_WORKSPACE, build, run

from magent_harness import (
    SNAPSHOT_VERSION,
    AccessLevel,
    AgentConfig,
    AgentEngine,
    Caller,
    EventType,
    InMemoryPromptStore,
    InvalidPromptResponse,
    PromptAlreadyResolved,
    PromptBroker,
    PromptClaimLost,
    PromptExpired,
    PromptKind,
    PromptNotAnswered,
    PromptNotFound,
    PromptStatus,
    Role,
    RunRequest,
    RunSnapshot,
    RunStatus,
    SnapshotError,
    ToolPolicy,
)
from magent_harness.testing import ScriptedModel, call_tools, reply, tool_call

CONFIRM = ToolPolicy(AccessLevel.CONFIRM)


def request(query: str = "Update the roadmap") -> RunRequest:
    return RunRequest(caller=EDITOR, session_id="s1", query=query, policy=CONFIRM)


def update(doc: str = "d1", content: str = "v2", call_id: str = "u1") -> object:
    return tool_call("update_document", {"document_id": doc, "content": content}, call_id)


def test_write_suspends_without_running_and_resumes_after_approval() -> None:
    h = build([call_tools(update(), content="I'll update d1.")])
    first = run(h.engine.run(request()))

    assert first.status is RunStatus.SUSPENDED
    assert h.workspace.writes == []
    prompt = first.prompt
    assert prompt is not None and prompt.kind is PromptKind.TOOL_APPROVAL
    assert prompt.status is PromptStatus.SUSPENDED
    assert prompt.request["tool"] == "update_document"
    assert prompt.request["arguments"] == {"document_id": "d1", "content": "v2"}
    assert "snapshot" not in prompt.public_view()
    assert h.sink.of_type(EventType.PROMPT_REQUIRED)[0].data["prompt"]["id"] == prompt.id
    assert len(h.model.calls) == 1  # suspended, not waiting

    answered = h.broker.resolve(EDITOR, prompt.id, {"approved": True}, session_id="s1")
    assert answered.status is PromptStatus.ANSWERED

    h.model.extend([reply("d1 now has v2.")])
    second = run(h.engine.resume(EDITOR, prompt.id))

    assert second.status is RunStatus.COMPLETED
    assert second.final_answer == "d1 now has v2."
    assert h.workspace.writes == [("d1", "v2")]
    assert h.broker.store.get(prompt.id).status is PromptStatus.DONE  # type: ignore[union-attr]
    resumed_call = h.model.calls[1].messages
    assert resumed_call[-2].tool_calls[0].id == "u1"
    assert resumed_call[-1].role == "tool" and resumed_call[-1].tool_call_id == "u1"
    assert second.steps[0].thought == "I'll update d1."
    assert [m.role for m in second.messages] == ["user", "assistant", "tool", "assistant"]


def test_resume_works_from_a_fresh_engine_sharing_only_the_store() -> None:
    store = InMemoryPromptStore()
    h = build([call_tools(update())], broker=PromptBroker(store))
    prompt = run(h.engine.run(request())).prompt
    assert prompt is not None
    snapshot = RunSnapshot.decode(prompt.snapshot)
    assert snapshot.version == SNAPSHOT_VERSION
    assert RunSnapshot.decode(snapshot.encode()).to_dict() == snapshot.to_dict()

    PromptBroker(store).resolve(EDITOR, prompt.id, {"approved": True})
    other_process = AgentEngine(
        ScriptedModel([reply("done")]), h.workspace.registry(), PromptBroker(store)
    )
    result = run(other_process.resume(EDITOR, prompt.id))
    assert result.status is RunStatus.COMPLETED
    assert h.workspace.writes == [("d1", "v2")]


def test_rejection_is_reported_to_the_model_and_nothing_runs() -> None:
    h = build([call_tools(update())])
    prompt = run(h.engine.run(request())).prompt
    assert prompt is not None
    h.broker.resolve(EDITOR, prompt.id, {"approved": False, "reason": "wrong document"})
    h.model.extend([reply("Okay, I left it alone.")])
    result = run(h.engine.resume(EDITOR, prompt.id))

    assert h.workspace.writes == []
    error = result.steps[0].tool_calls[0].result.error
    assert "declined" in error and "wrong document" in error
    assert h.broker.store.get(prompt.id).status is PromptStatus.DONE  # type: ignore[union-attr]


def test_user_edited_arguments_are_executed_and_disclosed() -> None:
    h = build([call_tools(update(content="draft"))])
    prompt = run(h.engine.run(request())).prompt
    assert prompt is not None
    edited = {"document_id": "d1", "content": "final"}
    h.broker.resolve(EDITOR, prompt.id, {"approved": True, "modified_args": edited})
    h.model.extend([reply("ok")])
    result = run(h.engine.resume(EDITOR, prompt.id))

    assert h.workspace.writes == [("d1", "final")]
    record = result.steps[0].tool_calls[0]
    assert json.loads(record.arguments) == edited
    assert record.result.output.startswith("(The user edited the arguments")


def test_invalid_edited_arguments_do_not_run_and_close_the_claim() -> None:
    h = build([call_tools(update())])
    prompt = run(h.engine.run(request())).prompt
    assert prompt is not None
    h.broker.resolve(EDITOR, prompt.id, {"approved": True, "modified_args": {"document_id": 7}})
    h.model.extend([reply("ok")])
    result = run(h.engine.resume(EDITOR, prompt.id))
    assert h.workspace.writes == []
    assert "the arguments the user edited are invalid" in result.steps[0].tool_calls[0].result.error
    assert h.broker.store.get(prompt.id).status is PromptStatus.FAILED  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "response",
    [
        {"approve": True},  # misspelled: must not read as a decision
        {"approved": "yes"},
        {"approved": False, "modified_args": {"x": 1}},
        {"approved": True, "modified_args": ["not", "an", "object"]},
        {"approved": True, "extra": 1},
    ],
)
def test_malformed_answers_are_rejected(response: dict[str, object]) -> None:
    h = build([call_tools(update())])
    prompt = run(h.engine.run(request())).prompt
    assert prompt is not None
    with pytest.raises(InvalidPromptResponse):
        h.broker.resolve(EDITOR, prompt.id, response)
    assert h.broker.store.get(prompt.id).status is PromptStatus.SUSPENDED  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "intruder",
    [Caller("ws1", "mallory", Role.OWNER), OTHER_WORKSPACE],
)
def test_only_the_prompt_owner_can_answer_or_resume(intruder: Caller) -> None:
    h = build([call_tools(update())])
    prompt = run(h.engine.run(request())).prompt
    assert prompt is not None
    with pytest.raises(PromptNotFound):
        h.broker.resolve(intruder, prompt.id, {"approved": True})
    with pytest.raises(PromptNotFound):
        h.broker.resolve(EDITOR, prompt.id, {"approved": True}, session_id="another-session")
    h.broker.resolve(EDITOR, prompt.id, {"approved": True})
    with pytest.raises(PromptNotFound):
        run(h.engine.resume(intruder, prompt.id))
    assert h.workspace.writes == []


def test_an_approval_executes_at_most_once() -> None:
    h = build([call_tools(update())])
    prompt = run(h.engine.run(request())).prompt
    assert prompt is not None
    with pytest.raises(PromptNotAnswered):
        run(h.engine.resume(EDITOR, prompt.id))
    answered = h.broker.resolve(EDITOR, prompt.id, {"approved": True})
    with pytest.raises(PromptAlreadyResolved):
        h.broker.resolve(EDITOR, prompt.id, {"approved": True})

    # Two workers race for the same answer: only one claim wins.
    h.broker.claim(answered)
    with pytest.raises(PromptClaimLost):
        h.broker.claim(answered)
    with pytest.raises(PromptAlreadyResolved):
        run(h.engine.resume(EDITOR, prompt.id))
    assert h.workspace.writes == []


def test_finished_calls_are_not_rerun_and_later_calls_run_on_resume() -> None:
    h = build(
        [
            call_tools(
                tool_call("read_document", {"document_id": "d1"}, "r1"),
                update(call_id="u1"),
                tool_call("read_document", {"document_id": "d2"}, "r2"),
            )
        ]
    )
    prompt = run(h.engine.run(request())).prompt
    assert prompt is not None
    assert h.workspace.reads == ["d1"]
    assert prompt.tool_call_id == "u1"

    h.broker.resolve(EDITOR, prompt.id, {"approved": True})
    h.model.extend([reply("ok")])
    result = run(h.engine.resume(EDITOR, prompt.id))
    assert h.workspace.reads == ["d1", "d2"]
    assert h.workspace.writes == [("d1", "v2")]
    assert [r.id for r in result.steps[0].tool_calls] == ["r1", "u1", "r2"]


def test_a_second_approval_in_the_same_round_suspends_again() -> None:
    h = build([call_tools(update("d1", call_id="u1"), update("d2", call_id="u2"))])
    first = run(h.engine.run(request())).prompt
    assert first is not None
    h.broker.resolve(EDITOR, first.id, {"approved": True})
    second = run(h.engine.resume(EDITOR, first.id))

    assert second.status is RunStatus.SUSPENDED
    assert second.prompt is not None and second.prompt.tool_call_id == "u2"
    assert h.workspace.writes == [("d1", "v2")]

    h.broker.resolve(EDITOR, second.prompt.id, {"approved": True})
    h.model.extend([reply("both updated")])
    final = run(h.engine.resume(EDITOR, second.prompt.id))
    assert final.status is RunStatus.COMPLETED
    assert h.workspace.writes == [("d1", "v2"), ("d2", "v2")]
    assert len(h.model.calls) == 2


def test_ask_user_round_trip() -> None:
    h = build(
        [call_tools(tool_call("ask_user", {"question": "Which quarter?", "options": ["Q3"]}))]
    )
    first = run(h.engine.run(request("Summarize the plan")))
    prompt = first.prompt
    assert prompt is not None and prompt.kind is PromptKind.ASK_USER
    assert prompt.request == {"question": "Which quarter?", "options": ["Q3"]}

    with pytest.raises(InvalidPromptResponse):
        h.broker.resolve(EDITOR, prompt.id, {"answer": "  "})
    h.broker.resolve(EDITOR, prompt.id, {"answer": "Q4 please"})
    h.model.extend([reply("Q4: ship agent.")])
    result = run(h.engine.resume(EDITOR, prompt.id))
    assert result.final_answer == "Q4: ship agent."
    assert h.model.calls[1].messages[-1].content == "The user answered: Q4 please"


def test_expired_prompts_cannot_be_answered() -> None:
    now = [1000.0]
    broker = PromptBroker(expiry_s=60, clock=lambda: now[0])
    h = build([call_tools(update()), call_tools(update(call_id="u9"))], broker=broker)
    first = run(h.engine.run(request())).prompt
    assert first is not None and first.expires_at == 1060.0
    now[0] = 1061.0
    with pytest.raises(PromptExpired):
        broker.resolve(EDITOR, first.id, {"approved": True})
    assert broker.store.get(first.id).status is PromptStatus.EXPIRED  # type: ignore[union-attr]

    second = run(h.engine.run(request())).prompt
    assert second is not None
    now[0] = 2000.0
    assert [p.id for p in broker.expire_due()] == [second.id]
    assert broker.expire_due() == []


def test_canceled_prompts_close_the_run_with_failed_calls() -> None:
    h = build([call_tools(update(call_id="u1"), update("d2", call_id="u2"))])
    prompt = run(h.engine.run(request())).prompt
    assert prompt is not None
    h.broker.cancel(EDITOR, prompt.id, "the user moved on")
    with pytest.raises(PromptAlreadyResolved):
        h.broker.resolve(EDITOR, prompt.id, {"approved": True})

    steps = RunSnapshot.decode(prompt.snapshot).closed_steps("the user moved on")
    records = steps[-1].tool_calls
    assert [r.id for r in records] == ["u1", "u2"]
    assert all(r.result.error == "the user moved on" for r in records)


def test_role_revoked_before_resume_blocks_the_approved_call() -> None:
    h = build([call_tools(update())])
    prompt = run(h.engine.run(request())).prompt
    assert prompt is not None
    h.broker.resolve(EDITOR, prompt.id, {"approved": True})
    demoted = Caller(EDITOR.workspace_id, EDITOR.user_id, Role.VIEWER)
    h.model.extend([reply("I no longer have permission.")])
    result = run(h.engine.resume(demoted, prompt.id))

    assert h.workspace.writes == []
    assert result.steps[0].tool_calls[0].result.error.startswith("not permitted")
    assert h.broker.store.get(prompt.id).status is PromptStatus.FAILED  # type: ignore[union-attr]
    assert "update_document" not in h.model.calls[1].tool_names


def test_non_interactive_runs_refuse_instead_of_suspending() -> None:
    h = build(
        [call_tools(update()), reply("I need you to confirm that in the app.")],
        config=AgentConfig(interactive=False, model_retry_delay_s=0),
    )
    result = run(h.engine.run(request()))
    assert result.status is RunStatus.COMPLETED
    assert result.prompt is None
    assert "needs the user's confirmation" in result.steps[0].tool_calls[0].result.error
    assert "ask_user" not in h.model.calls[0].tool_names
    assert h.workspace.writes == []


def test_unreadable_snapshots_fail_the_prompt_instead_of_guessing() -> None:
    store = InMemoryPromptStore()
    h = build([call_tools(update())], broker=PromptBroker(store))
    prompt = run(h.engine.run(request())).prompt
    assert prompt is not None
    data = json.loads(prompt.snapshot)
    data["version"] = 99
    store.transition(prompt.id, [PromptStatus.SUSPENDED], snapshot=json.dumps(data))
    h.broker.resolve(EDITOR, prompt.id, {"approved": True})

    with pytest.raises(SnapshotError, match="version 99"):
        run(h.engine.resume(EDITOR, prompt.id))
    assert store.get(prompt.id).status is PromptStatus.FAILED  # type: ignore[union-attr]
    assert h.workspace.writes == []


def test_snapshot_rejects_inconsistent_pending_rounds() -> None:
    h = build([call_tools(update())])
    prompt = run(h.engine.run(request())).prompt
    assert prompt is not None
    data = json.loads(prompt.snapshot)
    data["pending"]["blocked_index"] = 5
    with pytest.raises(SnapshotError, match="outside round"):
        RunSnapshot.decode(json.dumps(data))
    with pytest.raises(SnapshotError):
        RunSnapshot.decode("not json")
