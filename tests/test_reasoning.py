"""Provider reasoning survives history and recovery without becoming public output."""

from __future__ import annotations

import json

import pytest
from support import EDITOR, build, run

from magent_harness import (
    AgentConfig,
    AgentEngine,
    Caller,
    ChatResponse,
    CollectingSink,
    ContextSettings,
    Message,
    ModelError,
    PromptBroker,
    PromptStatus,
    Role,
    RunRequest,
    RunSnapshot,
    RunStatus,
    SnapshotError,
    TokenEstimator,
)
from magent_harness.context import ContextManager, trim_tool_results
from magent_harness.model import FINISH_CONTENT_FILTER, FINISH_LENGTH, FINISH_STOP
from magent_harness.testing import ScriptedModel, tool_call

PRIVATE = "private provider reasoning: preserve verbatim\n工作区分析"


@pytest.mark.parametrize("reasoning", [None, "", PRIVATE])
def test_message_round_trip_distinguishes_missing_and_empty_reasoning(
    reasoning: str | None,
) -> None:
    message = Message("assistant", "answer", reasoning_content=reasoning)
    encoded = message.to_dict()
    assert ("reasoning_content" in encoded) is (reasoning is not None)
    assert Message.from_dict(encoded) == message
    assert (
        Message.from_dict({"role": "assistant", "content": "old answer"}).reasoning_content is None
    )


@pytest.mark.parametrize("finish_reason", [FINISH_STOP, FINISH_LENGTH, FINISH_CONTENT_FILTER])
def test_final_answer_reasoning_survives_next_user_turn_but_is_not_public(
    finish_reason: str,
) -> None:
    h = build([ChatResponse("answer", finish_reason=finish_reason, reasoning_content=PRIVATE)])
    first = run(h.engine.run(RunRequest(EDITOR, "s1", "question")))
    assert first.final_answer == "answer"
    assert first.messages[-1].reasoning_content == PRIVATE
    assert PRIVATE not in repr(h.sink.events)
    assert PRIVATE not in repr(first.steps)

    h.model.extend([ChatResponse("next answer", reasoning_content="next reasoning")])
    run(h.engine.run(RunRequest(EDITOR, "s1", "next question", history=first.messages)))
    assert h.model.calls[-1].messages[-2].reasoning_content == PRIVATE


@pytest.mark.parametrize("decision", ["approved", "declined", "revoked"])
def test_reasoning_and_tool_pairs_survive_approval_in_a_new_engine(decision: str) -> None:
    first_reasoning = "first round reasoning"
    h = build(
        [
            ChatResponse(
                tool_calls=[tool_call("read_document", {"document_id": "d1"}, "r1")],
                reasoning_content=first_reasoning,
            ),
            ChatResponse(
                content="I can update this document.",
                tool_calls=[
                    tool_call("read_document", {"document_id": "d2"}, "r2"),
                    tool_call("update_document", {"document_id": "d1", "content": "v2"}, "w1"),
                ],
                reasoning_content=PRIVATE,
            ),
        ]
    )
    suspended = run(h.engine.run(RunRequest(EDITOR, "s1", "Update the plan")))
    prompt = suspended.prompt
    assert prompt is not None
    snapshot = RunSnapshot.decode(prompt.snapshot)
    assert snapshot.version == 2
    assert snapshot.messages[-2].reasoning_content == first_reasoning
    assert snapshot.pending.reasoning_content == PRIVATE
    assert snapshot.pending.thought == "I can update this document."
    assert PRIVATE not in repr(prompt.public_view())
    assert PRIVATE not in repr(h.sink.events)
    assert PRIVATE not in repr(suspended.steps)

    broker = PromptBroker(h.broker.store)
    broker.resolve(EDITOR, prompt.id, {"approved": decision != "declined"})
    caller = (
        Caller(EDITOR.workspace_id, EDITOR.user_id, Role.VIEWER)
        if decision == "revoked"
        else EDITOR
    )
    model = ScriptedModel([ChatResponse("done", reasoning_content="final reasoning")])
    sink = CollectingSink()
    engine = AgentEngine(model, h.workspace.registry(), broker, events=sink)
    completed = run(engine.resume(caller, prompt.id))
    assert completed.status is RunStatus.COMPLETED
    assert h.workspace.reads == ["d1", "d2"]
    assert h.workspace.writes == ([("d1", "v2")] if decision == "approved" else [])
    assistants = [m for m in model.calls[0].messages if m.role == "assistant"]
    assert [m.reasoning_content for m in assistants] == [first_reasoning, PRIVATE]
    assert [m.tool_call_id for m in model.calls[0].messages if m.role == "tool"] == [
        "r1",
        "r2",
        "w1",
    ]
    assert PRIVATE not in repr(sink.events)
    assert PRIVATE not in repr(completed.steps)
    expected_status = PromptStatus.FAILED if decision == "revoked" else PromptStatus.DONE
    stored_prompt = broker.store.get(prompt.id)
    assert stored_prompt is not None and stored_prompt.status is expected_status


def test_second_approval_preserves_pending_reasoning_and_does_not_repeat_a_write() -> None:
    h = build(
        [
            ChatResponse(
                tool_calls=[
                    tool_call("update_document", {"document_id": doc, "content": "v2"}, doc)
                    for doc in ("d1", "d2")
                ],
                reasoning_content="",
            )
        ]
    )
    first = run(h.engine.run(RunRequest(EDITOR, "s1", "Update both"))).prompt
    assert first is not None
    h.broker.resolve(EDITOR, first.id, {"approved": True})
    second = run(h.engine.resume(EDITOR, first.id)).prompt
    assert second is not None
    assert RunSnapshot.decode(second.snapshot).pending.reasoning_content == ""
    h.broker.resolve(EDITOR, second.id, {"approved": True})
    h.model.extend([ChatResponse("done", reasoning_content="final")])
    run(h.engine.resume(EDITOR, second.id))
    assert h.workspace.writes == [("d1", "v2"), ("d2", "v2")]
    assert h.model.calls[-1].messages[-3].reasoning_content == ""


def test_version_one_snapshot_still_resumes_and_reencodes_as_version_two() -> None:
    h = build(
        [
            ChatResponse(
                tool_calls=[tool_call("update_document", {"document_id": "d1", "content": "v2"})]
            )
        ]
    )
    prompt = run(h.engine.run(RunRequest(EDITOR, "s1", "Update"))).prompt
    assert prompt is not None
    data = json.loads(prompt.snapshot)
    data["version"] = 1
    old_snapshot = json.dumps(data)
    decoded = RunSnapshot.decode(old_snapshot)
    assert decoded.pending.reasoning_content is None
    assert all(message.reasoning_content is None for message in decoded.messages)
    assert json.loads(decoded.encode())["version"] == 2
    h.broker.store.transition(prompt.id, [PromptStatus.SUSPENDED], snapshot=old_snapshot)
    h.broker.resolve(EDITOR, prompt.id, {"approved": True})
    h.model.extend([ChatResponse("done")])
    assert run(h.engine.resume(EDITOR, prompt.id)).status is RunStatus.COMPLETED
    assert h.workspace.writes == [("d1", "v2")]


@pytest.mark.parametrize("location", ["pending", "message"])
def test_malformed_persisted_reasoning_fails_before_any_approved_write(location: str) -> None:
    h = build(
        [
            ChatResponse(
                tool_calls=[tool_call("update_document", {"document_id": "d1", "content": "v2"})],
                reasoning_content=PRIVATE,
            )
        ]
    )
    prompt = run(h.engine.run(RunRequest(EDITOR, "s1", "Update"))).prompt
    assert prompt is not None
    data = json.loads(prompt.snapshot)
    target = data["pending"] if location == "pending" else data["messages"][0]
    target["reasoning_content"] = {"unexpected": "object"}
    h.broker.store.transition(prompt.id, [PromptStatus.SUSPENDED], snapshot=json.dumps(data))
    h.broker.resolve(EDITOR, prompt.id, {"approved": True})
    with pytest.raises(SnapshotError, match="reasoning_content must be a string or null"):
        run(h.engine.resume(EDITOR, prompt.id))
    assert h.workspace.writes == []
    failed = h.broker.store.get(prompt.id)
    assert failed is not None and failed.status is PromptStatus.FAILED


def test_empty_answer_retry_preserves_real_reasoning() -> None:
    h = build([ChatResponse(reasoning_content=PRIVATE), ChatResponse("done", reasoning_content="")])
    result = run(h.engine.run(RunRequest(EDITOR, "s1", "question")))
    retry = h.model.calls[-1].messages
    assert retry[-2].role == "assistant" and retry[-2].content == ""
    assert retry[-2].reasoning_content == PRIVATE
    assert result.messages[-1].reasoning_content == ""
    assert PRIVATE not in repr(h.sink.events)


def test_truncated_tool_round_retains_reasoning_but_fallback_does_not_invent_it() -> None:
    h = build(
        [
            ChatResponse(
                tool_calls=[tool_call("read_document", '{"document_id":')],
                finish_reason=FINISH_LENGTH,
                reasoning_content=PRIVATE,
            )
        ],
        config=AgentConfig(max_length_rounds=1),
    )
    result = run(h.engine.run(RunRequest(EDITOR, "s1", "question")))
    assert result.stop_reason == "length_limit"
    assert result.messages[-3].reasoning_content == PRIVATE
    assert result.messages[-1].reasoning_content is None
    assert h.workspace.reads == []


@pytest.mark.parametrize("outcome", ["answer", "empty", "error"])
def test_final_synthesis_preserves_only_the_real_answer_reasoning(outcome: str) -> None:
    synthesis = (
        ModelError("unavailable")
        if outcome == "error"
        else ChatResponse("done" if outcome == "answer" else "", reasoning_content=PRIVATE)
    )
    h = build([synthesis], config=AgentConfig(max_iterations=0))
    result = run(h.engine.run(RunRequest(EDITOR, "s1", "question")))
    assert result.stop_reason == "max_iterations"
    assert result.messages[-1].reasoning_content == (PRIVATE if outcome == "answer" else None)
    assert PRIVATE not in repr(h.sink.events)


def test_estimator_counts_reasoning_and_compaction_keeps_the_tail_verbatim() -> None:
    estimator = TokenEstimator()
    ordinary = Message("assistant", "public answer")
    reasoning = Message("assistant", "public answer", reasoning_content=PRIVATE)
    assert estimator.message(reasoning) - estimator.message(ordinary) == estimator.text(PRIVATE)
    old_reasoning = "opaque old reasoning" * 500
    tail_call = tool_call("read_document", {"document_id": "d1"}, "tail")
    tail = Message("assistant", tool_calls=(tail_call,), reasoning_content=PRIVATE)
    messages = [
        Message("system", "system"),
        Message("user", "old question"),
        Message("assistant", "old answer", reasoning_content=old_reasoning),
        Message("user", "current question"),
        tail,
        Message("tool", "the result", tool_call_id="tail"),
    ]
    summarizer = ScriptedModel([ChatResponse("short summary", reasoning_content="summary private")])
    manager = ContextManager(
        ContextSettings(max_context_tokens=2_000, reserve_tokens=500, keep_recent_tokens=100),
        estimator,
        summarizer,
    )
    fitted, report = run(manager.fit(messages))
    assert report is not None and report.compacted
    assert fitted[-2] is tail
    assert fitted[-1].tool_call_id == fitted[-2].tool_calls[0].id
    assert old_reasoning not in repr(summarizer.calls[0].messages)
    assert PRIVATE not in repr(summarizer.calls[0].messages)
    assert "summary private" not in repr(fitted)
    assert report.tokens_after == estimator.messages(fitted)


def test_trimming_tool_results_leaves_reasoning_unchanged() -> None:
    call = tool_call("read_document", {"document_id": "d1"}, "r1")
    assistant = Message("assistant", tool_calls=(call,), reasoning_content=PRIVATE)
    messages = [assistant, Message("tool", "x" * 4_000, tool_call_id="r1")]
    fitted, trimmed = trim_tool_results(messages, TokenEstimator(), 200)
    assert trimmed
    assert fitted[0] is assistant
    assert fitted[1].tool_call_id == "r1"
    assert len(messages[1].content) == 4_000
