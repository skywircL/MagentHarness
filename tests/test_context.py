"""Context window management: compaction, tool-result trimming, overflow recovery."""

from __future__ import annotations

import json
from typing import Any

import pytest
from support import EDITOR, build, run

from magent_harness import (
    AgentConfig,
    ContextOverflowError,
    ContextSettings,
    EventType,
    FunctionTool,
    Message,
    ModelError,
    RunRequest,
    RunStatus,
    TokenEstimator,
    ToolCall,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    Usage,
)
from magent_harness.context import ContextManager, find_cut_point, parse_summary
from magent_harness.messages import KIND_COMPACTION_SUMMARY
from magent_harness.testing import ScriptedModel, call_tools, reply, tool_call

SMALL = ContextSettings(max_context_tokens=2_000, reserve_tokens=500, keep_recent_tokens=400)
QUERY = "Compare all the chapters"


def big_registry(chars: int = 2_400) -> ToolRegistry:
    async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult.ok(f"chapter {args['n']}: " + "lorem ipsum " * (chars // 12))

    registry = ToolRegistry()
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    registry.register(FunctionTool(ToolSpec("read_chapter", "Read.", schema), read))
    return registry


def chapter(n: int) -> Any:
    return call_tools(tool_call("read_chapter", {"n": n}, f"c{n}"))


def assert_tool_pairing(messages: list[Message]) -> None:
    issued: set[str] = set()
    for message in messages:
        if message.role == "assistant":
            issued = {call.id for call in message.tool_calls}
        elif message.role == "tool":
            assert message.tool_call_id in issued, message


def test_estimator_counts_cjk_densely() -> None:
    est = TokenEstimator()
    assert est.text("abcd" * 10) == 10
    assert est.text("文档工作台") == 5
    assert est.text("") == 0


def test_settings_reject_a_reserve_larger_than_the_window() -> None:
    with pytest.raises(ValueError):
        ContextSettings(max_context_tokens=1_000, reserve_tokens=1_000)
    assert SMALL.threshold == 1_500
    assert SMALL.effective_keep_recent == 375  # capped at a quarter of the usable window


def test_cut_point_never_separates_a_tool_result_from_its_call() -> None:
    call = ToolCall("c1", "read", "{}")
    messages = [
        Message("system", "s"),
        Message("user", "q"),
        Message("assistant", "", (call,)),
        Message("tool", "x" * 400, tool_call_id="c1"),
        Message("tool", "y" * 400, tool_call_id="c1"),
    ]
    est = TokenEstimator()
    for budget in range(0, 400, 10):
        cut = find_cut_point(messages, 1, budget, est)
        assert messages[cut].role != "tool"


def test_long_runs_compact_and_keep_the_request_verbatim() -> None:
    summarizer = ScriptedModel([reply("## Goal\nCompare chapters.\n## Progress\n- read 1-2")] * 5)
    h = build(
        [chapter(1), chapter(2), chapter(3), chapter(4), reply("Chapter 3 is longest.")],
        registry=big_registry(),
        config=AgentConfig(context=SMALL, model_retry_delay_s=0),
        summarizer=summarizer,
    )
    result = run(h.engine.run(RunRequest(EDITOR, "s1", QUERY)))

    assert result.status is RunStatus.COMPLETED
    compactions = h.sink.of_type(EventType.CONTEXT_COMPACTED)
    assert compactions and compactions[0].data["compacted"]
    assert compactions[0].data["tokens_after"] < compactions[0].data["tokens_before"]
    last_request = h.model.calls[-1].messages
    assert last_request[0].role == "system"
    assert last_request[1].kind == KIND_COMPACTION_SUMMARY
    summary, pinned = parse_summary(last_request[1].content)
    assert "Compare chapters" in summary
    assert pinned == QUERY
    assert_tool_pairing(last_request)
    # Steps keep the full results even though the transcript was compacted.
    assert len(result.steps) == 4
    assert all(len(r.result.output) > 2_000 for s in result.steps for r in s.tool_calls)
    # The summarizer saw a transcript, not live tool calls, and was offered no tools.
    assert summarizer.calls[0].tools == []
    assert json.loads(summarizer.calls[0].messages[-1].content)["conversation"]


def test_later_compactions_update_the_previous_summary() -> None:
    summarizer = ScriptedModel([reply("summary one"), reply("summary two"), reply("three")] * 3)
    h = build(
        [chapter(i) for i in range(1, 7)] + [reply("done")],
        registry=big_registry(),
        config=AgentConfig(context=SMALL, model_retry_delay_s=0),
        summarizer=summarizer,
    )
    run(h.engine.run(RunRequest(EDITOR, "s1", QUERY)))
    assert len(summarizer.calls) >= 2
    second_prompt = summarizer.calls[1].messages[-1].content
    assert json.loads(second_prompt)["previous_summary"] == "summary one"
    # The old summary is folded in, not stacked.
    kinds = [m.kind for m in h.model.calls[-1].messages]
    assert kinds.count(KIND_COMPACTION_SUMMARY) == 1


def test_failed_summaries_fall_back_to_trimming_tool_results() -> None:
    summarizer = ScriptedModel([ModelError("summarizer down")] * 10)
    h = build(
        [chapter(1), chapter(2), chapter(3), reply("ok")],
        registry=big_registry(),
        config=AgentConfig(context=SMALL, model_retry_delay_s=0),
        summarizer=summarizer,
    )
    result = run(h.engine.run(RunRequest(EDITOR, "s1", QUERY)))
    assert result.status is RunStatus.COMPLETED
    last_request = h.model.calls[-1].messages
    assert not any(m.kind == KIND_COMPACTION_SUMMARY for m in last_request)
    assert any("tool result trimmed" in m.content for m in last_request if m.role == "tool")
    assert_tool_pairing(last_request)
    # A compaction that cannot help is not retried every round.
    assert len(summarizer.calls) < 3


def test_provider_usage_is_the_baseline_for_the_next_estimate() -> None:
    # By character estimate the transcript (~1.2k tokens) is under the 1.5k
    # threshold, but the provider reports 1.95k (tool schemas, tokenizer
    # differences). The next round must trust the provider and compact.
    history = [Message("user" if i % 2 == 0 else "assistant", "w" * 1_000) for i in range(4)]
    est = TokenEstimator()
    assert est.messages(history) < SMALL.threshold
    summarizer = ScriptedModel([reply("short summary")])
    big_usage = Usage(prompt_tokens=1_900, completion_tokens=50)
    h = build(
        [
            call_tools(tool_call("read_document", {"document_id": "d1"}, "a"), usage=big_usage),
            reply("ok"),
        ],
        config=AgentConfig(context=SMALL, model_retry_delay_s=0),
        summarizer=summarizer,
    )
    run(h.engine.run(RunRequest(EDITOR, "s1", "q", history=history)))
    assert est.messages(h.model.calls[0].messages) < SMALL.threshold
    assert h.sink.of_type(EventType.CONTEXT_COMPACTED)
    assert h.model.calls[1].messages[1].kind == KIND_COMPACTION_SUMMARY


def test_context_overflow_is_recovered_once() -> None:
    history: list[Message] = []
    for i in range(6):
        history += [Message("user", f"question {i} " + "x" * 400), Message("assistant", "a" * 400)]
    summarizer = ScriptedModel([reply("earlier questions summarized")])
    h = build(
        [ContextOverflowError("too long"), reply("answer")],
        config=AgentConfig(model_retry_delay_s=0),
        summarizer=summarizer,
    )
    result = run(h.engine.run(RunRequest(EDITOR, "s1", "latest", history=history)))
    assert result.final_answer == "answer"
    event = h.sink.of_type(EventType.CONTEXT_COMPACTED)[0]
    assert event.data["forced"] and event.data["compacted"]
    assert len(h.model.calls[1].messages) < len(h.model.calls[0].messages)


def test_repeated_overflow_fails_the_run() -> None:
    h = build(
        [ContextOverflowError("too long"), ContextOverflowError("still too long")],
        summarizer=ScriptedModel([]),
    )
    result = run(h.engine.run(RunRequest(EDITOR, "s1", "q")))
    assert result.status is RunStatus.FAILED
    assert result.error == "still too long"


def test_disabled_management_leaves_messages_alone() -> None:
    manager = ContextManager(
        ContextSettings(max_context_tokens=0), TokenEstimator(), ScriptedModel([])
    )
    messages = [Message("system", "s"), Message("user", "x" * 100_000)]
    out, report = run(manager.fit(messages))
    assert out is messages and report is None
