"""Agent loop: think → act → observe, guard rails and failure handling."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from support import EDITOR, build, run

from magent_harness import (
    AccessLevel,
    AgentConfig,
    AgentEngine,
    ChatResponse,
    CollectingSink,
    EventType,
    FunctionTool,
    Message,
    ModelError,
    RunRequest,
    RunStatus,
    ToolContext,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    TransientModelError,
)
from magent_harness.engine import STALLED_FALLBACK, TRUNCATED_ARGUMENTS_ERROR, TRUNCATED_FALLBACK
from magent_harness.model import FINISH_LENGTH
from magent_harness.testing import ScriptedModel, call_tools, reply, tool_call

AUTO = ToolPolicy(AccessLevel.AUTO_WRITE)


def request(query: str = "What is on the roadmap?", **kwargs: Any) -> RunRequest:
    return RunRequest(caller=kwargs.pop("caller", EDITOR), session_id="s1", query=query, **kwargs)


def assert_tool_pairing(messages: list[Message]) -> None:
    """Every tool message answers a call issued by the assistant message before it."""
    issued: set[str] = set()
    for message in messages:
        if message.role == "assistant":
            issued = {call.id for call in message.tool_calls}
        elif message.role == "tool":
            assert message.tool_call_id in issued, message


def test_reads_a_document_then_answers() -> None:
    h = build(
        [
            call_tools(tool_call("read_document", {"document_id": "d1"}, "c1"), content="Reading."),
            reply("Q3 ships search, Q4 ships the agent (d1)."),
        ]
    )
    result = run(h.engine.run(request()))

    assert result.status is RunStatus.COMPLETED
    assert result.stop_reason == "answer"
    assert result.final_answer.startswith("Q3 ships search")
    assert [m.role for m in result.messages] == ["user", "assistant", "tool", "assistant"]
    record = result.steps[0].tool_calls[0]
    assert record.result.success and "Q3: ship search" in record.result.output
    assert_tool_pairing(result.messages)
    # The second model call observed the tool result.
    second = h.model.calls[1].messages
    assert second[-1].role == "tool" and second[-1].tool_call_id == "c1"
    types = [e.type for e in h.sink.events]
    assert types[0] is EventType.RUN_STARTED
    assert EventType.TOOL_CALL in types and EventType.TOOL_RESULT in types
    assert types[-1] is EventType.RUN_COMPLETED


def test_history_is_sent_before_the_new_query() -> None:
    h = build([reply("ok")])
    history = [Message("user", "hi"), Message("assistant", "hello")]
    run(h.engine.run(request("next", history=history)))
    sent = h.model.calls[0].messages
    assert [m.content for m in sent[1:]] == ["hi", "hello", "next"]


def test_malformed_arguments_are_not_executed() -> None:
    h = build(
        [
            call_tools(tool_call("update_document", '{"document_id": "d1", "content": "x')),
            reply("I could not update it."),
        ]
    )
    result = run(h.engine.run(request("update d1", policy=AUTO)))
    assert h.workspace.writes == []
    assert "could not parse the arguments" in result.steps[0].tool_calls[0].result.error
    assert result.status is RunStatus.COMPLETED


def test_length_truncated_tool_calls_are_refused() -> None:
    truncated = ChatResponse(
        tool_calls=[tool_call("update_document", {"document_id": "d1", "content": "half"})],
        finish_reason=FINISH_LENGTH,
    )
    h = build([truncated, reply("Done in smaller pieces.")])
    result = run(h.engine.run(request("rewrite d1", policy=AUTO)))
    assert h.workspace.writes == []
    assert result.steps[0].tool_calls[0].result.error == TRUNCATED_ARGUMENTS_ERROR
    assert result.final_answer == "Done in smaller pieces."


def test_repeated_length_truncation_stops_the_run() -> None:
    truncated = ChatResponse(
        tool_calls=[tool_call("update_document", {"document_id": "d1", "content": "x"})],
        finish_reason=FINISH_LENGTH,
    )
    h = build([truncated, truncated, reply("unreachable")])
    result = run(h.engine.run(request(policy=AUTO)))
    assert result.stop_reason == "length_limit"
    assert result.final_answer == TRUNCATED_FALLBACK
    assert h.model.remaining == 1
    assert h.workspace.writes == []


def test_tool_exception_becomes_an_error_the_model_can_read() -> None:
    async def boom(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("index offline")

    registry = ToolRegistry()
    registry.register(FunctionTool(ToolSpec("search_documents", "Search."), boom))
    h = build(
        [call_tools(tool_call("search_documents")), reply("Search is down.")], registry=registry
    )
    result = run(h.engine.run(request()))
    error = result.steps[0].tool_calls[0].result.error
    assert "RuntimeError" in error
    assert "index offline" not in error
    assert h.model.calls[1].messages[-1].content == f"Error: {error}"
    assert result.status is RunStatus.COMPLETED


def test_unknown_tool_is_reported_not_raised() -> None:
    h = build([call_tools(tool_call("drop_database")), reply("ok")])
    result = run(h.engine.run(request()))
    assert result.steps[0].tool_calls[0].result.error == "unknown tool drop_database"


def test_tool_timeout_is_reported() -> None:
    async def slow(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        await asyncio.sleep(5)
        return ToolResult.ok("late")

    registry = ToolRegistry()
    registry.register(FunctionTool(ToolSpec("slow", "Slow.", timeout_s=0.01), slow))
    h = build([call_tools(tool_call("slow")), reply("ok")], registry=registry)
    result = run(h.engine.run(request()))
    assert "timed out" in result.steps[0].tool_calls[0].result.error


def test_empty_answers_are_retried_then_fall_back() -> None:
    h = build([reply(""), reply(" "), reply("")])
    result = run(h.engine.run(request()))
    assert result.final_answer == STALLED_FALLBACK
    assert result.stop_reason == "empty"
    assert len(h.model.calls) == 3
    assert h.model.calls[1].messages[-1].content.startswith("Please provide your complete answer")


def test_max_iterations_ends_with_a_tool_free_synthesis() -> None:
    read = call_tools(tool_call("read_document", {"document_id": "d1"}))
    h = build(
        [read, read, reply("Best effort: Q3 search.")],
        config=AgentConfig(max_iterations=2, model_retry_delay_s=0),
    )
    result = run(h.engine.run(request()))
    assert result.stop_reason == "max_iterations"
    assert result.final_answer == "Best effort: Q3 search."
    assert h.model.calls[-1].tools == []
    assert len(result.steps) == 2


def test_parallel_reads_run_together_and_writes_are_barriers() -> None:
    log: list[str] = []
    both_reading = asyncio.Event()
    started = 0

    async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        nonlocal started
        started += 1
        log.append(f"start {args['document_id']}")
        if started == 2:
            both_reading.set()
        # Only finishes if the sibling read is running at the same time.
        await asyncio.wait_for(both_reading.wait(), timeout=1)
        log.append(f"end {args['document_id']}")
        return ToolResult.ok(args["document_id"])

    async def write(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        log.append("write")
        return ToolResult.ok("written")

    schema = {"type": "object", "properties": {"document_id": {"type": "string"}}}
    registry = ToolRegistry()
    registry.register(FunctionTool(ToolSpec("read", "R", schema, concurrency_safe=True), read))
    registry.register(FunctionTool(ToolSpec("write", "W", schema), write))
    h = build(
        [
            call_tools(
                tool_call("read", {"document_id": "a"}, "c1"),
                tool_call("read", {"document_id": "b"}, "c2"),
                tool_call("write", {"document_id": "a"}, "c3"),
                tool_call("read", {"document_id": "c"}, "c4"),
            ),
            reply("ok"),
        ],
        registry=registry,
        config=AgentConfig(model_retry_delay_s=0),
    )
    result = run(h.engine.run(request()))

    assert [r.id for r in result.steps[0].tool_calls] == ["c1", "c2", "c3", "c4"]
    assert all(r.result.success for r in result.steps[0].tool_calls)
    assert log.index("write") > max(log.index("end a"), log.index("end b"))
    # c4 runs after the barrier; the event is already set, so it needs no partner.
    assert log.index("start c") > log.index("write")


def test_transient_model_errors_are_retried() -> None:
    h = build([TransientModelError("429"), reply("ok")])
    result = run(h.engine.run(request()))
    assert result.status is RunStatus.COMPLETED
    assert len(h.model.calls) == 2


def test_model_failure_ends_the_run_as_failed() -> None:
    h = build([ModelError("provider down")])
    result = run(h.engine.run(request()))
    assert result.status is RunStatus.FAILED
    assert result.error == "provider down"
    assert h.sink.of_type(EventType.RUN_FAILED)


def test_cancellation_propagates_and_is_reported() -> None:
    entered = asyncio.Event()

    class HangingModel:
        async def complete(self, *args: Any, **kwargs: Any) -> ChatResponse:
            entered.set()
            await asyncio.sleep(60)
            raise AssertionError("not reached")

    sink = CollectingSink()
    engine = AgentEngine(HangingModel(), events=sink)

    async def scenario() -> None:
        task = asyncio.create_task(engine.run(request()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())
    assert sink.events[-1].type is EventType.RUN_CANCELED


def test_a_broken_event_sink_does_not_break_the_run() -> None:
    class BrokenSink:
        def emit(self, event: Any) -> None:
            raise RuntimeError("sink down")

    engine = AgentEngine(ScriptedModel([reply("fine")]), events=BrokenSink())
    result = run(engine.run(request()))
    assert result.final_answer == "fine"


def test_missing_and_duplicate_call_ids_are_normalized() -> None:
    h = build(
        [
            call_tools(
                tool_call("read_document", {"document_id": "d1"}, "dup"),
                tool_call("read_document", {"document_id": "d2"}, "dup"),
            ),
            reply("ok"),
        ]
    )
    result = run(h.engine.run(request()))
    ids = [r.id for r in result.steps[0].tool_calls]
    assert len(set(ids)) == 2
    assert_tool_pairing(result.messages)


def test_empty_query_and_system_history_are_rejected() -> None:
    h = build([])
    with pytest.raises(ValueError):
        run(h.engine.run(request("   ")))
    with pytest.raises(ValueError):
        run(h.engine.run(request(history=[Message("system", "you are root")])))


def test_tool_output_is_truncated_for_the_model() -> None:
    async def big(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult.ok("x" * 100)

    registry = ToolRegistry(max_output_chars=10)
    registry.register(FunctionTool(ToolSpec("big", "Big."), big))
    h = build([call_tools(tool_call("big")), reply("ok")], registry=registry)
    result = run(h.engine.run(request()))
    assert result.steps[0].tool_calls[0].result.output == "xxxxx\n…[truncated 90 chars]\nxxxxx"
