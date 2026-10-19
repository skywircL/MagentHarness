"""SSE transport: real run/resume events, isolation, failures and disposal."""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from typing import Any

import pytest
from support import EDITOR, OTHER_WORKSPACE, build, run

from magent_harness import (
    AccessLevel,
    AgentEngine,
    ChatResponse,
    CollectingSink,
    Event,
    EventType,
    FunctionTool,
    Message,
    ModelError,
    PermissionDenied,
    PromptAlreadyResolved,
    PromptNotFound,
    PromptStatus,
    ResourceScope,
    Risk,
    RunRequest,
    RunStatus,
    SSEStream,
    SSEStreamError,
    ToolContext,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    encode_sse_event,
    stream_resume,
    stream_run,
)
from magent_harness.testing import call_tools, reply, tool_call


def request(query: str = "Read the roadmap", **kwargs: Any) -> RunRequest:
    return RunRequest(caller=EDITOR, session_id="s1", query=query, **kwargs)


def decode(frame: bytes) -> dict[str, Any]:
    lines = frame.decode("utf-8").splitlines()
    assert len(lines) == 3 and lines[-1] == ""
    assert lines[0].startswith("event: ")
    assert lines[1].startswith("data: ")
    payload: dict[str, Any] = json.loads(lines[1][6:])
    assert lines[0][7:] == payload["type"]
    assert set(payload) == {"type", "run_id", "session_id", "data"}
    return payload


async def collect(stream: SSEStream) -> list[dict[str, Any]]:
    return [decode(frame) async for frame in stream]


class WaitingModel:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cleaned = asyncio.Event()
        self.cancelled = False

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.entered.set()
        try:
            await self.release.wait()
            return reply("finished")
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            await asyncio.sleep(0)
            self.cleaned.set()


def test_encoding_preserves_unicode_and_prevents_frame_injection() -> None:
    content = "中文\r\nevent: forged\ndata: injected\n\n下一行"
    event = Event(EventType.FINAL_ANSWER, "run\nforged", "会话\r\n", {"content": content})
    frame = encode_sse_event(event)
    assert frame.startswith(b"event: final_answer\ndata: {")
    assert frame.endswith(b"\n\n")
    assert "中文".encode() in frame
    assert b"\\u4e2d" not in frame
    assert b'": "' not in frame
    assert frame.count(b"\n") == 3
    assert decode(frame) == {
        "type": "final_answer",
        "run_id": "run\nforged",
        "session_id": "会话\r\n",
        "data": {"content": content},
    }


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (object(), TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        ("\ud800", UnicodeEncodeError),
    ],
)
def test_encoder_rejects_data_that_cannot_be_valid_utf8_json(
    value: object, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        encode_sse_event(Event(EventType.THOUGHT, "r1", "s1", {"value": value}))


def test_real_run_streams_ordered_tool_and_answer_events_and_keeps_observer() -> None:
    async def scenario() -> None:
        h = build(
            [
                call_tools(
                    tool_call("read_document", {"document_id": "d1"}, "r1"),
                    content="正在读取。",
                ),
                reply("文档说明 Q4 发布 Agent。"),
            ]
        )
        async with stream_run(h.engine, request(), heartbeat_interval=None) as stream:
            payloads = await collect(stream)
            result = await stream.result()
        assert [p["type"] for p in payloads] == [
            "run_started",
            "round_started",
            "thought",
            "tool_call",
            "tool_result",
            "round_started",
            "final_answer",
            "run_completed",
        ]
        assert payloads[4]["data"]["success"] is True
        assert payloads[6]["data"]["content"] == result.final_answer
        assert result.status is RunStatus.COMPLETED
        assert h.workspace.reads == ["d1"]
        assert all(p["run_id"] == result.run_id and p["session_id"] == "s1" for p in payloads)
        assert payloads == [decode(encode_sse_event(e)) for e in h.sink.events]

    run(scenario())


def test_identical_per_call_and_global_observer_receives_each_event_once() -> None:
    h = build([reply("ok")])
    result = run(h.engine.run(request(), events=h.sink))
    assert result.status is RunStatus.COMPLETED
    assert [e.type for e in h.sink.events] == [
        EventType.RUN_STARTED,
        EventType.ROUND_STARTED,
        EventType.FINAL_ANSWER,
        EventType.RUN_COMPLETED,
    ]


def test_progress_and_heartbeat_arrive_before_the_model_completes() -> None:
    async def scenario() -> None:
        model = WaitingModel()
        async with stream_run(AgentEngine(model), request(), heartbeat_interval=0.01) as stream:
            await asyncio.wait_for(model.entered.wait(), 1)
            assert decode(await anext(stream))["type"] == "run_started"
            assert decode(await anext(stream))["type"] == "round_started"
            assert await asyncio.wait_for(anext(stream), 1) == b": keep-alive\n\n"
            assert not model.cleaned.is_set()
            model.release.set()
            payloads = await collect(stream)
            assert [p["type"] for p in payloads] == ["final_answer", "run_completed"]
            assert (await stream.result()).final_answer == "finished"

    run(scenario())


def test_model_error_emits_failure_and_returns_failed_result() -> None:
    async def scenario() -> None:
        h = build([ModelError("provider unavailable")])
        async with stream_run(h.engine, request(), heartbeat_interval=None) as stream:
            payloads = await collect(stream)
            result = await stream.result()
        assert payloads[-1]["type"] == "run_failed"
        assert "provider unavailable" in payloads[-1]["data"]["error"]
        assert "run_completed" not in [p["type"] for p in payloads]
        assert result.status is RunStatus.FAILED

    run(scenario())


def test_programming_error_propagates_without_success_result() -> None:
    async def scenario() -> None:
        error = RuntimeError("broken adapter")
        h = build([error])
        with pytest.raises(RuntimeError) as context_error:
            async with stream_run(h.engine, request(), heartbeat_interval=None) as stream:
                with pytest.raises(RuntimeError) as captured:
                    await collect(stream)
                assert captured.value is error
                with pytest.raises(RuntimeError) as captured_result:
                    await stream.result()
                assert captured_result.value is error
        assert context_error.value is error
        assert not h.sink.of_type(EventType.RUN_COMPLETED)

    run(scenario())


@pytest.mark.parametrize(
    ("run_request", "error_type"),
    [
        (request("  "), ValueError),
        (request(scope=ResourceScope("other-workspace")), PermissionDenied),
    ],
)
def test_invalid_run_is_rejected_without_events_or_model_access(
    run_request: RunRequest, error_type: type[Exception]
) -> None:
    async def scenario() -> None:
        h = build([reply("must not run")])
        frames: list[bytes] = []
        with pytest.raises(error_type):
            async with stream_run(h.engine, run_request, heartbeat_interval=None) as stream:
                async for frame in stream:
                    frames.append(frame)
        assert frames == []
        assert h.sink.events == []
        assert h.model.calls == []

    run(scenario())


def test_approval_suspension_and_resume_stream_without_exposing_snapshot() -> None:
    async def scenario() -> None:
        h = build(
            [call_tools(tool_call("update_document", {"document_id": "d1", "content": "v2"}))]
        )
        async with stream_run(
            h.engine,
            request("Update d1", policy=ToolPolicy(AccessLevel.CONFIRM)),
            heartbeat_interval=None,
        ) as stream:
            suspended = await collect(stream)
            first = await stream.result()
        assert first.status is RunStatus.SUSPENDED
        assert h.workspace.writes == []
        assert [p["type"] for p in suspended[-2:]] == ["prompt_required", "run_suspended"]
        prompt = first.prompt
        assert prompt is not None
        assert suspended[-2]["data"]["prompt"] == prompt.public_view()
        assert "snapshot" not in json.dumps(suspended)
        h.broker.resolve(EDITOR, prompt.id, {"approved": True}, session_id="s1")
        h.model.extend([reply("Updated d1.")])
        async with stream_resume(
            h.engine, EDITOR, prompt.id, session_id="s1", heartbeat_interval=None
        ) as stream:
            resumed = await collect(stream)
            second = await stream.result()
        assert resumed[0]["type"] == "run_resumed"
        assert resumed[-1]["type"] == "run_completed"
        assert "tool_result" in [p["type"] for p in resumed]
        assert all(p["run_id"] == first.run_id for p in resumed)
        assert second.status is RunStatus.COMPLETED
        assert h.workspace.writes == [("d1", "v2")]
        stored = h.broker.store.get(prompt.id)
        assert stored is not None and stored.status is PromptStatus.DONE

    run(scenario())


@pytest.mark.parametrize("wrong_owner", [True, False])
def test_unauthorized_resume_emits_nothing_and_does_not_claim_prompt(wrong_owner: bool) -> None:
    async def scenario() -> None:
        h = build(
            [call_tools(tool_call("update_document", {"document_id": "d1", "content": "v2"}))]
        )
        first = await h.engine.run(request(policy=ToolPolicy(AccessLevel.CONFIRM)))
        assert first.prompt is not None
        prompt_id = first.prompt.id
        h.broker.resolve(EDITOR, prompt_id, {"approved": True})
        h.sink.events.clear()
        frames: list[bytes] = []
        with pytest.raises(PromptNotFound):
            async with stream_resume(
                h.engine,
                OTHER_WORKSPACE if wrong_owner else EDITOR,
                prompt_id,
                session_id="s1" if wrong_owner else "wrong-session",
                heartbeat_interval=None,
            ) as stream:
                async for frame in stream:
                    frames.append(frame)
        assert frames == [] and h.sink.events == []
        assert len(h.model.calls) == 1
        assert h.workspace.writes == []
        stored = h.broker.store.get(prompt_id)
        assert stored is not None and stored.status is PromptStatus.ANSWERED

    run(scenario())


def test_concurrent_streams_with_equal_run_and_session_ids_do_not_cross_talk() -> None:
    async def scenario() -> None:
        entered: set[str] = set()
        both_entered = asyncio.Event()
        release = asyncio.Event()

        class ConcurrentModel:
            async def complete(
                self,
                messages: list[Message],
                tools: list[ToolSpec],
                *,
                max_tokens: int | None = None,
            ) -> ChatResponse:
                query = messages[-1].content
                entered.add(query)
                if len(entered) == 2:
                    both_entered.set()
                await release.wait()
                return reply(f"answer:{query}")

        observer = CollectingSink()
        engine = AgentEngine(ConcurrentModel(), events=observer)
        async with (
            stream_run(engine, request("left", run_id="same"), heartbeat_interval=None) as left,
            stream_run(engine, request("right", run_id="same"), heartbeat_interval=None) as right,
        ):
            await asyncio.wait_for(both_entered.wait(), 1)
            release.set()
            left_events, right_events = await asyncio.gather(collect(left), collect(right))
            assert left_events[0]["data"]["query"] == "left"
            assert right_events[0]["data"]["query"] == "right"
            assert left_events[-2]["data"]["content"] == "answer:left"
            assert right_events[-2]["data"]["content"] == "answer:right"
            assert len(left_events) == len(right_events) == 4
            assert len(observer.events) == 8
            assert (await left.result()).final_answer == "answer:left"
            assert (await right.result()).final_answer == "answer:right"

    run(scenario())


@pytest.mark.parametrize("exit_mode", ["break", "exception", "aclose"])
def test_disposal_cancels_and_awaits_model_cleanup(exit_mode: str) -> None:
    async def scenario() -> None:
        before = asyncio.all_tasks()
        model = WaitingModel()
        with pytest.raises(LookupError) if exit_mode == "exception" else nullcontext():
            async with stream_run(AgentEngine(model), request(), heartbeat_interval=None) as stream:
                await asyncio.wait_for(model.entered.wait(), 1)
                if exit_mode == "exception":
                    raise LookupError("consumer stopped")
                if exit_mode == "aclose":
                    await stream.aclose()
                    assert model.cleaned.is_set()
                else:
                    async for _ in stream:
                        break
        assert model.cancelled and model.cleaned.is_set()
        assert asyncio.all_tasks() == before

    run(scenario())


def test_canceling_a_waiting_consumer_cleans_up_the_producer() -> None:
    async def scenario() -> None:
        before = asyncio.all_tasks()
        model = WaitingModel()
        waiting_for_next = asyncio.Event()

        async def consume() -> None:
            async with stream_run(AgentEngine(model), request(), heartbeat_interval=None) as stream:
                assert decode(await anext(stream))["type"] == "run_started"
                assert decode(await anext(stream))["type"] == "round_started"
                waiting_for_next.set()
                await anext(stream)

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(waiting_for_next.wait(), 1)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert model.cancelled and model.cleaned.is_set()
        assert asyncio.all_tasks() == before

    run(scenario())


@pytest.mark.parametrize("cancel_consumer", [False, True])
def test_repeated_close_and_caller_cancellation_cannot_interrupt_cleanup(
    cancel_consumer: bool,
) -> None:
    async def scenario() -> None:
        before = asyncio.all_tasks()
        entered = asyncio.Event()
        cleanup_started = asyncio.Event()
        cleanup_gate = asyncio.Event()
        cleanup_finished = asyncio.Event()
        streams: list[SSEStream] = []

        class CleanupModel:
            async def complete(
                self,
                messages: list[Message],
                tools: list[ToolSpec],
                *,
                max_tokens: int | None = None,
            ) -> ChatResponse:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup_started.set()
                    await cleanup_gate.wait()
                    cleanup_finished.set()
                return reply("unreachable")

        async def consume() -> None:
            async with stream_run(
                AgentEngine(CleanupModel()), request(), heartbeat_interval=None
            ) as stream:
                streams.append(stream)
                await entered.wait()

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(cleanup_started.wait(), 1)
        if cancel_consumer:
            consumer.cancel()
        close_again = asyncio.create_task(streams[0].aclose())
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not cleanup_finished.is_set()
            assert not consumer.done()
            assert not close_again.done()
        finally:
            cleanup_gate.set()
            results = await asyncio.gather(consumer, close_again, return_exceptions=True)
        assert cleanup_finished.is_set()
        if cancel_consumer:
            assert isinstance(results[0], asyncio.CancelledError)
        else:
            assert results[0] is None
        assert results[1] is None
        assert asyncio.all_tasks() == before

    run(scenario())


def test_disconnect_during_approved_tool_does_not_replay_side_effect() -> None:
    async def scenario() -> None:
        before = asyncio.all_tasks()
        entered = asyncio.Event()
        cleaned = asyncio.Event()
        writes: list[str] = []

        async def write(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            writes.append("saved")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()
            return ToolResult.ok("saved")

        registry = ToolRegistry()
        registry.register(FunctionTool(ToolSpec("write", "Write.", risk=Risk.WRITE), write))
        h = build([call_tools(tool_call("write"))], registry=registry)
        first = await h.engine.run(request(policy=ToolPolicy(AccessLevel.CONFIRM)))
        assert first.prompt is not None
        prompt_id = first.prompt.id
        h.broker.resolve(EDITOR, prompt_id, {"approved": True})
        async with stream_resume(h.engine, EDITOR, prompt_id, heartbeat_interval=None) as stream:
            await asyncio.wait_for(entered.wait(), 1)
            await stream.aclose()
            assert cleaned.is_set()
        assert writes == ["saved"]
        stored = h.broker.store.get(prompt_id)
        assert stored is not None and stored.status is PromptStatus.EXECUTING
        with pytest.raises(PromptAlreadyResolved):
            async with stream_resume(
                h.engine, EDITOR, prompt_id, heartbeat_interval=None
            ) as stream:
                await collect(stream)
        assert writes == ["saved"]
        assert asyncio.all_tasks() == before

    run(scenario())


def test_stream_failure_takes_precedence_over_a_later_producer_exception() -> None:
    async def scenario() -> None:
        h = build([RuntimeError("provider broke")])
        with pytest.raises(SSEStreamError, match="buffer"):
            async with stream_run(
                h.engine, request(), heartbeat_interval=None, max_buffered_events=1
            ) as stream:
                await stream.result()

    run(scenario())


@pytest.mark.parametrize(
    ("content", "options"),
    [
        ("completed immediately", {"max_buffered_events": 1}),
        ("界" * 100, {"max_event_bytes": 350}),
        ("\ud800", {}),
    ],
    ids=["buffer-overflow", "utf8-byte-limit", "invalid-utf8"],
)
def test_transport_failure_cannot_be_mistaken_for_completion(
    content: str, options: dict[str, int]
) -> None:
    async def scenario() -> None:
        before = asyncio.all_tasks()
        h = build([reply(content)])
        with pytest.raises(SSEStreamError):
            async with stream_run(
                h.engine, request(), heartbeat_interval=None, **options
            ) as stream:
                with pytest.raises(SSEStreamError):
                    await collect(stream)
                with pytest.raises(SSEStreamError):
                    await stream.result()
        assert asyncio.all_tasks() == before

    run(scenario())


@pytest.mark.parametrize(
    "options",
    [
        {"heartbeat_interval": 0},
        {"heartbeat_interval": -1},
        {"heartbeat_interval": float("inf")},
        {"heartbeat_interval": float("nan")},
        {"heartbeat_interval": True},
        {"max_buffered_events": 0},
        {"max_buffered_events": -1},
        {"max_buffered_events": 1.5},
        {"max_buffered_events": True},
        {"max_event_bytes": 0},
        {"max_event_bytes": -1},
        {"max_event_bytes": 1.5},
        {"max_event_bytes": True},
    ],
)
def test_invalid_transport_options_are_rejected_before_running(options: dict[str, Any]) -> None:
    async def scenario() -> None:
        h = build([reply("must not run")])
        with pytest.raises((TypeError, ValueError)):
            async with stream_run(h.engine, request(), **options):
                pytest.fail("invalid run options accepted")
        with pytest.raises((TypeError, ValueError)):
            async with stream_resume(h.engine, EDITOR, "prompt-id", **options):
                pytest.fail("invalid resume options accepted")
        assert h.model.calls == [] and h.sink.events == []

    run(scenario())
