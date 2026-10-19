"""Incremental provider decoding and real engine/SSE lifecycle contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from support import EDITOR, FakeWorkspace, run

from magent_harness import (
    AccessLevel,
    AgentConfig,
    AgentEngine,
    Caller,
    CollectingSink,
    ContextOverflowError,
    DeepSeekChatModel,
    EventType,
    FunctionTool,
    InMemoryPromptStore,
    Message,
    ModelError,
    PromptBroker,
    PromptStatus,
    Risk,
    Role,
    RunRequest,
    RunSnapshot,
    RunStatus,
    SSEStreamError,
    ToolContext,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    TransientModelError,
    stream_resume,
    stream_run,
)

TEST_KEY = "synthetic-stream-test-key"
PRIVATE_REASONING = "private stream reasoning 中文"
DONE = b"data: [DONE]\n\n"
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def frame(body: Any) -> bytes:
    return b"data: " + json.dumps(body, ensure_ascii=False).encode() + b"\n\n"


def chunk(
    delta: dict[str, Any] | None = None,
    *,
    finish: str | None = None,
    usage: dict[str, Any] | None = None,
) -> bytes:
    body: dict[str, Any] = {
        "id": "provider-response",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }
    if usage is not None:
        body["usage"] = usage
    return frame(body)


def call_fragment(
    index: int = 0,
    *,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str = "",
) -> dict[str, Any]:
    item: dict[str, Any] = {"index": index, "function": {"arguments": arguments}}
    if call_id is not None:
        item.update(id=call_id, type="function")
    if name is not None:
        item["function"]["name"] = name
    return item


class ByteStream(httpx.AsyncByteStream):
    """Provider byte chunks, gates and failures with observable cleanup."""

    def __init__(self, *actions: bytes | asyncio.Event | BaseException) -> None:
        self.actions = actions
        self.blocked = asyncio.Event()
        self.closed = asyncio.Event()
        self.yielded = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for action in self.actions:
            if isinstance(action, asyncio.Event):
                self.blocked.set()
                await action.wait()
            elif isinstance(action, BaseException):
                raise action
            else:
                self.yielded += 1
                yield action
                await asyncio.sleep(0)

    async def aclose(self) -> None:
        self.closed.set()


class ScriptedHTTP:
    def __init__(self, *responses: ByteStream | httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert self.responses, "unexpected provider request"
        response = self.responses.pop(0)
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=response)

    def body(self, index: int) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def answer_stream(content: str = "finished", *, reasoning: str | None = None) -> ByteStream:
    delta: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return ByteStream(chunk(delta), chunk(finish="stop", usage=USAGE), DONE)


def request(**kwargs: Any) -> RunRequest:
    return RunRequest(EDITOR, "stream-session", "Read the roadmap", **kwargs)


def decode_event(frame_bytes: bytes) -> dict[str, Any]:
    lines = frame_bytes.decode().splitlines()
    assert lines[0].startswith("event: ")
    payload: dict[str, Any] = json.loads(lines[1][6:])
    assert lines[0][7:] == payload["type"]
    return payload


def streaming_config(**kwargs: Any) -> AgentConfig:
    return AgentConfig(stream_model=True, model_retry_delay_s=0, **kwargs)


def test_text_callback_arrives_before_provider_finishes_and_wire_enables_streaming() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        source = ByteStream(
            chunk({"role": "assistant", "content": "先到"}),
            release,
            chunk({"content": "后到"}),
            chunk(finish="stop", usage=USAGE),
            DONE,
        )
        http = ScriptedHTTP(source)
        deltas: list[str] = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            async with DeepSeekChatModel(
                api_key=TEST_KEY, base_url="https://provider.example/v1", client=client
            ) as model:
                task = asyncio.create_task(
                    model.complete_stream(
                        [Message("user", "hello")], [], on_delta=deltas.append, max_tokens=123
                    )
                )
                try:
                    await asyncio.wait_for(source.blocked.wait(), 1)
                    assert deltas == ["先到"]
                    assert not task.done()
                    release.set()
                    result = await task
                finally:
                    release.set()
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                assert deltas == ["先到", "后到"]
                assert result.content == "先到后到"
                assert result.usage.total_tokens == 18
                assert source.closed.is_set()
            assert not client.is_closed
        assert str(http.requests[0].url) == "https://provider.example/v1/chat/completions"
        assert http.requests[0].headers["authorization"] == f"Bearer {TEST_KEY}"
        body = http.body(0)
        assert body["stream"] is True
        assert body["stream_options"]["include_usage"] is True
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_tokens"] == 123

    run(scenario())


def test_sse_utf8_byte_splits_comments_crlf_and_multiple_frames() -> None:
    raw = (
        b": provider keepalive\r\n\r\n\n"
        + chunk({"role": "assistant", "content": "你好🌍"}).replace(b"\n", b"\r\n")
        + b": comment\n\n"
        + chunk({"content": "\n下一行"})
        + chunk(finish="stop")
        + DONE
    )
    source = ByteStream(*(bytes([byte]) for byte in raw))
    http = ScriptedHTTP(source)

    async def scenario() -> None:
        deltas: list[str] = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            result = await model.complete_stream([], [], on_delta=deltas.append)
        assert result.content == "你好🌍\n下一行"
        assert "".join(deltas) == result.content
        assert result.usage.total_tokens == 0
        assert source.closed.is_set()

    run(scenario())


def test_interleaved_tool_argument_fragments_are_assembled_in_index_order() -> None:
    source = ByteStream(
        chunk(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    call_fragment(1, call_id="c2", name="read_document", arguments='{"document_'),
                    call_fragment(
                        0, call_id="c1", name="read_document", arguments='{"document_id":'
                    ),
                ],
            }
        ),
        chunk({"tool_calls": [call_fragment(0, arguments='"d1"}')]}),
        chunk({"tool_calls": [call_fragment(1, arguments='id":"d2"}')]}),
        chunk(finish="tool_calls", usage=USAGE),
        DONE,
    )
    http = ScriptedHTTP(source)

    async def scenario() -> None:
        deltas: list[str] = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            result = await model.complete_stream(
                [], [ToolSpec("read_document", "Read")], on_delta=deltas.append
            )
        assert [(call.id, call.name, json.loads(call.arguments)) for call in result.tool_calls] == [
            ("c1", "read_document", {"document_id": "d1"}),
            ("c2", "read_document", {"document_id": "d2"}),
        ]
        assert deltas == []
        assert result.finish_reason == "tool_calls"

    run(scenario())


def test_thinking_is_accumulated_and_preserved_without_text_callback_leaks() -> None:
    source = ByteStream(
        chunk({"role": "assistant", "reasoning_content": "private stream "}),
        chunk({"reasoning_content": "reasoning 中文", "content": None}),
        chunk({"content": "public "}),
        chunk({"content": "answer"}),
        chunk(finish="stop"),
        DONE,
    )
    http = ScriptedHTTP(source)

    async def scenario() -> None:
        deltas: list[str] = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, thinking=True, client=client)
            result = await model.complete_stream(
                [Message("assistant", "previous", reasoning_content="previous private")],
                [ToolSpec("read_document", "Read")],
                on_delta=deltas.append,
            )
        assert result.reasoning_content == PRIVATE_REASONING
        assert result.content == "public answer"
        assert deltas == ["public ", "answer"]
        assert http.body(0)["messages"][0]["reasoning_content"] == "previous private"

    run(scenario())


@pytest.mark.parametrize("usage_only", [False, True])
def test_usage_on_terminal_or_separate_usage_chunk_is_not_counted_twice(usage_only: bool) -> None:
    pieces = [
        chunk({"content": "answer"}),
        chunk(finish="stop", usage=None if usage_only else USAGE),
    ]
    if usage_only:
        pieces.append(frame({"choices": [], "usage": USAGE}))
    http = ScriptedHTTP(ByteStream(*pieces, DONE))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            result = await DeepSeekChatModel(api_key=TEST_KEY, client=client).complete_stream(
                [], [], on_delta=lambda _: None
            )
        assert result.usage.prompt_tokens == 11
        assert result.usage.completion_tokens == 7

    run(scenario())


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (400, "context_length_exceeded", ContextOverflowError),
        (422, "context_window_exceeded", ContextOverflowError),
        (401, "invalid_api_key", ModelError),
        (429, "rate_limit", TransientModelError),
        (503, "unavailable", TransientModelError),
    ],
)
def test_non_200_stream_response_uses_redacted_error_mapping(
    status: int, code: str, expected: type[ModelError]
) -> None:
    http = ScriptedHTTP(httpx.Response(status, json={"error": {"code": code, "message": TEST_KEY}}))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            with pytest.raises(expected) as caught:
                await model.complete_stream([], [], on_delta=lambda _: None)
            assert type(caught.value) is expected
            assert TEST_KEY not in str(caught.value)
        assert len(http.requests) == 1

    run(scenario())


@pytest.mark.parametrize(
    "pieces",
    [
        [chunk({"content": "unfinished"})],
        [chunk({"content": "unfinished"}), DONE],
        [chunk({"content": "unfinished"}), chunk(finish="stop")],
        [chunk({"content": "unfinished"}), httpx.ReadError("private provider transport data")],
    ],
    ids=["no-terminal", "no-finish", "no-done", "read-error"],
)
def test_interrupted_stream_never_returns_a_successful_response(pieces: list[Any]) -> None:
    source = ByteStream(*pieces)
    http = ScriptedHTTP(source)

    async def scenario() -> None:
        deltas: list[str] = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            with pytest.raises(ModelError) as caught:
                await model.complete_stream([], [], on_delta=deltas.append)
            assert "private provider transport data" not in str(caught.value)
        assert deltas == ["unfinished"]
        assert source.closed.is_set()

    run(scenario())


@pytest.mark.parametrize(
    "invalid_frame",
    [
        b"data: {not-json}\n\n",
        frame({"error": {"message": TEST_KEY}}),
        chunk({"content": 3}),
        chunk({"reasoning_content": ["private"]}),
        chunk({"role": "user"}),
        chunk({"tool_calls": [{"index": -1, "function": {"arguments": "{}"}}]}),
        chunk({"tool_calls": [{"index": True, "function": {"arguments": "{}"}}]}),
        chunk({"tool_calls": [{"function": {"arguments": "{}"}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": {}}}]}),
        frame({"choices": [{"index": 1, "delta": {"content": "wrong choice"}}]}),
        chunk(finish="stop", usage={"prompt_tokens": "many", "completion_tokens": 1}),
    ],
)
def test_malformed_stream_chunks_fail_with_safe_error(invalid_frame: bytes) -> None:
    source = ByteStream(invalid_frame, chunk(finish="stop"), DONE)
    http = ScriptedHTTP(source)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            with pytest.raises(ModelError) as caught:
                await model.complete_stream([], [], on_delta=lambda _: None)
            assert TEST_KEY not in str(caught.value)
        assert source.closed.is_set()

    run(scenario())


def test_total_deadline_expires_despite_regular_keepalives_and_closes_response() -> None:
    class KeepaliveStream(ByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            while True:
                self.yielded += 1
                yield b": still thinking\n\n"
                await asyncio.sleep(0.001)

    source = KeepaliveStream()
    http = ScriptedHTTP(source)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, timeout_s=0.02, client=client)
            with pytest.raises(TransientModelError, match="timed out"):
                await model.complete_stream([], [], on_delta=lambda _: None)
            assert not client.is_closed
        assert source.yielded > 1
        assert source.closed.is_set()

    run(scenario())


def test_cancel_propagates_and_closes_response_but_keeps_borrowed_client_open() -> None:
    async def scenario() -> None:
        source = ByteStream(chunk({"content": "partial"}), asyncio.Event())
        http = ScriptedHTTP(source)
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            async with DeepSeekChatModel(api_key=TEST_KEY, client=client) as model:
                task = asyncio.create_task(model.complete_stream([], [], on_delta=lambda _: None))
                await asyncio.wait_for(source.blocked.wait(), 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert source.closed.is_set()
            assert not client.is_closed

    run(scenario())


def test_real_engine_sse_emits_text_before_done_and_executes_only_complete_tool_calls() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        first = ByteStream(
            chunk(
                {
                    "role": "assistant",
                    "content": "正在读取",
                    "tool_calls": [
                        call_fragment(
                            call_id="c1", name="read_document", arguments='{"document_id":'
                        )
                    ],
                }
            ),
            chunk({"tool_calls": [call_fragment(arguments='"d1"}')]}),
            chunk(finish="tool_calls", usage=USAGE),
            release,
            DONE,
        )
        http = ScriptedHTTP(first, answer_stream("Q3 ships search."))
        workspace = FakeWorkspace.default()
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, client=client),
                workspace.registry(),
                config=streaming_config(),
            )
            async with stream_run(engine, request(), heartbeat_interval=None) as stream:
                payloads: list[dict[str, Any]] = []
                while not any(event["type"] == "text_delta" for event in payloads):
                    payloads.append(decode_event(await asyncio.wait_for(anext(stream), 1)))
                await asyncio.wait_for(first.blocked.wait(), 1)
                assert workspace.reads == []
                assert len(http.requests) == 1
                assert not first.closed.is_set()
                release.set()
                payloads.extend([decode_event(value) async for value in stream])
                result = await stream.result()
        assert result.status is RunStatus.COMPLETED
        assert result.stop_reason == "answer"
        assert result.final_answer == "Q3 ships search."
        assert workspace.reads == ["d1"]
        assert [event["type"] for event in payloads] == [
            "run_started",
            "round_started",
            "model_response_started",
            "text_delta",
            "model_response_completed",
            "thought",
            "tool_call",
            "tool_result",
            "round_started",
            "model_response_started",
            "text_delta",
            "model_response_completed",
            "final_answer",
            "run_completed",
        ]
        started = [event["data"] for event in payloads if event["type"] == "model_response_started"]
        completed = [
            event["data"] for event in payloads if event["type"] == "model_response_completed"
        ]
        deltas = [event["data"] for event in payloads if event["type"] == "text_delta"]
        assert len({event["response_id"] for event in started}) == 2
        assert [event["round"] for event in started] == [0, 1]
        assert [event["response_id"] for event in completed] == [
            event["response_id"] for event in started
        ]
        assert [event["response_id"] for event in deltas] == [
            event["response_id"] for event in started
        ]
        assert [event["kind"] for event in completed] == ["tool", "answer"]
        sent_history = http.body(1)["messages"]
        assert sent_history[-2]["tool_calls"][0]["function"]["arguments"] == '{"document_id":"d1"}'
        assert sent_history[-1]["tool_call_id"] == "c1"
        assert result.usage.total_tokens == 36

    run(scenario())


def test_partial_attempt_is_discarded_before_retry_without_polluting_history_or_tools() -> None:
    first = ByteStream(
        chunk(
            {
                "content": "discard this",
                "tool_calls": [
                    call_fragment(
                        call_id="c1", name="read_document", arguments='{"document_id":"d1"}'
                    )
                ],
            }
        ),
        httpx.ReadError("provider interrupted"),
    )
    http = ScriptedHTTP(first, answer_stream("clean answer"))
    workspace = FakeWorkspace.default()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, client=client),
                workspace.registry(),
                config=streaming_config(max_model_retries=1),
            )
            async with stream_run(engine, request(), heartbeat_interval=None) as stream:
                payloads = [decode_event(value) async for value in stream]
                result = await stream.result()
        assert result.status is RunStatus.COMPLETED
        assert result.final_answer == "clean answer"
        assert result.steps == []
        assert workspace.reads == []
        assert "discard this" not in json.dumps(http.body(1))
        assert "discard this" not in "".join(message.content for message in result.messages)
        lifecycle = [event for event in payloads if event["type"].startswith("model_response_")]
        assert [event["type"] for event in lifecycle] == [
            "model_response_started",
            "model_response_discarded",
            "model_response_started",
            "model_response_completed",
        ]
        assert lifecycle[0]["data"]["response_id"] == lifecycle[1]["data"]["response_id"]
        assert lifecycle[2]["data"]["response_id"] == lifecycle[3]["data"]["response_id"]
        assert lifecycle[0]["data"]["response_id"] != lifecycle[2]["data"]["response_id"]
        assert lifecycle[1]["data"]["reason"] == "error"
        assert lifecycle[0]["data"]["round"] == lifecycle[2]["data"]["round"] == 0
        assert first.closed.is_set()

    run(scenario())


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_truncated_or_filtered_stream_never_executes_tools(finish: str) -> None:
    first = ByteStream(
        chunk(
            {
                "content": "do not commit this",
                "tool_calls": [
                    call_fragment(
                        call_id="c1", name="read_document", arguments='{"document_id":"d1"}'
                    )
                ],
            }
        ),
        chunk(finish=finish),
        DONE,
    )
    http = ScriptedHTTP(first, answer_stream("safe answer"))
    workspace = FakeWorkspace.default()
    sink = CollectingSink()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, client=client),
                workspace.registry(),
                config=streaming_config(),
                events=sink,
            )
            result = await engine.run(request())
        assert result.status is RunStatus.COMPLETED
        assert workspace.reads == []
        discarded = sink.of_type(EventType.MODEL_RESPONSE_DISCARDED)
        assert len(discarded) == 1
        assert discarded[0].data["reason"] == finish
        if finish == "length":
            assert result.final_answer == "safe answer"
            assert not result.steps[0].tool_calls[0].result.success
        else:
            assert result.stop_reason == "content_filter"
            assert len(http.requests) == 1

    run(scenario())


@pytest.mark.parametrize("revoke_permission", [False, True])
def test_thinking_stream_approval_survives_fresh_engine_without_reasoning_leaks(
    revoke_permission: bool,
) -> None:
    first = ByteStream(
        chunk({"role": "assistant", "reasoning_content": "private stream "}),
        chunk({"reasoning_content": "reasoning 中文"}),
        chunk(
            {
                "content": "I will update the roadmap.",
                "tool_calls": [
                    call_fragment(
                        call_id="write1", name="update_document", arguments='{"document_id":"d1",'
                    )
                ],
            }
        ),
        chunk({"tool_calls": [call_fragment(arguments='"content":"v2"}')]}),
        chunk(finish="tool_calls", usage=USAGE),
        DONE,
    )
    http = ScriptedHTTP(first, answer_stream("finished", reasoning="private final"))
    workspace = FakeWorkspace.default()
    store = InMemoryPromptStore()
    sink = CollectingSink()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, thinking=True, client=client),
                workspace.registry(),
                PromptBroker(store),
                config=streaming_config(),
                events=sink,
            )
            async with stream_run(
                engine, request(policy=ToolPolicy(AccessLevel.CONFIRM)), heartbeat_interval=None
            ) as stream:
                frames = [value async for value in stream]
                suspended = await stream.result()
            assert suspended.status is RunStatus.SUSPENDED
            prompt = suspended.prompt
            assert prompt is not None
            assert workspace.writes == []
            snapshot = RunSnapshot.decode(prompt.snapshot)
            assert snapshot.pending.reasoning_content == PRIVATE_REASONING
            PromptBroker(store).resolve(EDITOR, prompt.id, {"approved": True})
            fresh = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, thinking=True, client=client),
                workspace.registry(),
                PromptBroker(store),
                config=streaming_config(),
                events=sink,
            )
            caller = (
                Caller(EDITOR.workspace_id, EDITOR.user_id, Role.VIEWER)
                if revoke_permission
                else EDITOR
            )
            async with stream_resume(fresh, caller, prompt.id, heartbeat_interval=None) as stream:
                frames.extend([value async for value in stream])
                result = await stream.result()
        assert result.status is RunStatus.COMPLETED
        assert result.final_answer == "finished"
        assert PRIVATE_REASONING.encode() not in b"".join(frames)
        assert b"private final" not in b"".join(frames)
        assert b"reasoning_content" not in b"".join(frames)
        assert b"snapshot" not in b"".join(frames)
        assert PRIVATE_REASONING not in json.dumps(
            [event.data for event in sink.events], ensure_ascii=False
        )
        assert http.body(1)["messages"][-2]["reasoning_content"] == PRIVATE_REASONING
        assert http.body(1)["messages"][-1]["tool_call_id"] == "write1"
        assert result.messages[-1].reasoning_content == "private final"
        saved = store.get(prompt.id)
        assert saved is not None
        assert saved.status is (PromptStatus.FAILED if revoke_permission else PromptStatus.DONE)
        assert workspace.writes == ([] if revoke_permission else [("d1", "v2")])
        if revoke_permission:
            assert result.steps[0].tool_calls[0].result.error.startswith("not permitted")

    run(scenario())


def test_sse_disconnect_cancels_active_provider_read_and_closes_response() -> None:
    async def scenario() -> None:
        source = ByteStream(chunk({"content": "partial answer"}), asyncio.Event())
        http = ScriptedHTTP(source)
        sink = CollectingSink()
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            engine = AgentEngine(model, config=streaming_config(), events=sink)
            async with stream_run(engine, request(), heartbeat_interval=None) as stream:
                while (
                    decode_event(await asyncio.wait_for(anext(stream), 1))["type"] != "text_delta"
                ):
                    pass
                await asyncio.wait_for(source.blocked.wait(), 1)
                await stream.aclose()
                assert source.closed.is_set()
                with pytest.raises(asyncio.CancelledError):
                    await stream.result()
            assert not client.is_closed
        assert sink.events[-1].type is EventType.RUN_CANCELED
        assert sink.of_type(EventType.MODEL_RESPONSE_DISCARDED)[0].data["reason"] == "canceled"
        assert not sink.of_type(EventType.MODEL_RESPONSE_COMPLETED)

    run(scenario())


def test_large_provider_packet_yields_to_fast_sse_consumer_without_false_overflow() -> None:
    packet = b"".join(chunk({"content": "字"}) for _ in range(300)) + chunk(finish="stop") + DONE
    source = ByteStream(packet)
    http = ScriptedHTTP(source)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, client=client), config=streaming_config()
            )
            async with stream_run(engine, request(), heartbeat_interval=None) as stream:
                payloads = [decode_event(value) async for value in stream]
                result = await stream.result()
        deltas = [event["data"]["content"] for event in payloads if event["type"] == "text_delta"]
        assert len(deltas) == 300
        assert result.final_answer == "字" * 300
        assert result.status is RunStatus.COMPLETED
        assert source.closed.is_set()

    run(scenario())


def test_sse_overflow_closes_http_before_any_tool_or_approval_side_effect() -> None:
    packet = (
        chunk(
            {
                "tool_calls": [
                    call_fragment(
                        call_id="write1",
                        name="update_document",
                        arguments='{"document_id":"d1","content":"v2"}',
                    )
                ]
            }
        )
        + b"".join(chunk({"content": "x"}) for _ in range(20))
        + chunk(finish="tool_calls")
        + DONE
    )
    source = ByteStream(packet)
    http = ScriptedHTTP(source)
    workspace = FakeWorkspace.default()
    sink = CollectingSink()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, client=client),
                workspace.registry(),
                config=streaming_config(),
                events=sink,
            )
            with pytest.raises(SSEStreamError, match="buffer"):
                async with stream_run(
                    engine,
                    request(policy=ToolPolicy(AccessLevel.CONFIRM)),
                    max_buffered_events=8,
                    heartbeat_interval=None,
                ) as stream:
                    await asyncio.wait_for(source.closed.wait(), 1)
                    await stream.result()
            assert not client.is_closed
        assert workspace.writes == []
        assert sink.of_type(EventType.TOOL_CALL) == []
        assert sink.of_type(EventType.PROMPT_REQUIRED) == []
        assert not sink.of_type(EventType.RUN_COMPLETED)

    run(scenario())


@pytest.mark.parametrize("tool_name", ["read_document", "update_document"])
def test_thinking_stream_without_reasoning_fails_before_tools_or_approval(tool_name: str) -> None:
    arguments = {"document_id": "d1"}
    if tool_name == "update_document":
        arguments["content"] = "forbidden update"
    source = ByteStream(
        chunk(
            {
                "tool_calls": [
                    call_fragment(call_id="c1", name=tool_name, arguments=json.dumps(arguments))
                ]
            }
        ),
        chunk(finish="tool_calls"),
        DONE,
    )
    http = ScriptedHTTP(source)
    workspace = FakeWorkspace.default()
    sink = CollectingSink()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, thinking=True, client=client),
                workspace.registry(),
                config=streaming_config(),
                events=sink,
            )
            result = await engine.run(request(policy=ToolPolicy(AccessLevel.CONFIRM)))
        assert result.status is RunStatus.FAILED
        assert "reasoning_content" in result.error
        assert result.prompt is None
        assert workspace.reads == workspace.writes == []
        assert not sink.of_type(EventType.TOOL_CALL)
        assert not sink.of_type(EventType.PROMPT_REQUIRED)
        assert not sink.of_type(EventType.MODEL_RESPONSE_COMPLETED)
        assert sink.of_type(EventType.MODEL_RESPONSE_DISCARDED)[0].data["reason"] == "error"

    run(scenario())


def test_stream_does_not_follow_redirect_or_use_borrowed_clients_default_auth() -> None:
    http = ScriptedHTTP(httpx.Response(302, headers={"location": "https://other.example/receive"}))

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(http),
            follow_redirects=True,
            auth=httpx.BasicAuth("unrelated", "unrelated"),
        ) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            with pytest.raises(ModelError):
                await model.complete_stream([], [], on_delta=lambda _: None)
        assert len(http.requests) == 1
        assert http.requests[0].url.host == "api.deepseek.com"
        assert http.requests[0].headers["authorization"] == f"Bearer {TEST_KEY}"

    run(scenario())


@pytest.mark.parametrize("field", ["id", "name", "type"])
def test_conflicting_tool_fragment_metadata_is_rejected(field: str) -> None:
    altered = call_fragment(arguments='"d1"}')
    if field == "name":
        altered["function"]["name"] = "update_document"
    else:
        altered[field] = "different"
    source = ByteStream(
        chunk(
            {
                "tool_calls": [
                    call_fragment(call_id="c1", name="read_document", arguments='{"document_id":')
                ]
            }
        ),
        chunk({"tool_calls": [altered]}),
        chunk(finish="tool_calls"),
        DONE,
    )
    http = ScriptedHTTP(source)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            with pytest.raises(ModelError):
                await model.complete_stream([], [], on_delta=lambda _: None)
        assert source.closed.is_set()

    run(scenario())


def test_repeated_stable_tool_metadata_is_not_concatenated() -> None:
    source = ByteStream(
        chunk(
            {
                "tool_calls": [
                    call_fragment(call_id="c1", name="read_document", arguments='{"document_id":')
                ]
            }
        ),
        chunk(
            {"tool_calls": [call_fragment(call_id="c1", name="read_document", arguments='"d1"}')]}
        ),
        chunk(finish="tool_calls"),
        DONE,
    )
    http = ScriptedHTTP(source)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            response = await model.complete_stream([], [], on_delta=lambda _: None)
        assert response.tool_calls[0].id == "c1"
        assert response.tool_calls[0].name == "read_document"
        assert response.tool_calls[0].arguments == '{"document_id":"d1"}'

    run(scenario())


def test_duplicate_call_ids_cannot_execute_ambiguous_tool_results() -> None:
    source = ByteStream(
        chunk(
            {
                "tool_calls": [
                    call_fragment(
                        0,
                        call_id="duplicate",
                        name="read_document",
                        arguments='{"document_id":"d1"}',
                    ),
                    call_fragment(
                        1,
                        call_id="duplicate",
                        name="read_document",
                        arguments='{"document_id":"d2"}',
                    ),
                ]
            }
        ),
        chunk(finish="tool_calls"),
        DONE,
    )
    http = ScriptedHTTP(source)
    workspace = FakeWorkspace.default()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, client=client),
                workspace.registry(),
                config=streaming_config(),
            )
            result = await engine.run(request())
        assert result.status is RunStatus.FAILED
        assert workspace.reads == []
        assert result.steps == []

    run(scenario())


@pytest.mark.parametrize("limit", ["event", "stream", "tools"])
def test_bounded_provider_stream_rejects_excess_before_execution(
    limit: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from magent_harness import _deepseek_stream

    if limit == "event":
        monkeypatch.setattr(_deepseek_stream, "MAX_EVENT_BYTES", 256)
        source = ByteStream(chunk({"content": "x" * 512}), chunk(finish="stop"), DONE)
    elif limit == "stream":
        monkeypatch.setattr(_deepseek_stream, "MAX_STREAM_BYTES", 256)
        source = ByteStream(*(b": keepalive\n\n" for _ in range(30)))
    else:
        monkeypatch.setattr(_deepseek_stream, "MAX_TOOL_CALLS", 1)
        source = ByteStream(
            chunk(
                {
                    "tool_calls": [
                        call_fragment(
                            1, call_id="c2", name="read_document", arguments='{"document_id":"d1"}'
                        )
                    ]
                }
            ),
            chunk(finish="tool_calls"),
            DONE,
        )
    http = ScriptedHTTP(source)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            with pytest.raises(ModelError):
                await model.complete_stream([], [], on_delta=lambda _: None)
        assert source.closed.is_set()

    run(scenario())


def test_text_callback_error_propagates_and_closes_response() -> None:
    error = RuntimeError("application callback failed")
    source = answer_stream()
    http = ScriptedHTTP(source)

    def fail(_: str) -> None:
        raise error

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            with pytest.raises(RuntimeError) as caught:
                await model.complete_stream([], [], on_delta=fail)
            assert caught.value is error
            assert source.closed.is_set()
            assert not client.is_closed

    run(scenario())


@pytest.mark.parametrize("access", [AccessLevel.AUTO_WRITE, AccessLevel.CONFIRM])
@pytest.mark.parametrize("arguments_mode", ["missing", "null", "explicit-empty"])
def test_missing_arguments_never_turn_into_an_executable_empty_write(
    access: AccessLevel,
    arguments_mode: str,
) -> None:
    function: dict[str, Any] = {"name": "touch_document"}
    if arguments_mode != "missing":
        function["arguments"] = None if arguments_mode == "null" else ""
    source = ByteStream(
        chunk(
            {"tool_calls": [{"index": 0, "id": "touch1", "type": "function", "function": function}]}
        ),
        chunk(finish="tool_calls"),
        DONE,
    )
    http = ScriptedHTTP(source, answer_stream())
    writes: list[str] = []
    sink = CollectingSink()
    registry = ToolRegistry()

    async def touch(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        ctx.scope.require_document(EDITOR.workspace_id, "d1")
        writes.append("d1")
        return ToolResult.ok("touched")

    registry.register(
        FunctionTool(
            ToolSpec(
                "touch_document",
                "Touch the active document.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                risk=Risk.WRITE,
            ),
            touch,
        )
    )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, client=client),
                registry,
                config=streaming_config(),
                events=sink,
            )
            result = await engine.run(request(policy=ToolPolicy(access)))
        assert source.closed.is_set()
        if arguments_mode != "explicit-empty":
            assert result.status is RunStatus.FAILED
            assert result.prompt is None
            assert writes == []
            assert len(http.requests) == 1
            assert not sink.of_type(EventType.TOOL_CALL)
            assert not sink.of_type(EventType.PROMPT_REQUIRED)
        elif access is AccessLevel.CONFIRM:
            assert result.status is RunStatus.SUSPENDED
            assert result.prompt is not None
            assert writes == []
        else:
            assert result.status is RunStatus.COMPLETED
            assert writes == ["d1"]

    run(scenario())
