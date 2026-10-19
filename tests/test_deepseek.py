"""DeepSeek's HTTP boundary and its real engine/tool/approval integration."""

from __future__ import annotations

import asyncio
import json
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
    InMemoryPromptStore,
    Message,
    ModelError,
    PromptBroker,
    PromptStatus,
    Role,
    RunRequest,
    RunSnapshot,
    RunStatus,
    ToolCall,
    ToolPolicy,
    ToolSpec,
    TransientModelError,
    stream_resume,
    stream_run,
)

TEST_KEY = "not-a-real-deepseek-key"
PRIVATE_REASONING = "private provider reasoning: preserve exactly\n\u4e2d\u6587"


def completion(
    content: str | None = "answer",
    *,
    calls: list[dict[str, Any]] | None = None,
    reasoning: str | None = None,
    finish: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls is not None:
        message["tool_calls"] = calls
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [
            {"message": message, "finish_reason": finish or ("tool_calls" if calls else "stop")}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def wire_call(name: str = "read_document", **arguments: Any) -> dict[str, Any]:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {"document_id": "d1"})},
    }


class ScriptedHTTP:
    def __init__(self, *responses: httpx.Response | BaseException) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert self.responses, "unexpected extra provider request"
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def body(self, index: int) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def request(**kwargs: Any) -> RunRequest:
    return RunRequest(EDITOR, "deepseek-session", "Read the roadmap", **kwargs)


def test_wire_mapping_uses_provider_schema_and_excludes_harness_metadata() -> None:
    http = ScriptedHTTP(httpx.Response(200, json=completion()))
    spec = ToolSpec("read_document", "Read a document", {"type": "object", "properties": {}})
    call = ToolCall("c1", "read_document", '{"document_id":"d1"}')
    messages = [
        Message("system", "system"),
        Message("user", "summary", kind="compaction_summary"),
        Message("assistant", "", (call,), reasoning_content=PRIVATE_REASONING),
        Message("tool", "document", tool_call_id="c1", name="read_document"),
    ]

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(
                api_key=TEST_KEY,
                base_url="https://provider.example/v1/",
                model="configured-model",
                client=client,
            )
            response = await model.complete(messages, [spec], max_tokens=123)
        assert response.content == "answer"
        assert response.usage.prompt_tokens == 11
        assert response.usage.completion_tokens == 7

    run(scenario())
    sent = http.requests[0]
    body = http.body(0)
    assert sent.method == "POST"
    assert str(sent.url) == "https://provider.example/v1/chat/completions"
    assert sent.headers["authorization"] == f"Bearer {TEST_KEY}"
    assert body["model"] == "configured-model"
    assert body["stream"] is False
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 123
    assert body["tools"] == [{"type": "function", "function": spec.to_schema()}]
    assert body["messages"][2]["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }
    ]
    assert body["messages"][3]["tool_call_id"] == "c1"
    assert not any(
        "kind" in message or "reasoning_content" in message for message in body["messages"]
    )
    assert messages[2].reasoning_content == PRIVATE_REASONING


@pytest.mark.parametrize("thinking", [False, True])
def test_real_engine_reads_a_document_then_answers(thinking: bool) -> None:
    http = ScriptedHTTP(
        httpx.Response(
            200,
            json=completion(
                None, calls=[wire_call()], reasoning=PRIVATE_REASONING if thinking else None
            ),
        ),
        httpx.Response(
            200,
            json=completion("Q3 ships search.", reasoning="final reasoning" if thinking else None),
        ),
    )
    workspace = FakeWorkspace.default()
    sink = CollectingSink()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, thinking=thinking, client=client)
            engine = AgentEngine(model, workspace.registry(), events=sink)
            result = await engine.run(request())
        assert result.status is RunStatus.COMPLETED
        assert result.final_answer == "Q3 ships search."
        assert [message.role for message in result.messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert result.messages[1].reasoning_content == (PRIVATE_REASONING if thinking else None)
        assert result.messages[-1].reasoning_content == ("final reasoning" if thinking else None)
        assert result.steps[0].thought == ""
        assert result.usage.total_tokens == 36

    run(scenario())
    assert workspace.reads == ["d1"]
    assert len(http.requests) == 2
    second = http.body(1)
    assert second["messages"][-1]["role"] == "tool"
    assert second["messages"][-1]["tool_call_id"] == "call_1"
    assert "Q3: ship search" in second["messages"][-1]["content"]
    if thinking:
        assert second["messages"][-2]["reasoning_content"] == PRIVATE_REASONING
    else:
        assert "reasoning_content" not in second["messages"][-2]
    assert PRIVATE_REASONING not in json.dumps(
        [event.data for event in sink.events], ensure_ascii=False
    )


@pytest.mark.parametrize(
    "first",
    [
        httpx.ReadTimeout("private network details"),
        httpx.ConnectError("private network details"),
        httpx.Response(408),
        httpx.Response(429, headers={"retry-after": "0"}),
        httpx.Response(503),
    ],
)
def test_engine_retries_transient_http_failures(first: httpx.Response | BaseException) -> None:
    http = ScriptedHTTP(first, httpx.Response(200, json=completion("recovered")))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, client=client),
                config=AgentConfig(model_retry_delay_s=0),
            )
            result = await engine.run(request())
        assert result.status is RunStatus.COMPLETED
        assert result.final_answer == "recovered"

    run(scenario())
    assert len(http.requests) == 2
    assert http.body(0) == http.body(1)


def test_engine_bounds_rate_limit_retries_and_redacts_provider_body() -> None:
    http = ScriptedHTTP(
        *[httpx.Response(429, text=f"{TEST_KEY}: private request") for _ in range(3)]
    )
    sink = CollectingSink()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, client=client),
                config=AgentConfig(max_model_retries=2, model_retry_delay_s=0),
                events=sink,
            )
            result = await engine.run(request())
        assert result.status is RunStatus.FAILED
        assert TEST_KEY not in result.error and "private request" not in result.error

    run(scenario())
    assert len(http.requests) == 3
    assert TEST_KEY not in json.dumps([event.data for event in sink.events])


def test_authentication_failure_is_not_retried() -> None:
    http = ScriptedHTTP(httpx.Response(401, json={"error": {"message": TEST_KEY}}))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(DeepSeekChatModel(api_key=TEST_KEY, client=client))
            result = await engine.run(request())
        assert result.status is RunStatus.FAILED
        assert TEST_KEY not in result.error

    run(scenario())
    assert len(http.requests) == 1


@pytest.mark.parametrize(
    ("status", "error", "expected"),
    [
        (400, {"code": "context_length_exceeded"}, ContextOverflowError),
        (422, {"code": "context_window_exceeded"}, ContextOverflowError),
        (
            400,
            {
                "message": "This model's maximum context length is 131072 tokens. "
                "However, you requested 150000 tokens."
            },
            ContextOverflowError,
        ),
        (400, {"message": "Invalid tools schema"}, ModelError),
        (400, {"message": "Invalid context configuration"}, ModelError),
        (401, {"message": "Unauthorized"}, ModelError),
        (402, {"message": "Insufficient balance"}, ModelError),
        (422, {"message": "Invalid parameters"}, ModelError),
        (429, {}, TransientModelError),
        (500, {}, TransientModelError),
    ],
)
def test_http_error_mapping_does_not_retry_inside_adapter(
    status: int, error: dict[str, str], expected: type[ModelError]
) -> None:
    http = ScriptedHTTP(httpx.Response(status, json={"error": error}))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            with pytest.raises(expected) as caught:
                await model.complete([Message("user", "hello")], [])
            assert type(caught.value) is expected

    run(scenario())
    assert len(http.requests) == 1


def test_provider_context_overflow_compacts_then_retries_through_real_engine() -> None:
    http = ScriptedHTTP(
        httpx.Response(400, json={"error": {"code": "context_length_exceeded"}}),
        httpx.Response(200, json=completion("Earlier questions summarized.")),
        httpx.Response(200, json=completion("answer after compaction")),
    )
    history = [
        Message("user" if index % 2 == 0 else "assistant", f"old {index}: " + "x" * 500)
        for index in range(12)
    ]
    sink = CollectingSink()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(DeepSeekChatModel(api_key=TEST_KEY, client=client), events=sink)
            result = await engine.run(request(history=history))
        assert result.status is RunStatus.COMPLETED
        assert result.final_answer == "answer after compaction"

    run(scenario())
    assert len(http.requests) == 3
    assert "tools" not in http.body(1)
    assert json.loads(http.body(1)["messages"][-1]["content"])["conversation"]
    assert len(http.body(2)["messages"]) < len(http.body(0)["messages"])
    assert "Earlier questions summarized." in http.body(2)["messages"][1]["content"]
    assert all("kind" not in message for message in http.body(2)["messages"])
    event = sink.of_type(EventType.CONTEXT_COMPACTED)[0]
    assert event.data["forced"] and event.data["compacted"]
    assert event.data["tokens_after"] < event.data["tokens_before"]


def test_repeated_provider_context_overflow_fails_after_one_recovery() -> None:
    http = ScriptedHTTP(
        *[
            httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
            for _ in range(2)
        ]
    )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(DeepSeekChatModel(api_key=TEST_KEY, client=client))
            result = await engine.run(request())
        assert result.status is RunStatus.FAILED

    run(scenario())
    assert len(http.requests) == 2


def test_request_deadline_is_transient_even_when_transport_does_not_timeout() -> None:
    cleaned = asyncio.Event()

    async def hanging(request: httpx.Request) -> httpx.Response:
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        raise AssertionError("unreachable")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(hanging)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, timeout_s=0.01, client=client)
            with pytest.raises(TransientModelError):
                await model.complete([Message("user", "hello")], [])
            assert cleaned.is_set()

    run(scenario())


def test_cancellation_propagates_through_http_and_engine() -> None:
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    sink = CollectingSink()

    async def hanging(request: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        raise AssertionError("unreachable")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(hanging)) as client:
            engine = AgentEngine(DeepSeekChatModel(api_key=TEST_KEY, client=client), events=sink)
            task = asyncio.create_task(engine.run(request()))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cleaned.is_set()
        assert sink.events[-1].type is EventType.RUN_CANCELED

    run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"choices": []},
        {"choices": [None]},
        {
            "choices": [
                {"message": {"role": "user", "content": "wrong role"}, "finish_reason": "stop"}
            ]
        },
        {"choices": [{"message": {"role": "assistant", "content": 7}, "finish_reason": "stop"}]},
        {
            "choices": [
                {"message": {"role": "assistant", "tool_calls": {}}, "finish_reason": "tool_calls"}
            ]
        },
        completion(
            calls=[{"id": "c1", "type": "function", "function": {"name": "read", "arguments": {}}}]
        ),
        completion(
            calls=[{"id": "c1", "type": "unknown", "function": {"name": "read", "arguments": "{}"}}]
        ),
        completion(
            calls=[{"id": "c1", "type": "function", "function": {"name": "", "arguments": "{}"}}]
        ),
        {**completion(), "usage": {"prompt_tokens": "many", "completion_tokens": 0}},
        {**completion(), "usage": {"prompt_tokens": -1, "completion_tokens": 0}},
    ],
)
def test_malformed_success_is_a_model_error(payload: Any) -> None:
    http = ScriptedHTTP(httpx.Response(200, json=payload))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            with pytest.raises(ModelError):
                await model.complete([Message("user", "hello")], [])

    run(scenario())


def test_non_json_success_is_redacted_and_missing_usage_defaults_to_zero() -> None:
    minimal = completion()
    del minimal["usage"]
    http = ScriptedHTTP(
        httpx.Response(200, text=f"<html>{TEST_KEY}</html>"),
        httpx.Response(200, json=minimal),
    )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            with pytest.raises(ModelError) as caught:
                await model.complete([Message("user", "hello")], [])
            assert TEST_KEY not in str(caught.value)
            response = await model.complete([Message("user", "hello")], [])
            assert response.usage.total_tokens == 0

    run(scenario())


def test_nonstreaming_whitespace_keepalives_are_accepted() -> None:
    http = ScriptedHTTP(httpx.Response(200, text="\n\n  \n" + json.dumps(completion())))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, client=client)
            response = await model.complete([Message("user", "hello")], [])
            assert response.content == "answer"

    run(scenario())


def test_corrupt_http_compression_fails_through_engine_without_leaking_body() -> None:
    requests: list[httpx.Request] = []

    def malformed(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"content-encoding": "gzip"}, content=TEST_KEY.encode())

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(malformed)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, client=client),
                config=AgentConfig(max_model_retries=1, model_retry_delay_s=0),
            )
            result = await engine.run(request())
        assert result.status is RunStatus.FAILED
        assert TEST_KEY not in result.error

    run(scenario())
    assert 1 <= len(requests) <= 2


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_incomplete_or_filtered_tool_calls_do_not_execute(finish: str) -> None:
    http = ScriptedHTTP(
        httpx.Response(200, json=completion(None, calls=[wire_call()], finish=finish)),
        httpx.Response(200, json=completion("safe answer")),
    )
    workspace = FakeWorkspace.default()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, client=client), workspace.registry()
            )
            result = await engine.run(request())
        assert result.status is RunStatus.COMPLETED
        assert workspace.reads == []
        if finish == "content_filter":
            assert result.stop_reason == "content_filter"
            assert len(http.requests) == 1
        else:
            assert result.steps[0].tool_calls[0].result.success is False
            assert len(http.requests) == 2

    run(scenario())


@pytest.mark.parametrize("finish", ["insufficient_system_resource", "aborted"])
def test_aborted_completion_retries_without_executing_partial_tools(finish: str) -> None:
    http = ScriptedHTTP(
        httpx.Response(200, json=completion(None, calls=[wire_call()], finish=finish)),
        httpx.Response(200, json=completion("retried safely")),
    )
    workspace = FakeWorkspace.default()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, client=client),
                workspace.registry(),
                config=AgentConfig(model_retry_delay_s=0),
            )
            result = await engine.run(request())
        assert result.final_answer == "retried safely"
        assert workspace.reads == []
        assert result.steps == []

    run(scenario())
    assert len(http.requests) == 2


def test_thinking_requires_reasoning_for_tool_history_before_network() -> None:
    http = ScriptedHTTP()
    history = [
        Message("assistant", "", (ToolCall("c1", "read", "{}"),)),
        Message("tool", "result", tool_call_id="c1"),
    ]

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(api_key=TEST_KEY, thinking=True, client=client)
            with pytest.raises(ModelError, match="reasoning_content"):
                await model.complete(history, [ToolSpec("read", "Read")])

    run(scenario())
    assert http.requests == []


@pytest.mark.parametrize("tool_name", ["read_document", "update_document"])
def test_missing_thinking_metadata_fails_before_tool_execution_or_approval(tool_name: str) -> None:
    arguments = {"document_id": "d1"}
    if tool_name == "update_document":
        arguments["content"] = "must not be written"
    http = ScriptedHTTP(
        httpx.Response(200, json=completion(None, calls=[wire_call(tool_name, **arguments)]))
    )
    workspace = FakeWorkspace.default()
    sink = CollectingSink()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            engine = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, thinking=True, client=client),
                workspace.registry(),
                events=sink,
            )
            result = await engine.run(request(policy=ToolPolicy(AccessLevel.CONFIRM)))
        assert result.status is RunStatus.FAILED
        assert "reasoning_content" in result.error
        assert result.prompt is None
        assert result.steps == []

    run(scenario())
    assert len(http.requests) == 1
    assert workspace.reads == []
    assert workspace.writes == []
    assert sink.of_type(EventType.PROMPT_REQUIRED) == []
    assert sink.of_type(EventType.TOOL_CALL) == []


def test_empty_reasoning_is_preserved_and_reasoning_effort_is_forwarded() -> None:
    http = ScriptedHTTP(httpx.Response(200, json=completion(reasoning="")))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            model = DeepSeekChatModel(
                api_key=TEST_KEY, thinking=True, reasoning_effort="high", client=client
            )
            response = await model.complete(
                [Message("assistant", "earlier", reasoning_content=""), Message("user", "next")],
                [ToolSpec("read", "Read")],
            )
            assert response.reasoning_content == ""

    run(scenario())
    assert http.body(0)["thinking"] == {"type": "enabled"}
    assert http.body(0)["reasoning_effort"] == "high"
    assert http.body(0)["messages"][0]["reasoning_content"] == ""


@pytest.mark.parametrize("revoke_permission", [False, True])
def test_thinking_survives_fresh_engine_approval_resume_without_public_leaks(
    revoke_permission: bool,
) -> None:
    http = ScriptedHTTP(
        httpx.Response(
            200,
            json=completion(
                "I will update the roadmap.",
                calls=[wire_call("update_document", document_id="d1", content="v2")],
                reasoning=PRIVATE_REASONING,
            ),
        ),
        httpx.Response(200, json=completion("finished", reasoning="private final reasoning")),
    )
    workspace = FakeWorkspace.default()
    store = InMemoryPromptStore()
    sink = CollectingSink()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            first = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, thinking=True, client=client),
                workspace.registry(),
                PromptBroker(store),
                events=sink,
            )
            async with stream_run(first, request(policy=ToolPolicy(AccessLevel.CONFIRM))) as stream:
                frames = [frame async for frame in stream]
                suspended = await stream.result()
            prompt = suspended.prompt
            assert prompt is not None and suspended.status is RunStatus.SUSPENDED
            assert workspace.writes == []
            snapshot = RunSnapshot.decode(prompt.snapshot)
            assert snapshot.pending is not None
            assert snapshot.pending.reasoning_content == PRIVATE_REASONING
            assert PRIVATE_REASONING not in json.dumps(prompt.public_view(), ensure_ascii=False)
            # Only serialized state and the shared prompt store cross this boundary.
            PromptBroker(store).resolve(EDITOR, prompt.id, {"approved": True})
            fresh = AgentEngine(
                DeepSeekChatModel(api_key=TEST_KEY, thinking=True, client=client),
                workspace.registry(),
                PromptBroker(store),
                events=sink,
            )
            caller = (
                Caller(EDITOR.workspace_id, EDITOR.user_id, Role.VIEWER)
                if revoke_permission
                else EDITOR
            )
            async with stream_resume(fresh, caller, prompt.id) as stream:
                frames.extend([frame async for frame in stream])
                result = await stream.result()
        assert result.status is RunStatus.COMPLETED
        assert result.messages[1].reasoning_content == PRIVATE_REASONING
        assert result.steps[0].thought == "I will update the roadmap."
        assert PRIVATE_REASONING.encode() not in b"".join(frames)
        assert b"private final reasoning" not in b"".join(frames)
        assert b"reasoning_content" not in b"".join(frames)
        saved = store.get(prompt.id)
        assert saved is not None
        if revoke_permission:
            assert workspace.writes == []
            assert saved.status is PromptStatus.FAILED
            assert result.steps[0].tool_calls[0].result.error.startswith("not permitted")
            assert "update_document" not in [
                tool["function"]["name"] for tool in http.body(1)["tools"]
            ]
        else:
            assert workspace.writes == [("d1", "v2")]
            assert saved.status is PromptStatus.DONE

    run(scenario())
    resumed = http.body(1)["messages"]
    assert resumed[-2]["reasoning_content"] == PRIVATE_REASONING
    assert resumed[-1]["tool_call_id"] == resumed[-2]["tool_calls"][0]["id"]
    assert PRIVATE_REASONING not in json.dumps(
        [event.data for event in sink.events], ensure_ascii=False
    )


def test_api_key_can_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", TEST_KEY)
    http = ScriptedHTTP(httpx.Response(200, json=completion()))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            async with DeepSeekChatModel(client=client) as model:
                await model.complete([Message("user", "hello")], [])
            assert not client.is_closed
            await model.aclose()
            assert not client.is_closed

    run(scenario())
    assert http.requests[0].headers["authorization"] == f"Bearer {TEST_KEY}"
    assert "tools" not in http.body(0)
    assert "max_tokens" not in http.body(0)


def test_borrowed_client_auth_and_redirect_defaults_cannot_override_adapter() -> None:
    http = ScriptedHTTP(httpx.Response(302, headers={"location": "https://other.example/receive"}))

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(http),
            auth=httpx.BasicAuth("unrelated-user", "unrelated-password"),
            follow_redirects=True,
        ) as client:
            async with DeepSeekChatModel(api_key=TEST_KEY, client=client) as model:
                with pytest.raises(ModelError):
                    await model.complete([Message("user", "hello")], [])
            assert not client.is_closed

    run(scenario())
    assert len(http.requests) == 1
    assert http.requests[0].url.host == "api.deepseek.com"
    assert http.requests[0].headers["authorization"] == f"Bearer {TEST_KEY}"


def test_owned_client_closes_on_context_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    http = ScriptedHTTP(httpx.Response(200, json=completion()))
    owned = httpx.AsyncClient(transport=httpx.MockTransport(http))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: owned)

    async def scenario() -> None:
        async with DeepSeekChatModel(api_key=TEST_KEY) as model:
            await model.complete([Message("user", "hello")], [])
            assert not owned.is_closed
        assert owned.is_closed
        await model.aclose()

    run(scenario())


@pytest.mark.parametrize(
    "options",
    [
        {"api_key": ""},
        {"api_key": "key\nheader"},
        {"timeout_s": 0},
        {"timeout_s": float("inf")},
        {"timeout_s": float("nan")},
        {"reasoning_effort": "high"},
        {"thinking": True, "reasoning_effort": "unsupported"},
        {"base_url": "https://username:password@provider.example"},
        {"base_url": "https://provider.example:invalid-port"},
    ],
)
def test_invalid_configuration_is_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DeepSeekChatModel(**{"api_key": TEST_KEY, **options})


def test_missing_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError):
        DeepSeekChatModel()
