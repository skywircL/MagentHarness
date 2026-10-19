"""DeepSeek Chat Completions adapter (optional HTTPX dependency)."""

from __future__ import annotations

import asyncio
import math
import os
import re
from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from ._deepseek_stream import MAX_EVENT_BYTES, CompletionStream, SSEDecoder
from .messages import Message, ToolCall, Usage
from .model import (
    FINISH_CONTENT_FILTER,
    FINISH_LENGTH,
    FINISH_STOP,
    FINISH_TOOL_CALLS,
    ChatResponse,
    ContextOverflowError,
    ModelError,
    TransientModelError,
)
from .tools import ToolSpec

if TYPE_CHECKING:
    import httpx


class DeepSeekChatModel:
    """An asyncio model adapter; use as an async context manager or call aclose.

    Thinking is explicitly disabled unless requested. An injected AsyncClient
    remains owned by its caller. No SDK retry loop is added: AgentEngine owns
    retries and context overflow recovery. The core package can be imported
    without HTTPX; install ``magent-harness[deepseek]`` to create this adapter.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        thinking: bool = False,
        reasoning_effort: Literal["low", "high", "max"] | None = None,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        try:
            import httpx
        except ImportError:
            raise ImportError("Install magent-harness[deepseek] to use DeepSeekChatModel") from None

        key = os.environ.get("DEEPSEEK_API_KEY", "") if api_key is None else api_key
        if not key or any(ord(ch) <= 32 or ord(ch) > 126 for ch in key):
            raise ValueError("a non-empty ASCII DeepSeek API key is required")
        if not model.strip() or any(ord(ch) < 32 for ch in model):
            raise ValueError("model must be a non-empty model identifier")
        if type(thinking) is not bool:
            raise ValueError("thinking must be a boolean")
        if reasoning_effort is not None and (
            not thinking or reasoning_effort not in ("low", "high", "max")
        ):
            raise ValueError("reasoning_effort requires thinking=True and low, high or max")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        try:
            url = urlsplit(base_url)
            # Parse the port too: urlsplit defers validation until it is read.
            _ = url.port
            httpx.URL(base_url)
        except (ValueError, httpx.InvalidURL):
            raise ValueError("base_url is not a valid HTTP(S) URL") from None
        if (
            url.scheme not in ("https", "http")
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or any(ch.isspace() for ch in base_url)
        ):
            raise ValueError(
                "base_url must be an HTTP(S) URL without credentials, query or fragment"
            )

        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._api_key = key
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient()
        self._closed = False

    async def __aenter__(self) -> DeepSeekChatModel:
        if self._closed:
            raise ModelError("DeepSeek adapter is closed")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
        self._closed = True

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        import httpx

        payload = self._payload(messages, tools, max_tokens=max_tokens, stream=False)
        try:
            # DeepSeek may send whitespace keepalives even with stream=False.
            # An inactivity timeout alone would not bound the whole request.
            async with asyncio.timeout(self.timeout_s):
                response = await self._client.request(
                    "POST",
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Accept": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_s,
                    follow_redirects=False,
                    auth=None,
                )
        except (TimeoutError, httpx.TimeoutException):
            raise TransientModelError("DeepSeek request timed out") from None
        except httpx.DecodingError:
            raise ModelError("DeepSeek response could not be decoded") from None
        except httpx.TransportError:
            raise TransientModelError("DeepSeek transport failed") from None
        except (TypeError, ValueError):
            raise ModelError("DeepSeek request could not be encoded") from None

        if response.status_code != 200:
            _raise_http_error(response)
        try:
            body = response.json()
        except ValueError:
            raise ModelError("DeepSeek returned invalid JSON") from None
        return self._result(body, tools)

    async def complete_stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        on_delta: Callable[[str], None],
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Emit public text deltas, then return a fully validated completion.

        Tool fragments and provider reasoning remain private until the complete
        response has passed validation. A disconnected or malformed stream never
        returns executable tool calls. The caller owns any displayed draft text.
        """
        import httpx

        payload = self._payload(messages, tools, max_tokens=max_tokens, stream=True)
        try:
            request = self._client.build_request(
                "POST",
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "text/event-stream",
                },
                json=payload,
                timeout=self.timeout_s,
            )
        except (TypeError, ValueError):
            raise ModelError("DeepSeek request could not be encoded") from None

        decoder = SSEDecoder()
        completion = CompletionStream()
        try:
            async with asyncio.timeout(self.timeout_s):
                response = await self._client.send(
                    request, stream=True, follow_redirects=False, auth=None
                )
                try:
                    if response.status_code != 200:
                        await _raise_stream_http_error(response)
                    async for data in response.aiter_bytes():
                        for event, body in decoder.feed(data):
                            delta = completion.accept(event, body)
                            if delta:
                                on_delta(delta)
                            # One HTTP read can contain hundreds of SSE frames.
                            # Let consumers drain their buffers and cancellation
                            # take effect before returning any executable calls.
                            await asyncio.sleep(0)
                            if completion.done:
                                return self._result(completion.body(), tools)
                        await asyncio.sleep(0)
                    raise TransientModelError("DeepSeek stream ended before completion")
                finally:
                    await response.aclose()
        except (TimeoutError, httpx.TimeoutException):
            raise TransientModelError("DeepSeek request timed out") from None
        except httpx.DecodingError:
            raise ModelError("DeepSeek response could not be decoded") from None
        except httpx.TransportError:
            raise TransientModelError("DeepSeek transport failed") from None

    def _payload(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        if self._closed or self._client.is_closed:
            raise ModelError("DeepSeek adapter or HTTP client is closed")
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
            raise ModelError("max_tokens must be a positive integer")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(messages, require_reasoning=self.thinking and bool(tools)),
            "stream": stream,
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [
                {"type": "function", "function": tool.to_schema()} for tool in tools
            ]

        return payload

    def _result(self, body: Any, tools: list[ToolSpec]) -> ChatResponse:
        result = _parse_response(body)
        if self.thinking and tools and result.reasoning_content is None:
            raise ModelError("DeepSeek thinking response is missing reasoning_content")
        return result

    def _messages(
        self, messages: list[Message], *, require_reasoning: bool
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for message in messages:
            item: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.role == "assistant":
                if require_reasoning and message.reasoning_content is None:
                    raise ModelError(
                        "DeepSeek thinking with tools requires reasoning_content for every "
                        "assistant history message; start a new history or disable thinking"
                    )
                if self.thinking and message.reasoning_content is not None:
                    item["reasoning_content"] = message.reasoning_content
                if message.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in message.tool_calls
                    ]
            if message.role == "tool":
                item["tool_call_id"] = message.tool_call_id
            elif message.name:
                item["name"] = message.name
            output.append(item)
        return output


async def _raise_stream_http_error(response: httpx.Response) -> None:
    import httpx

    if response.status_code not in (400, 422):
        _raise_http_error(response)
    # Only these statuses require the provider body for context classification.
    # Bound error bodies too; a stream must never bypass the memory limits.
    content = bytearray()
    async for chunk in response.aiter_bytes():
        if len(content) + len(chunk) > MAX_EVENT_BYTES:
            raise ModelError("DeepSeek error response exceeded the size limit")
        content.extend(chunk)
    _raise_http_error(httpx.Response(response.status_code, content=bytes(content)))


def _raise_http_error(response: httpx.Response) -> None:
    status = response.status_code
    if status in (408, 429) or 500 <= status <= 599:
        raise TransientModelError(f"DeepSeek temporarily unavailable (HTTP {status})")
    if status in (400, 422):
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            error = body["error"]
            code = error.get("code")
            message = error.get("message")
            if code in ("context_length_exceeded", "context_window_exceeded") or (
                isinstance(message, str) and _context_overflow_message(message.lower())
            ):
                raise ContextOverflowError("DeepSeek context window exceeded")
    # Provider bodies may echo document text or authentication data. Neither
    # logs nor the public RUN_FAILED event should receive those raw bodies.
    raise ModelError(f"DeepSeek request rejected (HTTP {status})")


def _context_overflow_message(message: str) -> bool:
    return bool(
        re.search(r"context (?:length|window)(?: limit)? (?:is |was |has been )?exceeded", message)
        or ("maximum context length" in message and ("requested" in message or "exceed" in message))
    )


def _parse_response(body: Any) -> ChatResponse:
    invalid = "DeepSeek returned an invalid chat completion"
    if not isinstance(body, dict):
        raise ModelError(invalid)
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ModelError(invalid)
    choice = choices[0]
    finish = choice.get("finish_reason")
    if finish in ("insufficient_system_resource", "aborted"):
        raise TransientModelError("DeepSeek generation was interrupted")
    if finish not in (FINISH_STOP, FINISH_TOOL_CALLS, FINISH_LENGTH, FINISH_CONTENT_FILTER):
        raise ModelError(invalid)
    message = choice.get("message")
    if (
        not isinstance(message, dict)
        or message.get("role") != "assistant"
        or "content" not in message
    ):
        raise ModelError(invalid)
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if content is not None and not isinstance(content, str):
        raise ModelError(invalid)
    if reasoning is not None and not isinstance(reasoning, str):
        raise ModelError(invalid)
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        raise ModelError(invalid)
    calls: list[ToolCall] = []
    for call in raw_calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ModelError(invalid)
        function = call.get("function")
        if (
            not isinstance(call.get("id"), str)
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"]
            or not isinstance(function.get("arguments"), str)
        ):
            raise ModelError(invalid)
        calls.append(ToolCall(call["id"], function["name"], function["arguments"]))
    if finish == FINISH_TOOL_CALLS and not calls:
        raise ModelError(invalid)
    if finish == FINISH_CONTENT_FILTER:
        # Never execute calls from a filtered completion.
        calls = []
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        raise ModelError(invalid)
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    if any(type(value) is not int or value < 0 for value in (prompt_tokens, completion_tokens)):
        raise ModelError(invalid)
    return ChatResponse(
        content=content or "",
        tool_calls=calls,
        finish_reason=finish,
        usage=Usage(prompt_tokens, completion_tokens),
        reasoning_content=reasoning,
    )
