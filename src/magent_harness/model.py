"""The chat-model boundary. Provider adapters implement :class:`ChatModel`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .messages import Message, ToolCall, Usage

if TYPE_CHECKING:
    from .tools import ToolSpec

FINISH_STOP = "stop"
FINISH_TOOL_CALLS = "tool_calls"
# The provider cut the completion at max tokens: content and tool-call
# arguments may be incomplete.
FINISH_LENGTH = "length"
FINISH_CONTENT_FILTER = "content_filter"


@dataclass
class ChatResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = FINISH_STOP
    usage: Usage = field(default_factory=Usage)
    # Provider history metadata; never display as the assistant's answer/thought.
    reasoning_content: str | None = None


class ModelError(Exception):
    """A model call failed. The engine ends the run as failed instead of crashing."""


class TransientModelError(ModelError):
    """A model call failed in a way worth retrying (rate limit, timeout, 5xx)."""


class ContextOverflowError(ModelError):
    """The provider rejected the request because it exceeds the context window."""


class ChatModel(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Return one assistant response for ``messages``.

        ``tools`` is empty when the engine wants plain text (summaries, the
        final synthesis after the round budget is spent).
        """
        ...


@runtime_checkable
class StreamingChatModel(ChatModel, Protocol):
    async def complete_stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        on_delta: Callable[[str], None],
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Deliver provisional public text, then return one validated response.

        Call ``on_delta`` only with content text, never reasoning or tool
        arguments. Yield to the event loop between deltas so consumers and
        cancellation can progress. Return tool calls only after the complete
        provider stream has been validated; interrupted streams must raise.
        Internal summaries still use ``complete`` without public callbacks.
        """
        ...
