"""Tool contract, registry and execution guard rails."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .identity import Caller, PermissionDenied, ResourceScope, Role
from .messages import ToolResult
from .policy import Risk

logger = logging.getLogger(__name__)

DEFAULT_TOOL_TIMEOUT_S = 60.0
DEFAULT_MAX_OUTPUT_CHARS = 16_000
DEFAULT_MAX_ERROR_CHARS = 4_000
DEFAULT_MAX_DATA_BYTES = 1024 * 1024

ASK_USER_TOOL_NAME = "ask_user"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    risk: Risk = Risk.READ
    # Opt-in: only side-effect-free tools may run alongside other calls.
    concurrency_safe: bool = False
    # Ask the user even when the access level would allow the call.
    always_confirm: bool = False
    # Overrides the role the risk normally requires.
    min_role: Role | None = None
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("tool timeout_s must be finite and positive")

    @property
    def required_role(self) -> Role:
        return self.min_role or self.risk.default_role

    def to_schema(self) -> dict[str, Any]:
        """Provider-neutral function schema for model adapters."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


ASK_USER_SPEC = ToolSpec(
    name=ASK_USER_TOOL_NAME,
    description=(
        "Ask the user a question when you cannot proceed without their input, such as "
        "choosing between interpretations. Do not use it for confirmation of changes; "
        "changes are confirmed automatically."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to show the user."},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional suggested answers.",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
)


@dataclass(frozen=True)
class ToolContext:
    """Server-owned facts about the call. Tools take identity from here, never from args."""

    caller: Caller
    scope: ResourceScope
    session_id: str
    run_id: str
    tool_call_id: str


class Tool(ABC):
    spec: ToolSpec

    @abstractmethod
    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult: ...


ToolFunction = Callable[[ToolContext, dict[str, Any]], Awaitable[ToolResult]]


class FunctionTool(Tool):
    def __init__(self, spec: ToolSpec, fn: ToolFunction) -> None:
        self.spec = spec
        self._fn = fn

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return await self._fn(ctx, args)


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _type_matches(expected: str | list[str], value: Any) -> bool:
    names = [expected] if isinstance(expected, str) else expected
    for name in names:
        types = _JSON_TYPES.get(name)
        if types is None:
            return True  # unknown type keyword: do not reject what we cannot judge
        # bool is an int subclass in Python but not a JSON number.
        if isinstance(value, bool) and name in ("integer", "number"):
            continue
        if isinstance(value, types):
            return True
    return False


def validate_arguments(schema: dict[str, Any], args: Any) -> list[str]:
    """Check the top level of ``args`` against a JSON-schema object subset.

    Covers required, additionalProperties: false, type and enum — enough to
    reject a malformed call before it spends a tool execution.
    """
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    errors: list[str] = []
    properties: dict[str, Any] = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            errors.append(f"missing required argument {name!r}")
    if schema.get("additionalProperties") is False:
        for name in sorted(set(args) - set(properties)):
            errors.append(f"unknown argument {name!r}")
    for name, value in args.items():
        prop = properties.get(name)
        if not isinstance(prop, dict):
            continue
        expected = prop.get("type")
        if expected is not None and not _type_matches(expected, value):
            errors.append(f"argument {name!r} must be of type {expected}")
        if "enum" in prop and value not in prop["enum"]:
            errors.append(f"argument {name!r} must be one of {prop['enum']}")
    return errors


class ToolRegistry:
    def __init__(
        self,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        *,
        max_error_chars: int = DEFAULT_MAX_ERROR_CHARS,
        max_data_bytes: int = DEFAULT_MAX_DATA_BYTES,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        for name, value in (
            ("max_output_chars", max_output_chars),
            ("max_error_chars", max_error_chars),
            ("max_data_bytes", max_data_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.max_output_chars = max_output_chars
        self.max_error_chars = max_error_chars
        self.max_data_bytes = max_data_bytes

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name == ASK_USER_TOOL_NAME:
            raise ValueError(f"{name} is reserved for the engine")
        if name in self._tools:
            raise ValueError(f"tool {name} is already registered")
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def execute(self, tool: Tool, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        """Run one call. Every failure becomes a failed result the model can read."""
        spec = tool.spec
        errors = validate_arguments(spec.parameters, args)
        if errors:
            return ToolResult.fail("invalid arguments: " + "; ".join(errors))
        try:
            async with asyncio.timeout(spec.timeout_s):
                result = await tool.execute(ctx, args)
        except PermissionDenied as exc:
            return self.govern_result(spec, ToolResult.fail(f"permission denied: {exc}"))
        except TimeoutError:
            return self._uncertain_failure(spec, f"{spec.name} timed out after {spec.timeout_s:g}s")
        except Exception as exc:
            # Exceptions can embed credentials, documents and request bodies.
            # Keep the class for diagnosis, never its text or traceback.
            logger.warning("tool execution failed (%s)", type(exc).__name__)
            return self._uncertain_failure(spec, f"tool execution failed ({type(exc).__name__})")
        if not isinstance(result, ToolResult):
            return self._uncertain_failure(spec, f"{spec.name} returned no result")
        return self.govern_result(spec, result)

    @staticmethod
    def _uncertain_failure(spec: ToolSpec, error: str) -> ToolResult:
        return ToolResult(
            success=False,
            error=error,
            data={"outcome_unknown": True} if spec.risk is not Risk.READ else {},
        )

    def govern_result(self, spec: ToolSpec, result: ToolResult) -> ToolResult:
        """Detach and bound a result before it reaches a journal or observer.

        Data is private JSON, never silently dropped: evidence and mutation
        outcomes may depend on it. Invalid/oversized data fails the call; a
        mutation then has an unknown outcome and cannot safely be replayed.
        """
        try:
            if (
                type(result.success) is not bool
                or not isinstance(result.output, str)
                or not isinstance(result.error, str)
                or not isinstance(result.data, dict)
            ):
                raise ValueError("invalid tool result fields")
            result.output.encode("utf-8")
            result.error.encode("utf-8")
            _validate_json_data(result.data, self.max_data_bytes)
            raw = json.dumps(
                result.data, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            if len(raw.encode("utf-8")) > self.max_data_bytes:
                raise ValueError("tool data exceeds limit")
            data = json.loads(raw)
            if "outcome_unknown" in data and type(data["outcome_unknown"]) is not bool:
                raise ValueError("invalid outcome flag")
            output, omitted = _excerpt(result.output, self.max_output_chars)
            if omitted:
                # A JSON result remains a complete JSON value, never a broken
                # document that could be mistaken for an executable payload.
                try:
                    parsed = json.loads(result.output)
                except (ValueError, RecursionError):
                    parsed = None
                if isinstance(parsed, (dict, list)):
                    output = json.dumps(
                        {"truncated": True, "omitted_chars": omitted, "excerpt": output},
                        ensure_ascii=False,
                    )
            error, error_omitted = _excerpt(result.error, self.max_error_chars)
            if omitted or error_omitted:
                data["result_governance"] = {
                    "output_omitted_chars": omitted,
                    "error_omitted_chars": error_omitted,
                }
            if (
                len(
                    json.dumps(
                        data, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                    ).encode("utf-8")
                )
                > self.max_data_bytes
            ):
                raise ValueError("governed tool data exceeds limit")
            return ToolResult(result.success, output, error, data)
        except (ValueError, TypeError, RecursionError, OverflowError):
            return self._uncertain_failure(spec, "tool returned invalid or oversized result data")


def _excerpt(text: str, limit: int) -> tuple[str, int]:
    if len(text) <= limit:
        return text, 0
    head = (limit + 1) // 2
    tail = limit - head
    omitted = len(text) - limit
    return text[:head] + f"\n…[truncated {omitted} chars]\n" + (
        text[-tail:] if tail else ""
    ), omitted


def _validate_json_data(data: dict[str, Any], byte_limit: int) -> None:
    # Bound nesting and node count before the JSON encoder can recurse or
    # stringify user-defined objects. This also rejects cycles deterministically.
    stack: list[tuple[Any, int]] = [(data, 0)]
    nodes = 0
    byte_count = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if depth > 32 or nodes > 10_000:
            raise ValueError("tool data exceeds structure limit")
        if isinstance(value, (dict, list)) and len(value) + len(stack) + nodes > 10_000:
            raise ValueError("tool data exceeds structure limit")
        if isinstance(value, dict):
            if any(not isinstance(key, str) or len(key) > byte_limit for key in value):
                raise ValueError("tool data keys must be strings")
            byte_count += sum(len(key.encode("utf-8")) for key in value)
            stack.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            stack.extend((child, depth + 1) for child in value)
        elif isinstance(value, str):
            if len(value) > byte_limit:
                raise ValueError("tool data string exceeds limit")
            byte_count += len(value.encode("utf-8"))
        elif value is not None and type(value) not in (str, int, float, bool):
            raise ValueError("tool data must be JSON")
        if byte_count > byte_limit:
            raise ValueError("tool data exceeds limit")
