"""Exercise request -> read tool -> answer without contacting a model by default.

uv run --extra deepseek python examples/deepseek_demo.py [--thinking] [--stream]
Add --live to call DeepSeek using DEEPSEEK_API_KEY (billable API requests).
--stream prints the SSE response body, including incremental text events.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from typing import Any

import httpx

from magent_harness import (
    AgentConfig,
    AgentEngine,
    Caller,
    DeepSeekChatModel,
    FunctionTool,
    Role,
    RunRequest,
    RunStatus,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    stream_run,
)

_DOCUMENT = {"id": "roadmap", "workspace_id": "demo", "content": "Q3 上线搜索。Q4 上线 Agent。"}


async def read_document(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if args["document_id"] != _DOCUMENT["id"]:
        return ToolResult.fail("document not found")
    ctx.scope.require_document(workspace_id=_DOCUMENT["workspace_id"], document_id=_DOCUMENT["id"])
    return ToolResult.ok(_DOCUMENT["content"])


class DemoStream(httpx.AsyncByteStream):
    def __init__(self, message: dict[str, Any], *, has_result: bool) -> None:
        self.message = message
        self.has_result = has_result

    async def __aiter__(self) -> AsyncIterator[bytes]:
        deltas: list[dict[str, Any]] = [{"role": "assistant"}]
        if "reasoning_content" in self.message:
            deltas.append({"reasoning_content": self.message["reasoning_content"]})
        if self.has_result:
            deltas.extend({"content": char} for char in self.message["content"])
        else:
            call = self.message["tool_calls"][0]
            args = call["function"]["arguments"]
            deltas.append(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": "read_document", "arguments": ""},
                        }
                    ]
                }
            )
            deltas.extend(
                {"tool_calls": [{"index": 0, "function": {"arguments": part}}]}
                for part in (args[:10], args[10:])
            )
        for delta in deltas:
            await asyncio.sleep(0)
            yield self.frame({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
        yield self.frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop" if self.has_result else "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            }
        )
        yield b"data: [DONE]\n\n"

    @staticmethod
    def frame(chunk: dict[str, Any]) -> bytes:
        return ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode()


def mock_completion(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    has_result = body["messages"][-1]["role"] == "tool"
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "roadmap: Q3 上线搜索。Q4 上线 Agent。" if has_result else None,
    }
    if body["thinking"]["type"] == "enabled":
        message["reasoning_content"] = "offline provider metadata"
    if not has_result:
        message["tool_calls"] = [
            {
                "id": "read_1",
                "type": "function",
                "function": {
                    "name": "read_document",
                    "arguments": '{"document_id":"roadmap"}',
                },
            }
        ]
    if body["stream"]:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=DemoStream(message, has_result=has_result),
        )
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": message, "finish_reason": "stop" if has_result else "tool_calls"}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        },
    )


async def main(*, live: bool, thinking: bool, streaming: bool) -> None:
    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            ToolSpec(
                "read_document",
                "Read the roadmap document before answering.",
                {
                    "type": "object",
                    "properties": {"document_id": {"type": "string"}},
                    "required": ["document_id"],
                    "additionalProperties": False,
                },
                concurrency_safe=True,
            ),
            read_document,
        )
    )
    transport = None if live else httpx.MockTransport(mock_completion)
    async with (
        httpx.AsyncClient(transport=transport) as client,
        DeepSeekChatModel(
            api_key=None if live else "offline-demo-key",
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            thinking=thinking,
            client=client,
        ) as model,
    ):
        engine = AgentEngine(model, registry, config=AgentConfig(stream_model=streaming))
        request = RunRequest(
            Caller("demo", "demo-user", Role.VIEWER),
            "demo-session",
            "请先调用 read_document 读取 roadmap 文档。然后说明 Q3 和 Q4 的计划。",
        )
        if streaming:
            async with stream_run(engine, request) as stream:
                async for frame in stream:
                    sys.stdout.buffer.write(frame)
                    sys.stdout.buffer.flush()
                result = await stream.result()
        else:
            result = await engine.run(request)
    if result.status is not RunStatus.COMPLETED:
        raise RuntimeError(result.error or f"run ended as {result.status}")
    if not any(record.result.success for step in result.steps for record in step.tool_calls):
        raise RuntimeError("the model did not complete the requested tool round trip")
    if result.stop_reason != "answer":
        raise RuntimeError("the model did not produce a normal final answer")
    if not streaming:
        print(result.final_answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call the real, billable DeepSeek API")
    parser.add_argument("--thinking", action="store_true", help="Enable reasoning metadata")
    parser.add_argument("--stream", action="store_true", help="Print SSE with model text deltas")
    args = parser.parse_args()
    asyncio.run(main(live=args.live, thinking=args.thinking, streaming=args.stream))
