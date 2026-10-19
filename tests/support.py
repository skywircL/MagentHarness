"""Shared test support: a fake workspace backend exposed as tools, and engine builders."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar

from magent_harness import (
    AgentConfig,
    AgentEngine,
    Caller,
    CollectingSink,
    FunctionTool,
    MemoryService,
    PromptBroker,
    Risk,
    Role,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from magent_harness.testing import ScriptedModel

T = TypeVar("T")


def run(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


EDITOR = Caller("ws1", "alice", Role.EDITOR)
VIEWER = Caller("ws1", "bob", Role.VIEWER)
OTHER_WORKSPACE = Caller("ws2", "alice", Role.OWNER)


@dataclass
class Doc:
    id: str
    workspace_id: str
    title: str
    versions: list[str]


def _doc_schema(**extra: Any) -> dict[str, Any]:
    properties: dict[str, Any] = {"document_id": {"type": "string"}}
    properties.update(extra)
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


@dataclass
class FakeWorkspace:
    """Stands in for the backend. Tools resolve documents from storage and check
    the run scope against the stored workspace, as real tools must."""

    docs: dict[str, Doc] = field(default_factory=dict)
    writes: list[tuple[str, str]] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)
    reads: list[str] = field(default_factory=list)

    @classmethod
    def default(cls) -> FakeWorkspace:
        ws = cls()
        ws.docs = {
            "d1": Doc("d1", "ws1", "Roadmap", ["Q3: ship search. Q4: ship agent."]),
            "d2": Doc("d2", "ws1", "Notes", ["Meeting notes about search quality."]),
            "d9": Doc("d9", "ws2", "Secret", ["Another workspace's plan."]),
        }
        return ws

    def _load(self, ctx: ToolContext, document_id: str) -> Doc:
        doc = self.docs.get(document_id)
        if doc is None:
            raise KeyError(f"no document {document_id}")
        ctx.scope.require_document(doc.workspace_id, doc.id)
        return doc

    def tools(self) -> list[FunctionTool]:
        async def search(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            hits = [
                f"{d.id}: {d.title}"
                for d in self.docs.values()
                if ctx.scope.allows_document(d.workspace_id, d.id)
                and args["query"].lower() in (d.title + d.versions[-1]).lower()
            ]
            return ToolResult.ok("\n".join(hits) or "no hits")

        async def read(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            doc = self._load(ctx, args["document_id"])
            self.reads.append(doc.id)
            return ToolResult.ok(f"[{doc.id} v{len(doc.versions)}] {doc.versions[-1]}")

        async def update(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            doc = self._load(ctx, args["document_id"])
            doc.versions.append(args["content"])
            self.writes.append((doc.id, args["content"]))
            return ToolResult.ok(f"{doc.id} is now at version {len(doc.versions)}")

        async def delete(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            doc = self._load(ctx, args["document_id"])
            self.deletes.append(doc.id)
            return ToolResult.ok(f"deleted {doc.id}")

        return [
            FunctionTool(
                ToolSpec(
                    "search_documents",
                    "Search documents.",
                    {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    concurrency_safe=True,
                ),
                search,
            ),
            FunctionTool(
                ToolSpec("read_document", "Read a document.", _doc_schema(), concurrency_safe=True),
                read,
            ),
            FunctionTool(
                ToolSpec(
                    "update_document",
                    "Write a new version of a document.",
                    _doc_schema(content={"type": "string"}),
                    risk=Risk.WRITE,
                ),
                update,
            ),
            FunctionTool(
                ToolSpec(
                    "delete_document",
                    "Delete a document.",
                    _doc_schema(),
                    risk=Risk.DESTRUCTIVE,
                ),
                delete,
            ),
        ]

    def registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        for tool in self.tools():
            registry.register(tool)
        return registry


@dataclass
class Harness:
    engine: AgentEngine
    model: ScriptedModel
    workspace: FakeWorkspace
    broker: PromptBroker
    sink: CollectingSink


def build(
    script: list[Any] | None = None,
    *,
    workspace: FakeWorkspace | None = None,
    registry: ToolRegistry | None = None,
    broker: PromptBroker | None = None,
    config: AgentConfig | None = None,
    memory: MemoryService | None = None,
    summarizer: Any = None,
) -> Harness:
    workspace = workspace or FakeWorkspace.default()
    model = ScriptedModel(script or [])
    broker = broker or PromptBroker()
    sink = CollectingSink()
    engine = AgentEngine(
        model,
        registry or workspace.registry(),
        broker,
        config=config or AgentConfig(model_retry_delay_s=0),
        memory=memory,
        events=sink,
        summarizer=summarizer,
    )
    return Harness(engine, model, workspace, broker, sink)
