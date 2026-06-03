"""WorkflowTool: unified abstraction over agents and MCP tools.

Avoids direct agent->tool coupling by providing a single call interface
that the WorkflowExecutor uses regardless of backend (agent, MCP, or custom).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional


class ToolResult:
    """Standardized result from any tool invocation."""

    __slots__ = ("success", "data", "error", "tokens_used")

    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: str = "",
        tokens_used: int = 0,
    ):
        self.success = success
        self.data = data
        self.error = error
        self.tokens_used = tokens_used

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "tokens_used": self.tokens_used,
        }


class WorkflowTool:
    """Unified tool interface wrapping an agent or MCP tool.

    Usage:
        tool = WorkflowTool.from_agent("rag_specialist", rag_func)
        tool = WorkflowTool.from_mcp("mysql_query", server_name="mysql", tool_name="query")
        result = await tool.invoke("query text", user_context={...})
    """

    def __init__(
        self,
        name: str,
        description: str,
        invoke_fn: Callable,
        tool_type: str = "agent",
        server_name: str = "",
        tool_name: str = "",
    ):
        self.name = name
        self.description = description
        self._invoke = invoke_fn
        self.tool_type = tool_type
        self.server_name = server_name
        self.tool_name = tool_name

    async def invoke(
        self,
        query: str,
        user_context: dict | None = None,
        step: Any = None,
        previous_results: dict[str, ToolResult] | None = None,
    ) -> ToolResult:
        """Invoke the tool with query and context."""
        try:
            result = self._invoke(
                query=query,
                user_context=user_context or {},
                step=step,
                previous_results=previous_results or {},
            )
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, dict):
                return ToolResult(success=True, data=result)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(success=True, data={"response": str(result)})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    @classmethod
    def from_agent(cls, name: str, invoke_fn: Callable) -> "WorkflowTool":
        """Wrap an existing agent node function."""
        _descriptions = {
            "rag_specialist": "Knowledge base retrieval and question answering",
            "web_searcher": "Web search for real-time information",
            "data_analyst": "SQL query generation and execution against MySQL",
            "local_graph_search": "Graph entity search with 1-hop neighbor expansion",
            "global_graph_search": "Community-level graph summary search",
            "direct_answer": "Direct LLM response for general knowledge",
        }
        return cls(
            name=name,
            description=_descriptions.get(name, name),
            invoke_fn=invoke_fn,
            tool_type="agent",
        )

    @classmethod
    def from_mcp(cls, name: str, server_name: str, tool_name: str) -> "WorkflowTool":
        """Wrap an MCP tool."""

        async def _mcp_invoke(query: str, user_context=None, **kwargs):
            from backend.agent.mcp_client import get_mcp_manager

            manager = get_mcp_manager()
            return await manager.call_tool(
                server_name=server_name,
                tool_name=tool_name,
                arguments={"query": query},
                tenant_id=user_context.get("tenant_id", 0) if user_context else 0,
                user_id=user_context.get("user_id", 0) if user_context else 0,
            )

        return cls(
            name=name,
            description=f"MCP tool: {server_name}/{tool_name}",
            invoke_fn=_mcp_invoke,
            tool_type="mcp",
            server_name=server_name,
            tool_name=tool_name,
        )


class ToolRegistry:
    """Registry of all available WorkflowTools, populated at executor startup."""

    def __init__(self):
        self._tools: dict[str, WorkflowTool] = {}

    def register(self, tool: WorkflowTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[WorkflowTool]:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "type": t.tool_type}
            for t in self._tools.values()
        ]


_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry
