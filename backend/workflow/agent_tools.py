"""Register existing orchestrator agents as WorkflowTools.

Called once at workflow executor startup to populate the ToolRegistry.
"""

from __future__ import annotations

from typing import Any

from backend.workflow.tool_runtime import WorkflowTool, ToolResult, get_tool_registry


def _make_agent_invoke(agent_name: str):
    """Create an invoke function for a specific orchestrator agent."""

    async def _invoke(
        query: str,
        user_context: dict | None = None,
        step: Any = None,
        previous_results: dict | None = None,
    ) -> ToolResult:
        from backend.agent.orchestrator import (
            rag_specialist_node,
            web_searcher_node,
            data_analyst_node,
            local_graph_search_node,
            global_graph_search_node,
            direct_answer_node,
        )

        _node_map = {
            "rag_specialist": rag_specialist_node,
            "web_searcher": web_searcher_node,
            "data_analyst": data_analyst_node,
            "local_graph_search": local_graph_search_node,
            "global_graph_search": global_graph_search_node,
            "direct_answer": direct_answer_node,
        }

        node_fn = _node_map.get(agent_name)
        if node_fn is None:
            return ToolResult(success=False, error=f"Unknown agent: {agent_name}")

        state = {
            "messages": [],
            "user_query": query,
            "user_context": user_context or {},
            "worker_outputs": {},
            "rag_trace": None,
            "web_search_trace": None,
            "agent_trace": None,
            "tool_outputs": {},
            "query_intent": None,
        }

        try:
            result_state = await node_fn(state)
            response = result_state.get("worker_outputs", {}).get(agent_name, "")
            if not response:
                msgs = result_state.get("messages", [])
                if msgs:
                    response = str(msgs[-1].content) if hasattr(msgs[-1], "content") else str(msgs[-1])
            return ToolResult(
                success=True,
                data={
                    "response": str(response),
                    "rag_trace": result_state.get("rag_trace"),
                    "web_search_trace": result_state.get("web_search_trace"),
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    return _invoke


def register_agent_tools():
    """Register all 6 orchestrator agents as WorkflowTools."""
    registry = get_tool_registry()

    agents = [
        "rag_specialist",
        "web_searcher",
        "data_analyst",
        "local_graph_search",
        "global_graph_search",
        "direct_answer",
    ]

    for name in agents:
        tool = WorkflowTool.from_agent(name, _make_agent_invoke(name))
        registry.register(tool)
