"""Register existing orchestrator agents as WorkflowTools.

Called once at workflow executor startup to populate the ToolRegistry.
Uses lightweight LLM calls instead of full orchestrator node functions
for fast, reliable workflow execution.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from backend.workflow.tool_runtime import WorkflowTool, ToolResult, get_tool_registry


def _build_contextual_prompt(agent_name: str, query: str, previous_results: dict | None = None) -> str:
    """Build a prompt with context from previous step results."""
    ctx = ""
    if previous_results:
        ctx = "\n\nPrevious step results:\n"
        for dep_id, result in previous_results.items():
            if hasattr(result, 'data') and result.data:
                response = result.data.get("response", str(result.data))
                ctx += f"[{dep_id}]: {response[:2000]}\n"
    return ctx


def _make_agent_invoke(agent_name: str):
    """Create a lightweight invoke function using direct LLM calls."""

    _tool_prompts = {
        "rag_specialist": "You are a knowledge retrieval specialist. Search and analyze documents to answer the query accurately.",
        "web_searcher": "You are a web search specialist. Provide real-time information and external data analysis.",
        "data_analyst": "You are a data analyst. Generate SQL queries and analyze structured data to answer business questions.",
        "local_graph_search": "You are a graph knowledge specialist. Explore entity relationships and connections.",
        "global_graph_search": "You are a knowledge curator. Provide high-level topic analysis and community insights.",
        "direct_answer": "You are a helpful AI assistant. Answer questions directly and concisely.",
    }

    system_prompt = _tool_prompts.get(agent_name, "You are a helpful AI assistant.")

    async def _invoke(
        query: str,
        user_context: dict | None = None,
        step: Any = None,
        previous_results: dict | None = None,
    ) -> ToolResult:
        try:
            from langchain.chat_models import init_chat_model
            from backend.config import get_settings
            settings = get_settings()
            model = init_chat_model(
                model="qwen-turbo", model_provider="openai",
                api_key=settings.ark_api_key, base_url=settings.base_url,
                temperature=0.0, max_tokens=1024, timeout=60,
            )
            ctx = _build_contextual_prompt(agent_name, query, previous_results)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Task: {query}{ctx}\n\nProvide a thorough response."),
            ]

            response = await model.ainvoke(messages)
            content = response.content if hasattr(response, "content") else str(response)

            return ToolResult(
                success=True,
                data={"response": content, "agent": agent_name},
                tokens_used=len(content) // 4,
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
