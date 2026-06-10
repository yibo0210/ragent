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
        ctx = "\n\n前置步骤结果:\n"
        for dep_id, result in previous_results.items():
            if hasattr(result, 'data') and result.data:
                response = result.data.get("response", str(result.data))
                ctx += f"[{dep_id}]: {response[:2000]}\n"
    return ctx


def _make_agent_invoke(agent_name: str):
    """Create a lightweight invoke function using direct LLM calls."""

    _tool_prompts = {
        "rag_specialist": "你是一个知识检索专家。搜索和分析文档来准确回答问题。请用中文回答。",
        "web_searcher": "你是一个网络搜索专家。提供实时信息和外部数据分析。请用中文回答。",
        "data_analyst": "你是一个数据分析师。生成SQL查询并分析结构化数据来回答业务问题。请用中文回答。",
        "local_graph_search": "你是一个知识图谱专家。探索实体关系和连接。请用中文回答。",
        "global_graph_search": "你是一个知识策展人。提供高层次主题分析和社区洞察。请用中文回答。",
        "direct_answer": "你是一个有用的AI助手。直接简洁地回答问题。请用中文回答。",
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

            ctx = _build_contextual_prompt(agent_name, query, previous_results)

            # rag_specialist: actually search the knowledge base
            if agent_name == "rag_specialist":
                try:
                    import asyncio
                    from backend.rag.utils import retrieve_documents
                    tenant_id = (user_context or {}).get("tenant_id", 0)
                    result = await asyncio.to_thread(
                        retrieve_documents, query=query, tenant_id=tenant_id, top_k=5,
                    )
                    docs = result.get("documents", []) or result.get("results", [])
                    if docs:
                        kb_context = "\n\n".join(
                            f"[{d.metadata.get('filename', 'doc') if hasattr(d, 'metadata') else 'doc'}]: {d.page_content[:800] if hasattr(d, 'page_content') else str(d)[:800]}"
                            for d in docs[:5]
                        )
                        ctx = f"\n\n知识库检索结果:\n{kb_context}{ctx}"
                except Exception:
                    pass

            # web_searcher: actually search the web
            if agent_name == "web_searcher":
                try:
                    import asyncio
                    from backend.agent.web_searcher import run_web_search
                    web_result = await asyncio.to_thread(run_web_search, query, 3)
                    results = web_result.get("results", []) if isinstance(web_result, dict) else []
                    if results:
                        web_ctx = "\n\n".join(
                            f"[{r.get('title', 'web')}]: {r.get('content', '')[:500]}"
                            for r in results[:3]
                        )
                        ctx = f"\n\n网络搜索结果:\n{web_ctx}{ctx}"
                except Exception:
                    pass

            model = init_chat_model(
                model="qwen-turbo", model_provider="openai",
                api_key=settings.ark_api_key, base_url=settings.base_url,
                temperature=0.0, max_tokens=1024, timeout=60,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"任务: {query}{ctx}\n\n请提供详细的中文回答。"),
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
