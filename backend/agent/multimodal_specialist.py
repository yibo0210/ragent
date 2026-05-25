"""多模态专家 Agent：视觉检索 + 回答生成。"""
from langchain_core.messages import HumanMessage, AIMessage

MULTIMODAL_KEYWORDS = ["图表", "曲线", "图像", "图片", "趋势图", "柱状图", "饼图", "示意图"]


def is_multimodal_query(query: str) -> bool:
    return any(kw in query for kw in MULTIMODAL_KEYWORDS)


def multimodal_specialist_node(state: dict) -> dict:
    """多模态专家节点：检索视觉内容 + 生成带引用的回答。"""
    from backend.rag.visual_retriever import retrieve_visual
    from backend.agent.orchestrator import _get_worker_model

    user_query = state.get("user_query", "")
    model = _get_worker_model()

    visual_results = retrieve_visual(user_query, top_k=5)

    context_parts = []
    for v in visual_results:
        entity = v.get("entity", v)
        url = entity.get("associated_media_urls", "")
        text = entity.get("text", "")[:400]
        context_parts.append(f"[{text}]" + (f"\n(图片: {url})" if url else ""))

    context = "\n\n".join(context_parts) if context_parts else "未找到相关图表，请基于文本知识回答。"

    prompt = (
        f"根据以下图表信息回答问题。如果上下文中有图片URL，请在回答中引用。\n\n"
        f"{context}\n\n## 用户问题\n\n{user_query}"
    )
    answer = model.invoke([HumanMessage(content=prompt)])
    content = answer.content if hasattr(answer, "content") else str(answer)

    agent_trace = state.get("agent_trace") or {}
    agent_trace["multimodal_sources"] = len(visual_results)

    return {
        "messages": [AIMessage(content=content)],
        "agent_trace": agent_trace,
        "worker_outputs": {"multimodal_specialist": {
            "answer": content, "visual_sources": len(visual_results),
        }},
    }
