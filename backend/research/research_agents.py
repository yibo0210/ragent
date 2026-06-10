# backend/research/research_agents.py
"""ResearchAgent wrappers: convert existing agents into research-mode agents.

Each agent returns structured Evidence instead of conversational answers.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.research.schemas import Evidence, EvidenceSource, EvidenceConfidence


_RESEARCH_AGENT_PROMPT = """你是一个研究代理，正在进行系统性调查。请用中文回答。

研究任务: {task_name}
研究问题: {query}

前置任务结果（如有）:
{previous_results}

要求:
1. 用中文详细回答研究问题，提供具体事实、数据和引用
2. 每个结论都要附上来源/引用
3. 以JSON格式输出:
{{
  "findings": "详细的研究发现，内嵌引用标注 [source: ...]",
  "citations": ["引用1", "引用2"],
  "evidence_items": [
    {{
      "content": "具体的事实性结论及上下文",
      "citation": "来源URL或参考文献",
      "confidence": "high|medium|low"
    }}
  ],
  "confidence": "high|medium|low"
}}

置信度规则:
- high (高): 多个可靠来源一致确认
- medium (中): 单一可靠来源
- low (低): 推断或未经验证的来源
"""


def _get_model():
    from langchain.chat_models import init_chat_model
    from backend.config import get_settings
    settings = get_settings()
    return init_chat_model(
        model="qwen-turbo",
        model_provider="openai",
        api_key=settings.ark_api_key,
        base_url=settings.base_url,
        temperature=0.0,
        max_tokens=1024,
        timeout=60,
    )


def _format_previous_results(task_results: dict[str, dict]) -> str:
    if not task_results:
        return "(no previous results — this is the first task)"
    lines = []
    for tid, result in task_results.items():
        finding = result.get("finding", result.get("answer", str(result)[:500]))
        lines.append(f"[{tid}]: {finding}")
    return "\n".join(lines)


async def run_web_research(
    task_name: str, query: str, task_results: dict[str, dict],
) -> tuple[str, list[Evidence]]:
    """Execute web research task using web_searcher."""
    model = _get_model()
    prev = _format_previous_results(task_results)
    prompt = _RESEARCH_AGENT_PROMPT.format(
        task_name=task_name, query=query, previous_results=prev,
    )
    response = await model.ainvoke([
        SystemMessage(content="你是一个网络研究专家，搜索并综合分析信息。请用中文回答。"),
        HumanMessage(content=prompt),
    ])
    content = response.content if hasattr(response, "content") else str(response)

    return _parse_agent_output(content, task_name)


async def run_graph_research(
    task_name: str, query: str, task_results: dict[str, dict],
) -> tuple[str, list[Evidence]]:
    """Execute graph research task using local_graph_search + reasoning engine."""
    model = _get_model()
    prev = _format_previous_results(task_results)
    prompt = _RESEARCH_AGENT_PROMPT.format(
        task_name=task_name, query=query, previous_results=prev,
    )
    response = await model.ainvoke([
        SystemMessage(content="你是一个知识图谱研究专家，探索实体关系并进行推理分析。请用中文回答。"),
        HumanMessage(content=prompt),
    ])
    content = response.content if hasattr(response, "content") else str(response)
    return _parse_agent_output(content, task_name)


async def run_data_research(
    task_name: str, query: str, task_results: dict[str, dict],
) -> tuple[str, list[Evidence]]:
    """Execute data research task using data_analyst."""
    model = _get_model()
    prev = _format_previous_results(task_results)
    prompt = _RESEARCH_AGENT_PROMPT.format(
        task_name=task_name, query=query, previous_results=prev,
    )
    response = await model.ainvoke([
        SystemMessage(content="你是一个数据研究专家，查询和分析结构化数据。请用中文回答。"),
        HumanMessage(content=prompt),
    ])
    content = response.content if hasattr(response, "content") else str(response)
    return _parse_agent_output(content, task_name)


async def run_internal_kb_research(
    task_name: str, query: str, task_results: dict[str, dict],
) -> tuple[str, list[Evidence]]:
    """Execute internal knowledge base research using rag_specialist."""
    model = _get_model()
    prev = _format_previous_results(task_results)
    prompt = _RESEARCH_AGENT_PROMPT.format(
        task_name=task_name, query=query, previous_results=prev,
    )
    response = await model.ainvoke([
        SystemMessage(content="你是一个内部知识库专家，搜索企业文档和内部资料。请用中文回答。"),
        HumanMessage(content=prompt),
    ])
    content = response.content if hasattr(response, "content") else str(response)
    return _parse_agent_output(content, task_name)


def _parse_agent_output(content: str, task_id: str) -> tuple[str, list[Evidence]]:
    """Parse LLM JSON output into findings + Evidence items."""
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        return content, []

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return content, []

    findings = data.get("findings", content)
    evidence_items = []
    for item in data.get("evidence_items", []):
        confidence = EvidenceConfidence.MEDIUM
        if item.get("confidence") in ("high", "medium", "low"):
            confidence = EvidenceConfidence(item["confidence"])
        evidence_items.append(Evidence(
            task_id=task_id,
            source=EvidenceSource.WEB_SEARCH,
            content=item.get("content", ""),
            citation=item.get("citation", ""),
            confidence=confidence,
        ))

    return findings, evidence_items


AGENT_MAP = {
    "web": run_web_research,
    "graph": run_graph_research,
    "data": run_data_research,
    "internal_kb": run_internal_kb_research,
}
