# backend/research/research_agents.py
"""ResearchAgent wrappers: convert existing agents into research-mode agents.

Each agent returns structured Evidence instead of conversational answers.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.research.schemas import Evidence, EvidenceSource, EvidenceConfidence


_RESEARCH_AGENT_PROMPT = """You are a research agent conducting a systematic investigation.

Research Task: {task_name}
Research Question: {query}

Context from previous tasks (if any):
{previous_results}

Instructions:
1. Answer the research question thoroughly with specific facts, data points, and citations
2. For every claim, provide a source/citation
3. Output your findings in this JSON format:
{{
  "findings": "your detailed findings with inline citations [source: ...]",
  "citations": ["citation 1", "citation 2"],
  "evidence_items": [
    {{
      "content": "a specific factual claim with context",
      "citation": "source URL or reference",
      "confidence": "high|medium|low"
    }}
  ],
  "confidence": "high|medium|low"
}}

Rules:
- high confidence: multiple reliable sources confirm
- medium confidence: single reliable source
- low confidence: inference or unverified source
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
        SystemMessage(content="You are a web research specialist. Search and synthesize."),
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
        SystemMessage(content="You are a graph research specialist. Explore entity relationships and reason."),
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
        SystemMessage(content="You are a data research specialist. Query and analyze structured data."),
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
        SystemMessage(content="You are an internal knowledge base specialist. Search enterprise documents."),
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
