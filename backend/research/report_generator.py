# backend/research/report_generator.py
"""ResearchReportGenerator: produces evidence-driven research reports."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from backend.research.schemas import ResearchState


class ResearchReportGenerator:
    """Generates structured research reports from collected evidence."""

    async def generate(
        self, state: ResearchState, tenant_id: int, user_id: int,
    ) -> dict:
        """Generate a markdown research report with evidence bindings."""
        from langchain.chat_models import init_chat_model
        from backend.config import get_settings
        settings = get_settings()

        plan = state.plan
        if not plan:
            return {}

        # Build evidence index
        evidence_by_task: dict[str, list] = {}
        for ev in state.evidence:
            evidence_by_task.setdefault(ev.task_id, []).append(ev)

        # Build task results summary
        tasks_summary = ""
        evidence_map = {}
        for task in plan.tasks:
            finding = state.task_results.get(task.task_id, {}).get("finding", "No results")
            tasks_summary += f"\n### {task.name}\n\n{finding}\n"
            task_evidence = evidence_by_task.get(task.task_id, [])
            for ev in task_evidence:
                cite = f"[{ev.id}]"
                evidence_map[ev.id] = ev.citation
                tasks_summary += f"\n> **Evidence {cite}** ({ev.confidence.value} confidence): {ev.content[:300]}\n"
                if ev.citation:
                    tasks_summary += f"> Source: {ev.citation}\n"

        # Generate full report via LLM
        model = init_chat_model(
            model="qwen-turbo",
            model_provider="openai",
            api_key=settings.ark_api_key,
            base_url=settings.base_url,
            temperature=0.0,
            max_tokens=4096,
            timeout=60,
        )
        from langchain_core.messages import SystemMessage, HumanMessage

        import json
        refs_text = "\n".join(f"[{eid}]: {url}" for eid, url in list(evidence_map.items())[:50])

        prompt = f"""Generate a professional research report based on the following evidence.

Research Goal: {plan.goal}

Collected Evidence and Findings:
{tasks_summary[:12000]}

Evidence IDs and Sources:
{refs_text}

Structure the report with:
1. Executive Summary (key takeaways in 3-5 sentences)
2. Key Findings (numbered list, each backed by evidence)
3. Detailed Analysis (organized by topic, not by task)
4. Implications & Recommendations
5. Limitations & Gaps
6. References (all evidence citations)

CRITICAL: Every factual claim must reference an Evidence ID in brackets, e.g. [ev_abc123].
Use markdown formatting: ## headings, **bold**, bullet points, > blockquotes for evidence."""

        response = await model.ainvoke([
            SystemMessage(content="You are a senior research analyst writing an evidence-driven report. Every claim must cite its source evidence ID."),
            HumanMessage(content=prompt),
        ])
        content = response.content if hasattr(response, "content") else str(response)

        report_id = f"rpt_{uuid.uuid4().hex[:12]}"

        return {
            "report_id": report_id,
            "execution_id": state.execution_id,
            "title": plan.goal,
            "format": "markdown",
            "content": content,
            "evidence_map": evidence_map,
            "executive_summary": self._extract_summary(content),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _extract_summary(self, content: str) -> str:
        """Extract executive summary section from report."""
        import re
        match = re.search(
            r"(?:Executive Summary|概要|摘要)[\s\S]*?(?=##|\Z)",
            content, re.IGNORECASE,
        )
        if match:
            return match.group(0).strip()[:500]
        # Fallback: first paragraph after title
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped[:500]
        return ""


_generator: ResearchReportGenerator | None = None


def get_report_generator() -> ResearchReportGenerator:
    global _generator
    if _generator is None:
        _generator = ResearchReportGenerator()
    return _generator
