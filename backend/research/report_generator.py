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
            for idx, ev in enumerate(task_evidence, 1):
                tag = f"[证据{idx}]"
                evidence_map[tag] = ev.citation or ev.content[:100]
                tasks_summary += f"\n> **{tag}** ({ev.confidence.value} 置信度): {ev.content[:300]}\n"
                if ev.citation:
                    tasks_summary += f"> 来源: {ev.citation}\n"

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
        refs_text = "\n".join(f"- {tag}: {url}" for tag, url in list(evidence_map.items())[:50])

        prompt = f"""根据以下证据生成一份专业的中文研究报告。

研究目标: {plan.goal}

收集的证据和发现:
{tasks_summary[:12000]}

证据来源清单:
{refs_text}

报告结构要求（全部使用中文撰写）:
1. 摘要 (3-5句话概括核心发现)
2. 关键发现 (编号列表，每一条附上证据引用标签如 [证据1])
3. 详细分析 (按主题组织，不是按任务)
4. 影响与建议
5. 局限性与不足
6. 参考文献

重要：每个事实性结论必须标注证据引用标签，如 [证据1]、[证据2]。
使用 Markdown 格式：## 标题、**加粗**、列表、> 引用证据。"""

        response = await model.ainvoke([
            SystemMessage(content="你是一位资深研究分析师，撰写证据驱动的研究报告。每个结论必须引用证据标签（如[证据1]）。请使用中文撰写。"),
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
            r"(?:摘要|Executive Summary)[\s\S]*?(?=##|\Z)",
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
