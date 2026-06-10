# backend/research/conflict_detector.py
"""ConflictDetector: detects contradictions in evidence using LLM."""

from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.research.schemas import EvidenceNode, ConflictDetection


_CONFLICT_PROMPT = """你是一个证据审查专家。比较两条证据，判断是否存在矛盾。

输出纯JSON:
{
  "has_conflict": true/false,
  "conflict_type": "factual|inferential|contextual|none",
  "explanation": "矛盾的简要解释",
  "resolution": "建议如何验证哪条证据更可靠"
}

规则:
- factual: 两个事实陈述互相矛盾（如数字不同）
- inferential: 推理结论矛盾（如对同一事实得出相反结论）
- contextual: 不同背景/时间导致看似矛盾
- 无矛盾则 conflict_type 用 "none"
"""


class ConflictDetector:
    """Uses LLM to detect contradictions between pairs of evidence."""

    async def detect(self, ev_a: EvidenceNode, ev_b: EvidenceNode) -> ConflictDetection:
        if ev_a.node_id == ev_b.node_id:
            return ConflictDetection(evidence_a=ev_a.node_id, evidence_b=ev_b.node_id, has_conflict=False)

        from langchain.chat_models import init_chat_model
        from backend.config import get_settings
        settings = get_settings()

        model = init_chat_model(
            model="qwen-turbo", model_provider="openai",
            api_key=settings.ark_api_key, base_url=settings.base_url,
            temperature=0.0, max_tokens=512, timeout=60,
        )

        prompt = f"""比较以下两条证据:

证据 A: [{ev_a.source}] {ev_a.content}

证据 B: [{ev_b.source}] {ev_b.content}"""

        response = await model.ainvoke([
            SystemMessage(content=_CONFLICT_PROMPT),
            HumanMessage(content=prompt),
        ])

        content = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return ConflictDetection(evidence_a=ev_a.node_id, evidence_b=ev_b.node_id)

        data = json.loads(json_match.group(0))
        return ConflictDetection(
            evidence_a=ev_a.node_id,
            evidence_b=ev_b.node_id,
            has_conflict=data.get("has_conflict", False),
            conflict_type=data.get("conflict_type", "none"),
            explanation=data.get("explanation", ""),
            resolution=data.get("resolution", ""),
        )

    async def detect_all_pairs(self, evidence_nodes: list[EvidenceNode]) -> list[ConflictDetection]:
        """Compare all pairs of evidence (N choose 2 comparisons)."""
        conflicts = []
        for i in range(len(evidence_nodes)):
            for j in range(i + 1, len(evidence_nodes)):
                conflict = await self.detect(evidence_nodes[i], evidence_nodes[j])
                if conflict.has_conflict:
                    conflicts.append(conflict)
        return conflicts


_detector: ConflictDetector | None = None


def get_conflict_detector() -> ConflictDetector:
    global _detector
    if _detector is None:
        _detector = ConflictDetector()
    return _detector
