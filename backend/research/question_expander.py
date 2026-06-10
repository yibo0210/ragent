# backend/research/question_expander.py
"""QuestionExpander: generates follow-up questions from conflicts and unverified hypotheses."""

from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.research.schemas import Hypothesis, ConflictDetection, ExpandedQuestion


_EXPANDER_PROMPT = """你是一个研究追问专家。基于当前研究状态，生成需要进一步调查的问题。

输出纯JSON:
{
  "questions": [
    {
      "question": "追问内容",
      "source": "conflict|hypothesis_unverified|evidence_gap",
      "priority": 0.8,
      "target_hypothesis": "H1"
    }
  ]
}

规则:
- conflict: 证据矛盾需要进一步验证
- hypothesis_unverified: 假设尚未被充分验证
- evidence_gap: 缺少关于某个方面的证据
"""


class QuestionExpander:
    """Generates follow-up research questions from gaps and conflicts."""

    async def expand(
        self,
        goal: str,
        hypotheses: list[Hypothesis],
        conflicts: list[ConflictDetection],
        verified_hypotheses: list[str],
    ) -> list[ExpandedQuestion]:
        from langchain.chat_models import init_chat_model
        from backend.config import get_settings
        settings = get_settings()

        model = init_chat_model(
            model="qwen-turbo", model_provider="openai",
            api_key=settings.ark_api_key, base_url=settings.base_url,
            temperature=0.0, max_tokens=1024, timeout=60,
        )

        hyp_text = "\n".join(
            f"- {h.hypothesis_id}: {h.statement} (状态: {h.status.value})"
            for h in hypotheses
        )
        conflict_text = "\n".join(
            f"- {c.evidence_a} vs {c.evidence_b}: {c.explanation}"
            for c in conflicts[:10]
        )
        unverified = [h.hypothesis_id for h in hypotheses if h.hypothesis_id not in verified_hypotheses]

        prompt = f"""研究目标: {goal}

假设状态:
{hyp_text or '(无假设)'}

已发现的矛盾:
{conflict_text or '(无矛盾)'}

未验证的假设: {', '.join(unverified) if unverified else '(无)'}

请生成需要追问的问题:"""

        response = await model.ainvoke([
            SystemMessage(content=_EXPANDER_PROMPT),
            HumanMessage(content=prompt),
        ])

        content = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return []

        data = json.loads(json_match.group(0))
        return [
            ExpandedQuestion(
                question=item["question"],
                source=item.get("source", "evidence_gap"),
                priority=item.get("priority", 0.5),
                target_hypothesis=item.get("target_hypothesis", ""),
            )
            for item in data.get("questions", [])
        ]


_expander: QuestionExpander | None = None


def get_question_expander() -> QuestionExpander:
    global _expander
    if _expander is None:
        _expander = QuestionExpander()
    return _expander
