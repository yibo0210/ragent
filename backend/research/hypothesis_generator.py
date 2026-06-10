# backend/research/hypothesis_generator.py
"""HypothesisGenerator: generates competing hypotheses from a research goal."""

from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.research.schemas import Hypothesis


_HYPOTHESIS_PROMPT = """你是一个研究假设生成专家。给定研究目标，提出 2-4 个竞争性假设。

每个假设包含:
- hypothesis_id: H1, H2, H3, ...
- statement: 用中文陈述假设
- rationale: 为什么提出这个假设（1-2句）
- verification_query: 验证这个假设需要研究什么问题

输出纯JSON:
{
  "hypotheses": [
    {
      "hypothesis_id": "H1",
      "statement": "组织结构问题是Intel AI失败的主因",
      "rationale": "Intel在2018-2023年间频繁更换CEO，战略方向多次转变",
      "verification_query": "Intel CEO变更历史及其对AI战略的具体影响",
      "priority": 0.9
    }
  ]
}

规则:
- 假设必须互斥或互补，不能重复
- 每个假设都要具体可验证，不是空泛陈述
- priority 0-1: 越关键假设优先级越高
"""


class HypothesisGenerator:
    """Generates competing hypotheses from a research goal via LLM."""

    async def generate(self, goal: str, context: str = "") -> list[Hypothesis]:
        from langchain.chat_models import init_chat_model
        from backend.config import get_settings
        settings = get_settings()

        model = init_chat_model(
            model="qwen-turbo", model_provider="openai",
            api_key=settings.ark_api_key, base_url=settings.base_url,
            temperature=0.3, max_tokens=1024, timeout=60,
        )

        prompt = f"研究目标: {goal}\n"
        if context:
            prompt += f"背景信息: {context[:2000]}\n"
        prompt += "\n请生成竞争性假设:"

        response = await model.ainvoke([
            SystemMessage(content=_HYPOTHESIS_PROMPT),
            HumanMessage(content=prompt),
        ])

        content = response.content if hasattr(response, "content") else str(response)
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            return []

        data = json.loads(json_match.group(0))
        return [
            Hypothesis(
                hypothesis_id=item["hypothesis_id"],
                statement=item.get("statement", ""),
                rationale=item.get("rationale", ""),
                verification_tasks=[item.get("verification_query", "")],
            )
            for item in data.get("hypotheses", [])
        ]


_generator: HypothesisGenerator | None = None


def get_hypothesis_generator() -> HypothesisGenerator:
    global _generator
    if _generator is None:
        _generator = HypothesisGenerator()
    return _generator
