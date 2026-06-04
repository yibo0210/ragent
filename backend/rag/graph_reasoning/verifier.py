"""ReasoningVerifier: validates answers against discovered reasoning paths."""

from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.rag.graph_reasoning.schemas import ReasoningPath, VerificationResult, Verdict


_VERIFIER_PROMPT = """Given a question, the discovered reasoning paths, and a draft answer, determine if the answer is supported by the paths.

Output JSON:
{
  "verdict": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED",
  "confidence": 0.0-1.0,
  "explanation": "why you made this verdict",
  "supporting_paths": [0, 1, ...]
}"""


class ReasoningVerifier:
    """LLM-powered reasoning verification."""

    async def verify(
        self,
        question: str,
        paths: list[ReasoningPath],
        draft_answer: str,
    ) -> VerificationResult:
        if not paths:
            return VerificationResult(
                verdict=Verdict.UNSUPPORTED,
                confidence=0.0,
                explanation="No reasoning paths available",
            )

        paths_text = ""
        for i, path in enumerate(paths):
            chain_parts = []
            for j in range(len(path.nodes)):
                chain_parts.append(f"({path.nodes[j]})")
                if j < len(path.edges):
                    chain_parts.append(f"-[{path.edges[j]}]->")
            paths_text += f"Path {i}: {''.join(chain_parts)}\n"

        prompt = f"""Question: {question}

Reasoning Paths:
{paths_text[:4000]}

Draft Answer: {draft_answer[:2000]}

{_VERIFIER_PROMPT}"""

        try:
            from backend.agent.model_router import get_model_for_agent
            model = get_model_for_agent("supervisor")
            response = await model.ainvoke([
                SystemMessage(content="You verify reasoning chains."),
                HumanMessage(content=prompt),
            ])
            content = response.content if hasattr(response, "content") else str(response)
            json_match = re.search(r"\{[\s\S]*\}", content)
            if not json_match:
                return VerificationResult(verdict=Verdict.PARTIAL, confidence=0.5,
                    explanation=f"Failed to parse LLM output: {content[:100]}")
            data = json.loads(json_match.group(0))
            return VerificationResult(
                verdict=Verdict(data.get("verdict", "PARTIAL")),
                confidence=float(data.get("confidence", 0.5)),
                explanation=data.get("explanation", ""),
                supporting_paths=data.get("supporting_paths", []),
            )
        except Exception as e:
            return VerificationResult(verdict=Verdict.PARTIAL, confidence=0.5,
                explanation=f"Verification error: {e}")


_verifier: ReasoningVerifier | None = None


def get_reasoning_verifier() -> ReasoningVerifier:
    global _verifier
    if _verifier is None:
        _verifier = ReasoningVerifier()
    return _verifier
