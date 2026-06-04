"""ReasoningPlanner: converts NL query into structured ReasoningPlan."""

from __future__ import annotations

import re

from backend.rag.graph_reasoning.schemas import ReasoningPlan, ReasoningStrategy

_TYPE_TO_STRATEGY: dict[str, ReasoningStrategy] = {
    "factoid": ReasoningStrategy.FACTOID,
    "entity_relation": ReasoningStrategy.ENTITY_RELATION,
    "multi_hop": ReasoningStrategy.MULTI_HOP,
    "temporal": ReasoningStrategy.TEMPORAL,
    "comparison": ReasoningStrategy.COMPARISON,
    "global_summary": ReasoningStrategy.FACTOID,
}

_REASONING_TYPES = {"multi_hop", "entity_relation", "temporal"}

_ENTITY_NAME_PATTERN = re.compile(
    r"\b(?:"
    r"OpenAI|Google|Microsoft|Apple|Amazon|Meta|Tesla|Netflix|"
    r"Kubernetes|Docker|Redis|Kafka|PostgreSQL|MySQL|Milvus|Neo4j|"
    r"Sam Altman|Elon Musk|Satya Nadella|"
    r"Y Combinator|SoftBank|ByteDance|Tencent"
    r")\b",
    re.IGNORECASE,
)


class ReasoningPlanner:
    """Converts natural language query into a ReasoningPlan."""

    def plan(self, query: str, query_type: str, entity_names: list[str] = None) -> ReasoningPlan:
        strategy = _TYPE_TO_STRATEGY.get(query_type, ReasoningStrategy.FACTOID)
        need_reasoning = query_type in _REASONING_TYPES

        if entity_names is None:
            entity_names = [m.group(0) for m in _ENTITY_NAME_PATTERN.finditer(query)]

        return ReasoningPlan(
            query_type=query_type,
            start_entities=entity_names,
            target_relations=[],
            max_hops=3 if query_type == "multi_hop" else 1,
            reasoning_strategy=strategy,
            need_reasoning=need_reasoning,
        )


_planner: ReasoningPlanner | None = None


def get_reasoning_planner() -> ReasoningPlanner:
    global _planner
    if _planner is None:
        _planner = ReasoningPlanner()
    return _planner
