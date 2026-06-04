"""RetrievalPlanner: query-aware retrieval strategy decision engine.

Replaces hardcoded retrieval decisions with intent-driven strategy selection.
"""

from __future__ import annotations

from pydantic import BaseModel


class RetrievalPlan(BaseModel):
    query_type: str
    use_dense: bool = True
    use_sparse: bool = True
    use_graph: bool = True
    use_community: bool = False
    graph_hops: int = 1
    rerank_top_k: int = 10
    fusion_strategy: str = "rrf"

    @property
    def enabled_channels(self) -> list[str]:
        channels = []
        if self.use_dense:
            channels.append("dense")
        if self.use_sparse:
            channels.append("sparse")
        if self.use_graph:
            channels.append("graph")
        if self.use_community:
            channels.append("community")
        return channels


STRATEGY_MAP: dict[str, RetrievalPlan] = {
    "factoid": RetrievalPlan(
        query_type="factoid",
        use_dense=True, use_sparse=True, use_graph=False, use_community=False,
        graph_hops=0, rerank_top_k=10, fusion_strategy="rrf",
    ),
    "entity_relation": RetrievalPlan(
        query_type="entity_relation",
        use_dense=True, use_sparse=False, use_graph=True, use_community=False,
        graph_hops=1, rerank_top_k=10, fusion_strategy="rrf",
    ),
    "multi_hop": RetrievalPlan(
        query_type="multi_hop",
        use_dense=True, use_sparse=False, use_graph=True, use_community=False,
        graph_hops=3, rerank_top_k=10, fusion_strategy="graph_first",
    ),
    "global_summary": RetrievalPlan(
        query_type="global_summary",
        use_dense=True, use_sparse=False, use_graph=False, use_community=True,
        graph_hops=0, rerank_top_k=10, fusion_strategy="rrf",
    ),
    "temporal": RetrievalPlan(
        query_type="temporal",
        use_dense=True, use_sparse=True, use_graph=True, use_community=False,
        graph_hops=1, rerank_top_k=10, fusion_strategy="rrf",
    ),
    "comparison": RetrievalPlan(
        query_type="comparison",
        use_dense=True, use_sparse=True, use_graph=True, use_community=False,
        graph_hops=1, rerank_top_k=10, fusion_strategy="rrf",
    ),
}


class RetrievalPlanner:
    """Given a query_type, return the optimal RetrievalPlan."""

    def plan(self, query_type: str = "", intent: dict = None) -> RetrievalPlan:
        qtype = query_type or "factoid"
        if intent and intent.get("query_type"):
            qtype = intent["query_type"]
        plan = STRATEGY_MAP.get(qtype)
        if plan is None:
            plan = STRATEGY_MAP["factoid"]
        if intent and intent.get("graph_hops", plan.graph_hops) != plan.graph_hops:
            plan = plan.model_copy(update={"graph_hops": intent["graph_hops"]})
        if intent and intent.get("graph_skip"):
            plan = plan.model_copy(update={"use_graph": False, "use_community": False})
        return plan

    def plan_from_query_type(self, query_type: str) -> RetrievalPlan:
        return STRATEGY_MAP.get(query_type, STRATEGY_MAP["factoid"])


_planner: RetrievalPlanner | None = None


def get_retrieval_planner() -> RetrievalPlanner:
    global _planner
    if _planner is None:
        _planner = RetrievalPlanner()
    return _planner
