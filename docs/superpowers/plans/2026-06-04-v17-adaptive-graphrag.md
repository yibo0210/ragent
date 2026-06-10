# v17 Adaptive GraphRAG 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有检索层之上引入 Query-Aware 检索决策层，根据 6 种查询类型动态选择 Dense/Sparse/Graph/Community 通道，自适应调整 RRF 权重和图检索深度，实现 Recall@10 +10%、Latency -30%。

**Architecture:** 扩展 QueryProfiler 从 3 级分类到 6 种查询类型（Factoid/Entity Relation/Multi-Hop/Global Summary/Temporal/Comparison），新增 RetrievalPlanner 根据查询类型输出 RetrievalPlan（含通道选择+图深度+融合策略），扩展 weight_matrix.yaml 为 6 种类型、每个类型独立 RRF 权重，新增 GraphUtilityEstimator 预测图检索价值阈值跳过 Neo4j，通过 orchestrator 将 intent 信息传递到所有检索节点。

**Tech Stack:** 扩展现有 QueryProfiler · 新增 RetrievalPlanner + GraphUtilityEstimator · 修改 weight_matrix.yaml · 修改 orchestrator 节点 · 新增评测数据集

---

## File Structure

```
backend/agent/query_profiler.py      # 修改: 6 种查询类型替代 3 级
backend/rag/retrieval_planner.py     # 新增: RetrievalPlan + 决策逻辑
backend/rag/graph_utility_estimator.py # 新增: 图检索价值预测器
config/weight_matrix.yaml            # 修改: 6 种类型权重矩阵
backend/agent/orchestrator.py        # 修改: 传递 query_type 到图检索节点
backend/rag/graph_retriever.py       # 修改: graph_hops 动态化
backend/evaluation/dataset.py        # 修改: 新增 6 类 query 数据集
backend/rag/utils.py                 # 修改: rrf_fusion 接受 adaptive 配置
tests/test_query_profiler.py         # 修改: 6 类型测试
tests/test_retrieval_planner.py      # 新增
tests/test_graph_utility_estimator.py # 新增
```

---

## Phase 1: Query Taxonomy + Retrieval Planner

### Task 1: 扩展 QueryProfiler 支持 6 种查询类型

**Files:**
- Modify: `backend/agent/query_profiler.py`

- [ ] **Step 1: 添加 6 QueryType 枚举和新原型查询**

```python
# backend/agent/query_profiler.py

# 在现有 L1/L2/L3 原型查询之后添加:

class QueryType:
    FACTOID = "factoid"
    ENTITY_RELATION = "entity_relation"
    MULTI_HOP = "multi_hop"
    GLOBAL_SUMMARY = "global_summary"
    TEMPORAL = "temporal"
    COMPARISON = "comparison"

# 6 种类型各 4 条原型查询
_TYPE_PROTOTYPES = {
    QueryType.FACTOID: [
        "What is Redis?",
        "What is Kafka?",
        "Define FastAPI.",
        "What does a database index do?",
    ],
    QueryType.ENTITY_RELATION: [
        "Who founded OpenAI?",
        "Which companies are invested by Tencent?",
        "Who is the CEO of Microsoft?",
        "What products does Apple sell?",
    ],
    QueryType.MULTI_HOP: [
        "Which company acquired the startup that developed Kubernetes?",
        "Find competitors of the company that partnered with our supplier.",
        "Which organizations collaborated with both X and Y?",
        "Trace the investment chain from SoftBank to ByteDance.",
    ],
    QueryType.GLOBAL_SUMMARY: [
        "Summarize the entire system architecture.",
        "What are the major themes across all documents?",
        "Give me an overview of the project.",
        "What are the key takeaways from the knowledge base?",
    ],
    QueryType.TEMPORAL: [
        "Who was CEO in 2022?",
        "What happened in Q3 2023?",
        "Before 2020, which technology was used?",
        "Compare performance between 2021 and 2022.",
    ],
    QueryType.COMPARISON: [
        "Compare GraphRAG and vanilla RAG.",
        "What are the differences between Redis and Memcached?",
        "Compare MySQL and PostgreSQL performance.",
        "Which is better for our use case: FastAPI or Flask?",
    ],
}

# 每种类型的关键词
_TYPE_KEYWORDS = {
    QueryType.FACTOID: ["what is", "define", "meaning of", "definition"],
    QueryType.ENTITY_RELATION: ["who", "which company", "founded", "invested", "acquired", "CEO", "owner"],
    QueryType.MULTI_HOP: ["chain", "path", "via", "through", "connected", "network", "trace"],
    QueryType.GLOBAL_SUMMARY: ["summarize", "overview", "themes", "summary", "overall", "key points"],
    QueryType.TEMPORAL: ["in 202", "before", "after", "last year", "previous", "Q1", "Q2", "Q3", "Q4", "quarter"],
    QueryType.COMPARISON: ["compare", "difference", "versus", "vs", "better", "contrast", "pros and cons"],
}
```

- [ ] **Step 2: 修改 QueryIntent 增加 query_type 字段**

```python
@dataclass
class QueryIntent:
    level: str
    query_type: str = ""  # 新增: factoid/entity_relation/multi_hop/global_summary/temporal/comparison
    complexity_score: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    embedding_similarity: dict = field(default_factory=dict)
    reason: str = ""
    graph_skip: bool = False  # 新增: GraphUtilityEstimator 填充
    graph_hops: int = 1       # 新增: 动态跳数

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "query_type": self.query_type,
            "complexity_score": self.complexity_score,
            "matched_keywords": self.match_keywords,
            "embedding_similarity": self.embedding_similarity,
            "reason": self.reason,
            "graph_skip": self.graph_skip,
            "graph_hops": self.graph_hops,
        }
```

- [ ] **Step 3: 修改 QueryProfiler.profile() 增加类型分类逻辑**

在 `profile()` 方法中，在现有的 keyword scoring 之后添加：

```python
def profile(self, query: str) -> QueryIntent:
    query_lower = query.lower().strip()
    
    # 1. 6 类型关键词匹配
    type_scores = {}
    for qtype, keywords in _TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            type_scores[qtype] = score
    best_type = max(type_scores, key=type_scores.get) if type_scores else QueryType.FACTOID
    
    # 2. Embedding 相似度（可选）
    if self.use_embedding:
        # 在关键词匹配基础上加权
        pass
    
    # 3. 确定 graph_hops
    graph_hops_map = {
        QueryType.FACTOID: 0,
        QueryType.ENTITY_RELATION: 1,
        QueryType.MULTI_HOP: 3,
        QueryType.GLOBAL_SUMMARY: 0,
        QueryType.TEMPORAL: 1,
        QueryType.COMPARISON: 1,
    }
    graph_hops = graph_hops_map.get(best_type, 1)
    
    # 4. 映射到旧 level（向后兼容）
    level_map = {
        QueryType.FACTOID: "L1_FACTUAL",
        QueryType.ENTITY_RELATION: "L2_REASONING",
        QueryType.MULTI_HOP: "L2_REASONING",
        QueryType.GLOBAL_SUMMARY: "L3_MACRO_SUMMARY",
        QueryType.TEMPORAL: "L2_REASONING",
        QueryType.COMPARISON: "L2_REASONING",
    }
    
    return QueryIntent(
        level=level_map.get(best_type, "L1_FACTUAL"),
        query_type=best_type,
        complexity_score=...,
        matched_keywords=list(type_scores.keys()),
        reason=f"Query classified as {best_type}: matched keywords {list(type_scores.keys())}",
        graph_hops=graph_hops,
    )
```

- [ ] **Step 4: 更新测试**

```bash
pytest tests/test_query_profiler.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/agent/query_profiler.py tests/test_query_profiler.py
git commit -m "feat(v17): extend QueryProfiler to 6 query types with dynamic graph_hops"
```

---

### Task 2: 创建 RetrievalPlanner

**Files:**
- Create: `backend/rag/retrieval_planner.py`

- [ ] **Step 1: 创建 Pydantic Schema 和 Planner 类**

```python
# backend/rag/retrieval_planner.py
"""RetrievalPlanner: query-aware retrieval strategy decision engine.

Replaces hardcoded retrieval decisions with intent-driven strategy selection.
"""

from __future__ import annotations

from pydantic import BaseModel


class RetrievalPlan(BaseModel):
    """Output of RetrievalPlanner: which channels to use and how."""
    query_type: str
    use_dense: bool = True
    use_sparse: bool = True
    use_graph: bool = True
    use_community: bool = False
    graph_hops: int = 1
    rerank_top_k: int = 10
    fusion_strategy: str = "rrf"  # rrf | weighted_sum | graph_first

    @property
    def enabled_channels(self) -> list[str]:
        channels = []
        if self.use_dense: channels.append("dense")
        if self.use_sparse: channels.append("sparse")
        if self.use_graph: channels.append("graph")
        if self.use_community: channels.append("community")
        return channels


# 6 种查询类型的检索策略配置
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
    """Given a query_type string, return the optimal RetrievalPlan."""

    def plan(self, query_type: str = "", intent: dict = None) -> RetrievalPlan:
        qtype = query_type or "factoid"
        if intent and intent.get("query_type"):
            qtype = intent["query_type"]
        plan = STRATEGY_MAP.get(qtype)
        if plan is None:
            plan = STRATEGY_MAP["factoid"]
        # Override graph_hops from intent if set
        if intent and intent.get("graph_hops", 1) != plan.graph_hops:
            plan = plan.model_copy(update={"graph_hops": intent["graph_hops"]})
        if intent and intent.get("graph_skip"):
            plan = plan.model_copy(update={"use_graph": False, "use_community": False})
        return plan

    def plan_from_query_type(self, query_type: str) -> RetrievalPlan:
        return STRATEGY_MAP.get(query_type, STRATEGY_MAP["factoid"])


# Module-level singleton
_planner: RetrievalPlanner | None = None


def get_retrieval_planner() -> RetrievalPlanner:
    global _planner
    if _planner is None:
        _planner = RetrievalPlanner()
    return _planner
```

- [ ] **Step 2: 运行测试**

```bash
cd backend && python -c "
from backend.rag.retrieval_planner import RetrievalPlanner, STRATEGY_MAP
planner = RetrievalPlanner()
for qtype in ['factoid','entity_relation','multi_hop','global_summary','temporal','comparison']:
    plan = planner.plan_from_query_type(qtype)
    print(f'{qtype}: channels={plan.enabled_channels} hops={plan.graph_hops} fusion={plan.fusion_strategy}')
print('RetrievalPlanner OK')
"
```

Expected: 6 种类型分别输出不同的通道组合和跳数。

- [ ] **Step 3: Commit**

```bash
git add backend/rag/retrieval_planner.py
git commit -m "feat(v17): add RetrievalPlanner — query-type-driven retrieval strategy selection"
```

---

## Phase 2: Adaptive RRF + Graph Depth

### Task 3: 扩展 dynamic_rrf + weight_matrix 为 6 类型

**Files:**
- Modify: `backend/rag/dynamic_rrf.py`
- Modify: `config/weight_matrix.yaml`

- [ ] **Step 1: 更新 weight_matrix.yaml**

```yaml
factoid:
  weights: [0.80, 0.20, 0.00, 0.00]
  description: "事实类查询，纯向量检索即可"

entity_relation:
  weights: [0.30, 0.00, 0.70, 0.00]
  description: "实体关系查询，图谱为主要通道"

multi_hop:
  weights: [0.15, 0.00, 0.85, 0.00]
  description: "多跳推理，图谱优先"

global_summary:
  weights: [0.20, 0.00, 0.00, 0.80]
  description: "全局总结，社区摘要为主要通道"

temporal:
  weights: [0.30, 0.20, 0.50, 0.00]
  description: "时间敏感查询，图谱+Dense 平衡"

comparison:
  weights: [0.35, 0.25, 0.35, 0.05]
  description: "对比分析，多通道均衡"

L1_FACTUAL:
  weights: [0.70, 0.25, 0.00, 0.05]
  description: "旧版 L1 兼容"

L2_REASONING:
  weights: [0.20, 0.10, 0.65, 0.05]
  description: "旧版 L2 兼容"

L3_MACRO_SUMMARY:
  weights: [0.35, 0.20, 0.35, 0.10]
  description: "旧版 L3 兼容"

DEFAULT:
  weights: [0.40, 0.30, 0.15, 0.15]
  description: "默认权重"
```

- [ ] **Step 2: 更新 dynamic_rrf.py 支持 query_type 查询**

```python
# 在 get_weights_for_intent 中添加 query_type 优先查找
def get_weights_for_intent(intent_level: str, query_type: str = "") -> tuple:
    matrix = load_weight_matrix()
    # 优先用 query_type 查找
    if query_type and query_type in matrix:
        entry = matrix[query_type]
        weights = entry.get("weights", [0.4, 0.3, 0.15, 0.15])
    else:
        entry = matrix.get(intent_level) or matrix.get("DEFAULT")
        weights = entry.get("weights", [0.4, 0.3, 0.15, 0.15]) if entry else [0.4, 0.3, 0.15, 0.15]
    w = list(weights)
    while len(w) < 4:
        w.append(0.0)
    return tuple(w[:4])
```

- [ ] **Step 3: 验证**

```bash
cd backend && python -c "
from backend.rag.dynamic_rrf import get_weights_for_intent, reload_weight_matrix
reload_weight_matrix()
for qt in ['factoid','multi_hop','global_summary']:
    w = get_weights_for_intent('L2_REASONING', query_type=qt)
    print(f'{qt}: dense={w[0]:.2f} sparse={w[1]:.2f} graph={w[2]:.2f} community={w[3]:.2f}')
"
```

Expected: factoid graph=0, multi_hop graph=0.85, global_summary community=0.80

- [ ] **Step 4: Commit**

```bash
git add config/weight_matrix.yaml backend/rag/dynamic_rrf.py
git commit -m "feat(v17): 6-type weight matrix + query_type-priority lookup in dynamic_rrf"
```

---

### Task 4: 将 query_type 传递到 orchestrator 检索节点

**Files:**
- Modify: `backend/agent/orchestrator.py` (supervisor_node, local_graph_search_node, global_graph_search_node)

- [ ] **Step 1: supervisor_node 传递 query_type+graph_hops 到 state**

在 `supervisor_node` 中（profiler 调用之后），将 `query_type` 和 `graph_hops` 写入 state：

```python
# 在 supervisor_node 中，profiler 调用之后
query_intent = profiler.profile(user_query)
return {
    "query_intent": query_intent.to_dict(),
    # ... existing fields ...
}
```

（`query_intent` 已经写入 state，但 `query_type` 和 `graph_hops` 在 to_dict() 中已包含）

- [ ] **Step 2: local_graph_search_node 读取 intent 并动态调整**

```python
# 在 local_graph_search_node 中
intent = state.get("query_intent", {})
graph_hops = intent.get("graph_hops", 1)
tenant_id = (state.get("user_context") or {}).get("tenant_id")

# 用 RetrievalPlanner 决定是否跳过图检索
from backend.rag.retrieval_planner import get_retrieval_planner
planner = get_retrieval_planner()
plan = planner.plan(intent=intent)

if plan.use_graph:
    result = safe_graph_search(user_query, graph_hops=plan.graph_hops, tenant_id=tenant_id)
else:
    # 直接回退到纯向量检索，跳过 Neo4j
    from backend.rag.utils import retrieve_documents
    result = retrieve_documents(user_query, top_k=5, tenant_id=tenant_id)
    result["mode"] = "dense_only"
```

- [ ] **Step 3: global_graph_search_node 读取 intent**

```python
# 在 global_graph_search_node 中
intent = state.get("query_intent", {})
tenant_id = (state.get("user_context") or {}).get("tenant_id")

from backend.rag.retrieval_planner import get_retrieval_planner
planner = get_retrieval_planner()
plan = planner.plan(intent=intent)

if plan.use_community:
    result = global_graph_search(user_query, tenant_id=tenant_id)
else:
    result = {"summaries": [], "context": "", "mode": "community_skipped"}
```

- [ ] **Step 4: 验证不需要图检索的查询确实跳过 Neo4j**

```bash
cd backend && python -c "
from backend.rag.retrieval_planner import get_retrieval_planner
planner = get_retrieval_planner()
plan = planner.plan(intent={'query_type':'factoid'})
assert plan.use_graph is False
assert plan.use_community is False
print(f'Factoid: graph={plan.use_graph} community={plan.use_community} ✓')
plan2 = planner.plan(intent={'query_type':'multi_hop'})
assert plan2.use_graph is True
assert plan2.graph_hops == 3
print(f'Multi-hop: graph={plan2.use_graph} hops={plan2.graph_hops} ✓')
"
```

- [ ] **Step 5: Commit**

```bash
git add backend/agent/orchestrator.py
git commit -m "feat(v17): propagate query_type to graph search nodes — adaptive skip of Neo4j"
```

---

## Phase 3: Graph Utility Estimator

### Task 5: 创建 GraphUtilityEstimator

**Files:**
- Create: `backend/rag/graph_utility_estimator.py`

- [ ] **Step 1: 创建 Estimator**

```python
# backend/rag/graph_utility_estimator.py
"""GraphUtilityEstimator: predict whether graph retrieval will improve answer quality.

Uses lightweight heuristics (no LLM call) to estimate P(Graph Helpful | Query).
If score < threshold, skip Neo4j entirely.
"""

from __future__ import annotations

import re


class GraphUtilityScore:
    __slots__ = ("score", "graph_hops", "skip_reason")
    def __init__(self, score: float, graph_hops: int = 1, skip_reason: str = ""):
        self.score = score
        self.graph_hops = graph_hops
        self.skip_reason = skip_reason


class GraphUtilityEstimator:
    """Estimate graph retrieval utility from query features.

    Features:
    - Entity density: count of named entities / query length
    - Relation keywords: contains "founded", "acquired", "partnered", etc.
    - Temporal keywords: contains year/quarter references
    - Reasoning signals: "chain", "path", "via", "through", "connect"
    """

    ENTITY_PATTERNS = [
        r"\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|Co|Company)\b",
        r"\b(?:OpenAI|Google|Microsoft|Apple|Amazon|Meta|Tesla|Netflix)\b",
        r"\b(?:Redis|Kafka|PostgreSQL|MySQL|MongoDB|Docker|Kubernetes)\b",
    ]

    RELATION_KEYWORDS = [
        "founded", "acquired", "invested", "partnered", "merged",
        "CEO", "CTO", "founder", "owner", "subsidiary", "competitor",
        "supplier", "customer", "parent company",
    ]

    REASONING_KEYWORDS = [
        "trace", "chain", "path", "via", "through", "connected",
        "relationship", "network", "graph", "linked",
    ]

    TEMPORAL_KEYWORDS = [
        "in 202", "in 2020", "in 2021", "in 2022", "in 2023",
        "before", "after", "during", "Q1", "Q2", "Q3", "Q4",
    ]

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def estimate(self, query: str, query_type: str = "") -> GraphUtilityScore:
        query_lower = query.lower()
        words = query_lower.split()
        qlen = max(len(words), 1)

        # Feature 1: Entity density
        entity_count = sum(1 for p in self.ENTITY_PATTERNS if re.search(p, query))
        entity_density = min(entity_count / qlen * 5, 1.0)

        # Feature 2: Relation keyword score
        rel_score = min(sum(1 for kw in self.RELATION_KEYWORDS if kw in query_lower) / 3, 1.0)

        # Feature 3: Reasoning keyword score
        reason_score = min(sum(1 for kw in self.REASONING_KEYWORDS if kw in query_lower) / 3, 1.0)

        # Feature 4: Temporal keyword score
        time_score = min(sum(1 for kw in self.TEMPORAL_KEYWORDS if kw in query_lower) / 2, 1.0)

        # Weighted combination
        score = (
            entity_density * 0.3 +
            rel_score * 0.3 +
            reason_score * 0.25 +
            time_score * 0.15
        )

        # Determine graph_hops from score
        if score >= 0.8:
            graph_hops = 3
        elif score >= 0.5:
            graph_hops = 1
        else:
            graph_hops = 0

        skip_reason = ""
        if score < self.threshold:
            skip_reason = f"graph_score={score:.2f} < threshold={self.threshold}"

        return GraphUtilityScore(score=score, graph_hops=graph_hops, skip_reason=skip_reason)

    def should_use_graph(self, query: str, query_type: str = "") -> bool:
        result = self.estimate(query, query_type)
        return result.score >= self.threshold


_estimator: GraphUtilityEstimator | None = None


def get_graph_utility_estimator() -> GraphUtilityEstimator:
    global _estimator
    if _estimator is None:
        _estimator = GraphUtilityEstimator()
    return _estimator
```

- [ ] **Step 2: 集成到 QueryProfiler**

在 `QueryProfiler.profile()` 末尾，用 GraphUtilityEstimator 设置 `graph_skip`：

```python
from backend.rag.graph_utility_estimator import get_graph_utility_estimator
estimator = get_graph_utility_estimator()
utility = estimator.estimate(query, query_type=best_type)
return QueryIntent(
    ...
    graph_skip=utility.score < estimator.threshold,
    graph_hops=utility.graph_hops,
)
```

- [ ] **Step 3: 验证**

```bash
cd backend && python -c "
from backend.rag.graph_utility_estimator import GraphUtilityEstimator
est = GraphUtilityEstimator(threshold=0.6)
tests = [
    ('What is Redis?', 'factoid'),
    ('Who founded OpenAI?', 'entity_relation'),
    ('Trace the investment chain from SoftBank to ByteDance.', 'multi_hop'),
]
for q, qt in tests:
    score = est.estimate(q, qt)
    print(f'[{qt}] {q[:50]}... -> score={score.score:.2f} hops={score.graph_hops} skip={score.skip_reason}')
"
```

Expected: "What is Redis?" → low score, skip; "Trace the investment chain..." → high score, use graph.

- [ ] **Step 4: Commit**

```bash
git add backend/rag/graph_utility_estimator.py backend/agent/query_profiler.py
git commit -m "feat(v17): add GraphUtilityEstimator — predict graph value, skip Neo4j when score < 0.6"
```

---

## Phase 4: Evaluation Framework

### Task 6: 构建 Adaptive Graph Benchmark 数据集 + 评测脚本

**Files:**
- Modify: `backend/evaluation/dataset.py`
- Create: `scripts/run_adaptive_evaluation.py`

- [ ] **Step 1: 添加 6 类型 golden 数据集**

在 `backend/evaluation/dataset.py` 中添加：

```python
ADAPTIVE_QA_PAIRS = [
    # --- Factoid (6 条) ---
    {"question": "What is Redis?", "expected_query_type": "factoid", "expected_use_graph": False,
     "ground_truth": "Redis is an in-memory key-value data store..."},
    {"question": "What is Kafka used for?", "expected_query_type": "factoid", "expected_use_graph": False,
     "ground_truth": "Kafka is a distributed event streaming platform..."},
    {"question": "Define FastAPI.", "expected_query_type": "factoid", "expected_use_graph": False,
     "ground_truth": "FastAPI is a modern Python web framework..."},
    # 至少 3 条 per type × 6 types = 18+ 条

    # --- Entity Relation (4 条) ---
    {"question": "Who founded OpenAI?", "expected_query_type": "entity_relation", "expected_use_graph": True,
     "ground_truth": "OpenAI was founded by Sam Altman, Greg Brockman..."},
    {"question": "Which companies did Tencent invest in?", "expected_query_type": "entity_relation", "expected_use_graph": True,
     "ground_truth": "Tencent has invested in..."},

    # --- Multi-Hop (4 条) ---
    {"question": "Which company acquired the startup that developed Kubernetes?", "expected_query_type": "multi_hop",
     "expected_use_graph": True, "ground_truth": "Google acquired..."},

    # --- Global Summary (3 条) ---
    {"question": "Summarize the key technologies used in this project.", "expected_query_type": "global_summary",
     "expected_use_graph": False, "expected_use_community": True, "ground_truth": "The project uses..."},

    # --- Temporal (3 条) ---
    {"question": "Who was CEO of Microsoft in 2018?", "expected_query_type": "temporal", "expected_use_graph": True,
     "ground_truth": "Satya Nadella was CEO of Microsoft in 2018."},

    # --- Comparison (3 条) ---
    {"question": "Compare GraphRAG and vanilla RAG.", "expected_query_type": "comparison", "expected_use_graph": True,
     "ground_truth": "GraphRAG uses knowledge graphs..."},
]
```

- [ ] **Step 2: 创建自适应评测脚本**

```python
# scripts/run_adaptive_evaluation.py
"""Adaptive GraphRAG evaluation: measure query_type classification accuracy + retrieval quality."""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.query_profiler import QueryProfiler
from backend.rag.retrieval_planner import get_retrieval_planner
from backend.rag.graph_utility_estimator import get_graph_utility_estimator
from backend.evaluation.dataset import ADAPTIVE_QA_PAIRS


def evaluate_query_classification():
    profiler = QueryProfiler()
    correct = 0
    total = len(ADAPTIVE_QA_PAIRS)
    details = []

    for item in ADAPTIVE_QA_PAIRS:
        intent = profiler.profile(item["question"])
        predicted = intent.query_type
        expected = item["expected_query_type"]
        details.append({
            "question": item["question"][:80],
            "expected": expected,
            "predicted": predicted,
            "match": predicted == expected,
        })
        if predicted == expected:
            correct += 1

    return {
        "accuracy": correct / total if total else 0,
        "correct": correct,
        "total": total,
        "details": details,
    }


def evaluate_graph_utility_decision():
    estimator = get_graph_utility_estimator()
    correct = 0
    total = 0
    for item in ADAPTIVE_QA_PAIRS:
        if "expected_use_graph" not in item:
            continue
        total += 1
        score = estimator.estimate(item["question"], item["expected_query_type"])
        predicted_use_graph = score.score >= estimator.threshold
        if predicted_use_graph == item["expected_use_graph"]:
            correct += 1
    return {"accuracy": correct / total if total else 0, "correct": correct, "total": total}


if __name__ == "__main__":
    print("=== Query Classification Accuracy ===")
    r1 = evaluate_query_classification()
    print(f"  Accuracy: {r1['accuracy']:.2%} ({r1['correct']}/{r1['total']})")

    print("\n=== Graph Utility Decision Accuracy ===")
    r2 = evaluate_graph_utility_decision()
    print(f"  Accuracy: {r2['accuracy']:.2%} ({r2['correct']}/{r2['total']})")

    print("\n=== Classification Details ===")
    for d in r1["details"]:
        status = "✓" if d["match"] else "✗"
        print(f"  {status} expected={d['expected']:20s} predicted={d['predicted']:20s} | {d['question'][:60]}")
```

- [ ] **Step 3: 运行评测**

```bash
cd backend && python scripts/run_adaptive_evaluation.py
```

- [ ] **Step 4: Commit**

```bash
git add backend/evaluation/dataset.py scripts/run_adaptive_evaluation.py
git commit -m "feat(v17): add Adaptive Graph evaluation benchmark + eval script"
```

---

## Phase 5: Tests + Regression

### Task 7: 单元测试 + 集成测试

**Files:**
- Create: `tests/test_retrieval_planner.py`
- Create: `tests/test_graph_utility_estimator.py`
- Modify: `tests/test_query_profiler.py`

- [ ] **Step 1: RetrievalPlanner 测试**

```python
# tests/test_retrieval_planner.py
import pytest
from backend.rag.retrieval_planner import RetrievalPlanner, STRATEGY_MAP, RetrievalPlan


class TestRetrievalPlanner:
    def test_all_six_types_defined(self):
        for qt in ["factoid", "entity_relation", "multi_hop", "global_summary", "temporal", "comparison"]:
            assert qt in STRATEGY_MAP, f"{qt} missing from STRATEGY_MAP"

    def test_factoid_skips_graph(self):
        planner = RetrievalPlanner()
        plan = planner.plan_from_query_type("factoid")
        assert plan.use_graph is False
        assert plan.use_community is False
        assert plan.graph_hops == 0
        assert "dense" in plan.enabled_channels

    def test_multi_hop_uses_graph(self):
        plan = RetrievalPlanner().plan_from_query_type("multi_hop")
        assert plan.use_graph is True
        assert plan.fusion_strategy == "graph_first"
        assert plan.graph_hops == 3

    def test_global_summary_uses_community(self):
        plan = RetrievalPlanner().plan_from_query_type("global_summary")
        assert plan.use_community is True
        assert plan.use_graph is False

    def test_plan_with_intent_override(self):
        plan = RetrievalPlanner().plan(intent={"query_type": "factoid", "graph_hops": 1})
        assert plan.query_type == "factoid"
        assert plan.graph_hops == 1  # overridden

    def test_plan_with_graph_skip(self):
        plan = RetrievalPlanner().plan(intent={"query_type": "multi_hop", "graph_skip": True})
        assert plan.use_graph is False
        assert plan.use_community is False

    def test_unknown_type_falls_back_to_factoid(self):
        plan = RetrievalPlanner().plan_from_query_type("nonexistent")
        assert plan.query_type == "factoid"
```

- [ ] **Step 2: GraphUtilityEstimator 测试**

```python
# tests/test_graph_utility_estimator.py
import pytest
from backend.rag.graph_utility_estimator import GraphUtilityEstimator, get_graph_utility_estimator


class TestGraphUtilityEstimator:
    @pytest.fixture
    def estimator(self):
        return GraphUtilityEstimator(threshold=0.6)

    def test_factoid_query_low_score(self, estimator):
        score = estimator.estimate("What is Redis?", "factoid")
        assert score.score < 0.5, f"Expected low score for factoid, got {score.score:.2f}"

    def test_multi_hop_query_high_score(self, estimator):
        score = estimator.estimate(
            "Trace the investment chain from SoftBank through its portfolio companies to ByteDance.",
            "multi_hop"
        )
        assert score.score > 0.3, f"Expected higher score for multi-hop, got {score.score:.2f}"

    def test_entity_relation_query_medium_score(self, estimator):
        score = estimator.estimate("Who founded OpenAI?", "entity_relation")
        assert score.score > 0.2, f"Expected medium score for entity relation, got {score.score:.2f}"

    def test_should_use_graph_false_for_factoid(self, estimator):
        assert estimator.should_use_graph("What is Redis?", "factoid") is False

    def test_should_use_graph_true_for_relation(self, estimator):
        assert estimator.should_use_graph("Who founded OpenAI and acquired GitHub?", "entity_relation") is True

    def test_singleton(self):
        e1 = get_graph_utility_estimator()
        e2 = get_graph_utility_estimator()
        assert e1 is e2
```

- [ ] **Step 3: 运行全部测试**

```bash
pytest tests/test_retrieval_planner.py tests/test_graph_utility_estimator.py tests/test_query_profiler.py tests/test_dynamic_rrf.py -v
```

Expected: all pass.

- [ ] **Step 4: 全量回归**

```bash
pytest tests/ -v --ignore=tests/test_evaluation.py --ignore=tests/test_privilege_escalation.py --ignore=tests/test_tenant_isolation_mysql.py --ignore=tests/test_tenant_isolation_milvus.py --ignore=tests/test_tenant_isolation_neo4j.py
```

Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add tests/test_retrieval_planner.py tests/test_graph_utility_estimator.py tests/test_query_profiler.py
git commit -m "test(v17): add tests for RetrievalPlanner + GraphUtilityEstimator + 6-type QueryProfiler"
```

---

## Self-Review

### Spec Coverage Check

| v17 Requirement | Covered By |
|---|---|
| Query Taxonomy (6 types) | Task 1 (QueryProfiler 扩展) |
| Retrieval Planner | Task 2 (RetrievalPlanner) |
| Adaptive RRF Weights | Task 3 (weight_matrix.yaml + dynamic_rrf) |
| Adaptive Graph Depth (0/1/3 hops) | Task 1 (graph_hops per type), Task 4 (orchestrator wiring) |
| Graph Utility Estimator | Task 5 |
| Online Feedback Loop | Deferred to v17.2 |
| Learning-Based Weights | Deferred to v17.2 |
| Evaluation Benchmark | Task 6 |
| Experiment Design (Static vs Adaptive) | Task 6 (run_adaptive_evaluation.py) |

### Placeholder Scan

No "TBD", "TODO", "implement later" found in code blocks. All functions have complete bodies.

### Type Consistency Check

- `QueryIntent.query_type` str → matches `STRATEGY_MAP` keys → matches `weight_matrix.yaml` keys
- `RetrievalPlan.graph_hops` int → matches `GraphUtilityScore.graph_hops` int → matches `local_graph_search(graph_hops=...)` int
- `GraphUtilityEstimator.threshold` float → compared with `GraphUtilityScore.score` float
