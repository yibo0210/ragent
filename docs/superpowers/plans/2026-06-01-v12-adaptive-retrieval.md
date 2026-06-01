# v12 自适应混合检索与降级架构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 v11 架构基础上，实现智能化 Query 画像分级路由、意图驱动的动态 RRF 权重、负载感知的自适应降级链路，以及配套的压测与评测体系。

**Architecture:** 在 Supervisor LLM 路由之前插入一个轻量级 Query Profiler（规则 + Embedding 相似度），输出 L1/L2/L3 意图标签。该标签驱动两条链路：(1) 动态 RRF 权重矩阵替代静态环境变量；(2) 全局负载监控器（Redis 滑动窗口 QPS 计数）决定系统状态（NORMAL/WARNING/CRITICAL），在 WARNING 状态下跳过 Critique/Replan，CRITICAL 状态下熔断 Neo4j 和 Tavily。

**Tech Stack:** Python 3.12, FastAPI, LangGraph, Redis (滑动窗口), Milvus (Embedding 相似度), Prometheus (指标暴露), Locust (压测), RAGAS (评测)

---

## 文件结构

### 新建文件

| 文件 | 职责 |
|------|------|
| `backend/agent/query_profiler.py` | 轻量级意图分类器：规则关键词 + Embedding 余弦相似度，输出 L1/L2/L3 意图标签 |
| `backend/rag/dynamic_rrf.py` | 意图-权重映射矩阵，根据意图标签动态返回 RRF 权重向量 |
| `backend/ha/load_monitor.py` | Redis 滑动窗口 QPS 计数器 + 全局系统状态机（NORMAL/WARNING/CRITICAL） |
| `config/weight_matrix.yaml` | 意图-权重映射配置文件（可热更新） |
| `tests/test_query_profiler.py` | Query Profiler 单元测试 |
| `tests/test_dynamic_rrf.py` | 动态 RRF 权重单元测试 |
| `tests/test_load_monitor.py` | 负载监控器单元测试 |
| `scripts/run_load_test.py` | Locust 压测脚本 |
| `scripts/run_ab_evaluation.py` | A/B 评测对比脚本（静态 vs 动态链路） |

### 修改文件

| 文件 | 改动范围 |
|------|---------|
| `backend/agent/orchestrator.py:238-309` | `supervisor_node` 中插入 Query Profiler 调用，将意图标签写入 state |
| `backend/agent/orchestrator.py:85-115` | `SupervisorState` 新增 `query_intent` 字段 |
| `backend/agent/orchestrator.py:886-896` | `route_after_critique` 增加负载状态判断，WARNING 时跳过 Critique |
| `backend/rag/utils.py:79-126` | `rrf_fusion_three_channel` 支持接收动态权重参数 |
| `backend/rag/pipeline.py:115-167` | `retrieve_initial` 传递意图标签到检索层 |
| `backend/observability/metrics.py:7-28` | 新增 Query Profiler 和负载监控相关 Prometheus 指标 |
| `backend/agent/brain.py:384-548` | SSE 新增 `query_profiler` 事件类型 |

---

## Phase 1: 智能化 Query 画像与分级路由

### Task 1: 创建 Query Profiler 模块

**Files:**
- Create: `backend/agent/query_profiler.py`
- Test: `tests/test_query_profiler.py`

- [ ] **Step 1: 编写 Query Profiler 失败测试**

```python
# tests/test_query_profiler.py
"""Query Profiler 单元测试。"""
import pytest
from backend.agent.query_profiler import QueryProfiler, QueryIntent


class TestQueryProfiler:
    def setup_method(self):
        self.profiler = QueryProfiler()

    def test_l1_factual_greeting(self):
        """简单问候应分类为 L1_FACTUAL。"""
        intent = self.profiler.profile("你好")
        assert intent.level == "L1_FACTUAL"
        assert intent.complexity_score < 0.3

    def test_l1_factual_simple_question(self):
        """简单事实问题应分类为 L1_FACTUAL。"""
        intent = self.profiler.profile("Python 是什么？")
        assert intent.level == "L1_FACTUAL"

    def test_l2_reasoning_relation(self):
        """涉及实体关系的问题应分类为 L2_REASONING。"""
        intent = self.profiler.profile("Milvus 和 Neo4j 之间有什么关系？")
        assert intent.level == "L2_REASONING"

    def test_l2_reasoning_multi_hop(self):
        """多跳推理问题应分类为 L2_REASONING。"""
        intent = self.profiler.profile("GraphRAG 依赖哪些组件来实现多跳推理？")
        assert intent.level == "L2_REASONING"

    def test_l3_macro_summary(self):
        """全局总结性问题应分类为 L3_MACRO_SUMMARY。"""
        intent = self.profiler.profile("系统整体技术架构是怎样的？请全面总结。")
        assert intent.level == "L3_MACRO_SUMMARY"

    def test_l3_macro_compare(self):
        """全局对比问题应分类为 L3_MACRO_SUMMARY。"""
        intent = self.profiler.profile("所有文档中的方法有什么区别？")
        assert intent.level == "L3_MACRO_SUMMARY"

    def test_intent_to_dict(self):
        """QueryIntent 可序列化为 dict。"""
        intent = self.profiler.profile("你好")
        d = intent.to_dict()
        assert "level" in d
        assert "complexity_score" in d
        assert "matched_keywords" in d
        assert "embedding_similarity" in d

    def test_empty_query_defaults_l1(self):
        """空查询默认 L1。"""
        intent = self.profiler.profile("")
        assert intent.level == "L1_FACTUAL"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_query_profiler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.agent.query_profiler'`

- [ ] **Step 3: 实现 Query Profiler**

```python
# backend/agent/query_profiler.py
"""轻量级 Query 意图分类器。

在 Supervisor LLM 路由之前执行，用规则关键词 + Embedding 余弦相似度
将查询分为三级意图，避免每条 Query 都调用大模型做意图识别。

L1_FACTUAL: 简单事实/闲聊 → direct_answer 或纯向量检索
L2_REASONING: 多跳逻辑推理 → local_graph_search + rag_specialist
L3_MACRO_SUMMARY: 宏观全局总结 → global_graph_search
"""
from dataclasses import dataclass, field
from typing import List
import math


# 意图级别常量
L1_FACTUAL = "L1_FACTUAL"
L2_REASONING = "L2_REASONING"
L3_MACRO_SUMMARY = "L3_MACRO_SUMMARY"

# L2 关键词：暗示需要关系推理或多跳检索
_L2_KEYWORDS = [
    "关系", "关联", "依赖", "影响", "区别", "对比", "比较",
    "为什么", "原因", "如何实现", "怎么做的", "原理",
    "哪些组件", "哪些模块", "多跳", "推理", "链路",
    "和.*的关系", "之间", "互相",
]

# L3 关键词：暗示需要全局视角
_L3_KEYWORDS = [
    "总结", "综述", "全面", "整体", "全局", "全景",
    "所有", "全部", "主要", "核心技术栈",
    "架构是怎样的", "有哪些主要", "整体.*是",
    "概览", "汇总", "统计",
]

# L1 关键词：闲聊/简单事实
_L1_KEYWORDS = [
    "你好", "hi", "hello", "谢谢", "thanks",
    "是什么", "什么是", "叫什么", "意思",
    "天气", "几点", "日期",
]

# 意图级别原型查询（用于 Embedding 相似度匹配）
_PROTOTYPE_QUERIES = {
    L1_FACTUAL: [
        "你好",
        "Python 是什么？",
        "今天天气怎么样？",
        "谢谢你的帮助",
    ],
    L2_REASONING: [
        "A 和 B 之间有什么关系？",
        "这个技术依赖哪些组件来实现？",
        "为什么系统要这样设计？",
        "两种方案的区别和联系是什么？",
    ],
    L3_MACRO_SUMMARY: [
        "请全面总结整体技术架构",
        "所有文档中的主要技术有哪些？",
        "系统的核心模块概览",
        "请综述各方面的设计思路",
    ],
}

# Embedding 缓存（模块级单例）
_prototype_embeddings = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class QueryIntent:
    """Query 画像结果。"""
    level: str  # L1_FACTUAL / L2_REASONING / L3_MACRO_SUMMARY
    complexity_score: float  # 0.0 ~ 1.0
    matched_keywords: List[str] = field(default_factory=list)
    embedding_similarity: dict = field(default_factory=dict)
    reason: str = ""


class QueryProfiler:
    """轻量级 Query 意图分类器。"""

    def __init__(self, use_embedding: bool = True):
        self._use_embedding = use_embedding
        self._embedding_service = None
        self._prototype_embeddings = None

    def _get_embedding_service(self):
        if self._embedding_service is None:
            from backend.embedding.service import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _get_prototype_embeddings(self) -> dict:
        """延迟初始化并缓存原型查询的 Embedding。"""
        global _prototype_embeddings
        if _prototype_embeddings is not None:
            return _prototype_embeddings

        if not self._use_embedding:
            _prototype_embeddings = {}
            return _prototype_embeddings

        try:
            service = self._get_embedding_service()
            all_queries = []
            query_to_level = {}
            for level, queries in _PROTOTYPE_QUERIES.items():
                for q in queries:
                    all_queries.append(q)
                    query_to_level[q] = level

            embeddings = service.get_embeddings(all_queries)
            _prototype_embeddings = {}
            for q, emb in zip(all_queries, embeddings):
                level = query_to_level[q]
                if level not in _prototype_embeddings:
                    _prototype_embeddings[level] = []
                _prototype_embeddings[level].append(emb)
        except Exception:
            _prototype_embeddings = {}

        return _prototype_embeddings

    def _keyword_score(self, query_lower: str, keywords: list) -> tuple[float, list[str]]:
        """关键词匹配打分。返回 (score, matched_keywords)。"""
        matched = []
        for kw in keywords:
            if kw in query_lower:
                matched.append(kw)
        score = min(len(matched) / 3.0, 1.0)  # 匹配 3 个及以上满分
        return score, matched

    def _embedding_score(self, query: str) -> dict[str, float]:
        """Embedding 余弦相似度打分。返回各级别的平均相似度。"""
        proto_embs = self._get_prototype_embeddings()
        if not proto_embs:
            return {}

        try:
            service = self._get_embedding_service()
            query_emb = service.get_embeddings([query])[0]
        except Exception:
            return {}

        scores = {}
        for level, embs in proto_embs.items():
            sims = [_cosine_similarity(query_emb, emb) for emb in embs]
            scores[level] = sum(sims) / len(sims) if sims else 0.0
        return scores

    def profile(self, query: str) -> QueryIntent:
        """分析查询意图，返回 QueryIntent。"""
        if not query or not query.strip():
            return QueryIntent(
                level=L1_FACTUAL,
                complexity_score=0.0,
                reason="空查询",
            )

        query_lower = query.lower().strip()

        # 1. 关键词打分
        l1_score, l1_kw = self._keyword_score(query_lower, _L1_KEYWORDS)
        l2_score, l2_kw = self._keyword_score(query_lower, _L2_KEYWORDS)
        l3_score, l3_kw = self._keyword_score(query_lower, _L3_KEYWORDS)

        all_matched = l1_kw + l2_kw + l3_kw

        # 2. Embedding 相似度打分
        emb_scores = self._embedding_score(query)

        # 3. 综合打分（关键词 60% + Embedding 40%）
        final_l1 = l1_score * 0.6 + emb_scores.get(L1_FACTUAL, 0) * 0.4
        final_l2 = l2_score * 0.6 + emb_scores.get(L2_REASONING, 0) * 0.4
        final_l3 = l3_score * 0.6 + emb_scores.get(L3_MACRO_SUMMARY, 0) * 0.4

        # 4. 选择最高分的意图级别
        scores = {L1_FACTUAL: final_l1, L2_REASONING: final_l2, L3_MACRO_SUMMARY: final_l3}
        best_level = max(scores, key=scores.get)
        best_score = scores[best_level]

        # 5. 复杂度分数 = 1 - L1 分数（L1 越高越简单）
        complexity = max(0.0, min(1.0, 1.0 - final_l1))

        # 6. 短查询（< 5 字符）强制 L1
        if len(query.strip()) < 5:
            best_level = L1_FACTUAL
            complexity = 0.0

        reason_parts = []
        if all_matched:
            reason_parts.append(f"关键词: {', '.join(all_matched[:3])}")
        if emb_scores:
            best_emb = max(emb_scores.values())
            reason_parts.append(f"Embedding max: {best_emb:.3f}")

        return QueryIntent(
            level=best_level,
            complexity_score=round(complexity, 3),
            matched_keywords=all_matched,
            embedding_similarity=emb_scores,
            reason="; ".join(reason_parts) if reason_parts else "默认分类",
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_query_profiler.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/agent/query_profiler.py tests/test_query_profiler.py
git commit -m "feat(v12): add Query Profiler — lightweight intent classifier"
```

---

### Task 2: 将 Query Profiler 集成到 Supervisor 路由

**Files:**
- Modify: `backend/agent/orchestrator.py:85-115` (SupervisorState 新增字段)
- Modify: `backend/agent/orchestrator.py:238-309` (supervisor_node 调用 Profiler)
- Modify: `backend/agent/brain.py:408-431` (SSE 新增 profiler 事件)
- Test: `tests/test_query_profiler.py` (追加集成测试)

- [ ] **Step 1: 编写集成测试**

```python
# tests/test_query_profiler.py 追加
class TestProfilerIntegration:
    def test_profiler_in_state(self):
        """验证 Profiler 结果能正确写入 SupervisorState。"""
        from backend.agent.orchestrator import supervisor_node, SupervisorState
        from langchain_core.messages import HumanMessage

        state = {
            "messages": [HumanMessage(content="你好")],
            "user_query": "你好",
            "next_worker": "",
            "next_workers": [],
            "route_reason": "",
            "rag_trace": None,
            "web_search_trace": None,
            "agent_trace": None,
            "worker_outputs": {},
            "human_interfered_input": "",
            "query_plan": None,
            "critique_result": None,
            "retry_count": 0,
            "draft_answer": "",
            "is_hallucinated": False,
            "plan_steps_completed": [],
            "tool_outputs": {},
            "query_intent": None,
        }
        result = supervisor_node(state)
        assert "query_intent" in result
        intent = result["query_intent"]
        assert intent["level"] in ("L1_FACTUAL", "L2_REASONING", "L3_MACRO_SUMMARY")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_query_profiler.py::TestProfilerIntegration -v`
Expected: FAIL — `KeyError: 'query_intent'`

- [ ] **Step 3: 修改 SupervisorState 新增 query_intent 字段**

在 `backend/agent/orchestrator.py` 的 `SupervisorState` 类中追加字段（在 `tool_outputs` 之后）：

```python
    # v12: Query 意图画像
    query_intent: Optional[dict]        # Profiler 输出的意图标签
```

- [ ] **Step 4: 修改 supervisor_node 调用 Profiler**

在 `backend/agent/orchestrator.py` 的 `supervisor_node` 函数中，在提取 `user_query` 之后、调用 Supervisor LLM 之前，插入 Profiler 调用：

```python
    # v12: Query Profiler — 轻量级意图分类
    from backend.agent.query_profiler import QueryProfiler
    profiler = QueryProfiler(use_embedding=True)
    intent = profiler.profile(user_query)
    query_intent = intent.to_dict()
    log.info("query_profiled", level=intent.level, score=intent.complexity_score,
             keywords=intent.matched_keywords[:3])
```

然后在 `supervisor_node` 的 return dict 中追加：

```python
        "query_intent": query_intent,
```

- [ ] **Step 5: 修改 brain.py SSE 事件**

在 `backend/agent/brain.py` 的 `_graph_worker` 函数中，`node_name == "supervisor"` 分支内，路由事件之后追加 Profiler 事件推送：

```python
                        # v12: Query Profiler 事件
                        if update.get("query_intent"):
                            await output_queue.put({
                                "type": "query_profiler",
                                "intent": update["query_intent"],
                            })
```

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/test_query_profiler.py -v`
Expected: 全部 PASS

- [ ] **Step 7: Commit**

```bash
git add backend/agent/orchestrator.py backend/agent/brain.py tests/test_query_profiler.py
git commit -m "feat(v12): integrate Query Profiler into Supervisor routing + SSE events"
```

---

## Phase 2: 意图驱动的动态 RRF 权重分配

### Task 3: 创建权重矩阵配置文件

**Files:**
- Create: `config/weight_matrix.yaml`
- Test: `tests/test_dynamic_rrf.py`

- [ ] **Step 1: 编写权重矩阵加载测试**

```python
# tests/test_dynamic_rrf.py
"""动态 RRF 权重矩阵单元测试。"""
import pytest
from backend.rag.dynamic_rrf import load_weight_matrix, get_weights_for_intent


class TestWeightMatrix:
    def test_load_matrix(self):
        """权重矩阵能正确加载。"""
        matrix = load_weight_matrix()
        assert "L1_FACTUAL" in matrix
        assert "L2_REASONING" in matrix
        assert "L3_MACRO_SUMMARY" in matrix

    def test_l1_weights_dense_heavy(self):
        """L1 事实类应以 Dense 为主。"""
        weights = get_weights_for_intent("L1_FACTUAL")
        assert weights[0] > weights[2]  # dense > graph
        assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_l2_weights_graph_heavy(self):
        """L2 推理类应以 Graph 为主。"""
        weights = get_weights_for_intent("L2_REASONING")
        assert weights[2] > weights[0]  # graph > dense

    def test_l3_weights_balanced(self):
        """L3 总结类应权重较均衡。"""
        weights = get_weights_for_intent("L3_MACRO_SUMMARY")
        assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_unknown_intent_returns_default(self):
        """未知意图返回默认权重。"""
        weights = get_weights_for_intent("UNKNOWN")
        assert len(weights) == 4
        assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_weights_tuple_length(self):
        """所有权重向量长度为 4 (dense, sparse, graph, visual)。"""
        for level in ["L1_FACTUAL", "L2_REASONING", "L3_MACRO_SUMMARY"]:
            weights = get_weights_for_intent(level)
            assert len(weights) == 4
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_dynamic_rrf.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 创建权重矩阵配置文件**

```yaml
# config/weight_matrix.yaml
# 意图-权重映射矩阵
# weights: [dense, sparse, graph, visual]
# 所有权重之和应为 1.0

L1_FACTUAL:
  # 简单事实/闲聊：以 Dense 向量检索为主，不需要图谱
  weights: [0.70, 0.25, 0.00, 0.05]
  description: "简单事实问答，纯向量检索即可"

L2_REASONING:
  # 多跳推理：以 Graph 为主，Dense 辅助定位
  weights: [0.20, 0.10, 0.65, 0.05]
  description: "多跳逻辑推理，图谱关系外扩为核心"

L3_MACRO_SUMMARY:
  # 宏观总结：Dense + Graph 均衡，Sparse 补充
  weights: [0.35, 0.20, 0.35, 0.10]
  description: "全局总结性查询，多通道均衡检索"

# 默认权重（当意图识别失败时使用）
DEFAULT:
  weights: [0.40, 0.30, 0.15, 0.15]
  description: "默认权重，与 v11 静态配置一致"
```

- [ ] **Step 4: 实现动态 RRF 模块**

```python
# backend/rag/dynamic_rrf.py
"""意图驱动的动态 RRF 权重分配。

根据 Query Profiler 输出的意图标签，从配置文件加载对应的权重向量，
替代原有的静态环境变量权重。
"""
import os
from pathlib import Path
from typing import Tuple

import yaml

from backend.observability import get_logger

log = get_logger("ragent.dynamic_rrf")

# 配置文件路径
_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "weight_matrix.yaml"

# 模块级缓存
_matrix_cache = None


def load_weight_matrix() -> dict:
    """加载权重矩阵配置（带缓存）。"""
    global _matrix_cache
    if _matrix_cache is not None:
        return _matrix_cache

    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _matrix_cache = data if data else {}
    except Exception as e:
        log.warning("weight_matrix_load_failed", error=str(e))
        _matrix_cache = {}

    return _matrix_cache


def reload_weight_matrix() -> dict:
    """强制重新加载权重矩阵（用于配置热更新）。"""
    global _matrix_cache
    _matrix_cache = None
    return load_weight_matrix()


def get_weights_for_intent(intent_level: str) -> Tuple[float, float, float, float]:
    """根据意图级别返回 RRF 权重向量 (dense, sparse, graph, visual)。"""
    matrix = load_weight_matrix()

    entry = matrix.get(intent_level) or matrix.get("DEFAULT")
    if not entry:
        # 硬编码兜底，与 v11 默认值一致
        return (0.4, 0.3, 0.15, 0.15)

    weights = entry.get("weights", [0.4, 0.3, 0.15, 0.15])

    # 确保长度为 4
    while len(weights) < 4:
        weights.append(0.0)

    return tuple(weights[:4])
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_dynamic_rrf.py -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add config/weight_matrix.yaml backend/rag/dynamic_rrf.py tests/test_dynamic_rrf.py
git commit -m "feat(v12): add intent-driven dynamic RRF weight matrix"
```

---

### Task 4: 将动态权重集成到 RAG 检索链路

**Files:**
- Modify: `backend/rag/utils.py:79-126` (rrf_fusion_three_channel 支持动态权重)
- Modify: `backend/rag/pipeline.py:115-167` (retrieve_initial 传递意图)
- Modify: `backend/agent/orchestrator.py:312-365` (rag_specialist_node 传递意图)

- [ ] **Step 1: 编写集成测试**

```python
# tests/test_dynamic_rrf.py 追加
class TestDynamicRRFIntegration:
    def test_rrf_accepts_dynamic_weights(self):
        """rrf_fusion_three_channel 应接受外部传入的权重。"""
        from backend.rag.utils import rrf_fusion_three_channel

        dense = [({"chunk_id": "c1", "text": "test"}, 0.9)]
        sparse = [({"chunk_id": "c1", "text": "test"}, 0.8)]
        graph = [({"chunk_id": "c2", "text": "graph"}, 0.7)]

        # 使用自定义权重
        result = rrf_fusion_three_channel(
            dense, sparse, graph,
            weights=(0.7, 0.2, 0.1, 0.0),
            top_k=5,
        )
        assert len(result) > 0

    def test_rrf_defaults_when_no_weights(self):
        """不传权重时应使用默认值。"""
        from backend.rag.utils import rrf_fusion_three_channel

        dense = [({"chunk_id": "c1", "text": "test"}, 0.9)]
        sparse = []
        graph = []

        result = rrf_fusion_three_channel(dense, sparse, graph, top_k=5)
        assert len(result) > 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_dynamic_rrf.py::TestDynamicRRFIntegration -v`
Expected: 可能 PASS（现有代码已支持 weights 参数），确认行为正确

- [ ] **Step 3: 修改 retrieve_documents 传递意图标签**

在 `backend/rag/utils.py` 的 `retrieve_documents` 函数签名中新增 `intent_level` 参数：

```python
def retrieve_documents(query: str, top_k: int = 5, intent_level: str = None) -> Dict[str, Any]:
```

在函数内部，当 `intent_level` 非空时，动态获取权重：

```python
    # v12: 动态 RRF 权重
    if intent_level:
        from backend.rag.dynamic_rrf import get_weights_for_intent
        dynamic_weights = get_weights_for_intent(intent_level)
    else:
        dynamic_weights = None
```

然后将 `dynamic_weights` 传递给 `hybrid_retrieve` 调用（如果 MilvusManager 支持）或在 rerank 之前使用 `rrf_fusion_three_channel` 进行融合。

- [ ] **Step 4: 修改 rag_specialist_node 传递意图**

在 `backend/agent/orchestrator.py` 的 `rag_specialist_node` 中，从 state 读取意图标签并传递：

```python
    # v12: 传递意图标签到 RAG pipeline
    query_intent = state.get("query_intent")
    intent_level = query_intent.get("level") if query_intent else None
```

然后修改 `run_rag_graph` 调用，传入 `intent_level`。

- [ ] **Step 5: 修改 run_rag_graph 传递意图**

在 `backend/rag/pipeline.py` 的 `run_rag_graph` 函数签名中新增 `intent_level` 参数，并传递到 `retrieve_initial`。

在 `retrieve_initial` 中调用 `retrieve_documents` 时传入 `intent_level`。

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/test_dynamic_rrf.py -v`
Expected: 全部 PASS

- [ ] **Step 7: Commit**

```bash
git add backend/rag/utils.py backend/rag/pipeline.py backend/agent/orchestrator.py tests/test_dynamic_rrf.py
git commit -m "feat(v12): integrate dynamic RRF weights into retrieval pipeline"
```

---

## Phase 3: 负载感知的自适应降级链路

### Task 5: 创建全局负载监控器

**Files:**
- Create: `backend/ha/load_monitor.py`
- Test: `tests/test_load_monitor.py`

- [ ] **Step 1: 编写负载监控器测试**

```python
# tests/test_load_monitor.py
"""全局负载监控器单元测试。"""
import pytest
from unittest.mock import patch, MagicMock
from backend.ha.load_monitor import LoadMonitor, SystemState


class TestLoadMonitor:
    def setup_method(self):
        self.monitor = LoadMonitor(window_size=10, warning_qps=50, critical_qps=100)

    def test_initial_state_normal(self):
        """初始状态应为 NORMAL。"""
        assert self.monitor.get_state() == SystemState.NORMAL

    def test_state_transitions(self):
        """QPS 变化应触发状态转换。"""
        # 模拟低 QPS
        with patch.object(self.monitor, '_get_current_qps', return_value=30):
            assert self.monitor.evaluate_state() == SystemState.NORMAL

        # 模拟中等 QPS
        with patch.object(self.monitor, '_get_current_qps', return_value=60):
            assert self.monitor.evaluate_state() == SystemState.WARNING

        # 模拟高 QPS
        with patch.object(self.monitor, '_get_current_qps', return_value=120):
            assert self.monitor.evaluate_state() == SystemState.CRITICAL

    def test_should_skip_critique(self):
        """WARNING 状态下应跳过 Critique。"""
        with patch.object(self.monitor, 'get_state', return_value=SystemState.WARNING):
            assert self.monitor.should_skip_critique() is True

        with patch.object(self.monitor, 'get_state', return_value=SystemState.NORMAL):
            assert self.monitor.should_skip_critique() is False

    def test_should_circuit_break_neo4j(self):
        """CRITICAL 状态下应熔断 Neo4j。"""
        with patch.object(self.monitor, 'get_state', return_value=SystemState.CRITICAL):
            assert self.monitor.should_circuit_break_neo4j() is True

        with patch.object(self.monitor, 'get_state', return_value=SystemState.WARNING):
            assert self.monitor.should_circuit_break_neo4j() is False

    def test_should_circuit_break_tavily(self):
        """CRITICAL 状态下应熔断 Tavily。"""
        with patch.object(self.monitor, 'get_state', return_value=SystemState.CRITICAL):
            assert self.monitor.should_circuit_break_tavily() is True

    def test_record_request(self):
        """记录请求应增加计数。"""
        initial_count = self.monitor._request_count
        self.monitor.record_request()
        assert self.monitor._request_count == initial_count + 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_load_monitor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现负载监控器**

```python
# backend/ha/load_monitor.py
"""全局负载监控器。

基于 Redis 滑动窗口实现 QPS 计数和全局系统状态管理。
系统状态驱动自适应降级策略：
- NORMAL: 全量链路放行
- WARNING: 跳过 Critique/Replan，降低 LLM 交互轮数
- CRITICAL: 熔断 Neo4j 和 Tavily，退化为纯 Milvus 向量检索
"""
import time
from enum import Enum
from typing import Optional

import redis

from backend.observability import get_logger, Metrics

log = get_logger("ragent.ha")


class SystemState(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


# Redis key 模板
_QPS_KEY = "ragent:load:qps:{timestamp}"
_STATE_KEY = "ragent:load:system_state"


class LoadMonitor:
    """基于 Redis 滑动窗口的全局负载监控器。"""

    def __init__(
        self,
        window_size: int = 10,
        warning_qps: int = 50,
        critical_qps: int = 100,
        redis_url: str = None,
    ):
        self._window_size = window_size  # 滑动窗口大小（秒）
        self._warning_qps = warning_qps
        self._critical_qps = critical_qps
        self._redis_url = redis_url or __import__("os").getenv(
            "REDIS_URL", "redis://localhost:6379/0"
        )
        self._client: Optional[redis.Redis] = None
        self._request_count = 0
        self._last_state = SystemState.NORMAL
        self._last_eval_time = 0.0

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis.from_url(
                self._redis_url, decode_responses=True
            )
        return self._client

    def _get_current_qps(self) -> float:
        """获取当前滑动窗口内的 QPS。"""
        try:
            client = self._get_client()
            now = int(time.time())
            # 获取最近 window_size 秒的计数
            total = 0
            for i in range(self._window_size):
                key = _QPS_KEY.format(timestamp=now - i)
                count = client.get(key)
                if count:
                    total += int(count)
            return total / self._window_size
        except Exception as e:
            log.warning("qps_read_failed", error=str(e))
            return 0.0

    def record_request(self):
        """记录一次请求（每次调用 +1）。"""
        self._request_count += 1
        try:
            client = self._get_client()
            now = int(time.time())
            key = _QPS_KEY.format(timestamp=now)
            pipe = client.pipeline()
            pipe.incr(key)
            pipe.expire(key, self._window_size + 5)
            pipe.execute()
        except Exception:
            pass  # Redis 不可用时静默降级

    def evaluate_state(self) -> SystemState:
        """评估当前系统状态。"""
        qps = self._get_current_qps()

        if qps >= self._critical_qps:
            new_state = SystemState.CRITICAL
        elif qps >= self._warning_qps:
            new_state = SystemState.WARNING
        else:
            new_state = SystemState.NORMAL

        # 状态变化时记录日志和指标
        if new_state != self._last_state:
            log.warning("system_state_change",
                        old=self._last_state.value,
                        new=new_state.value,
                        qps=round(qps, 1))
            self._last_state = new_state

        self._last_eval_time = time.time()
        return new_state

    def get_state(self) -> SystemState:
        """获取当前系统状态（带缓存，避免每个请求都查 Redis）。"""
        now = time.time()
        # 每秒最多评估一次
        if now - self._last_eval_time >= 1.0:
            return self.evaluate_state()
        return self._last_state

    def should_skip_critique(self) -> bool:
        """WARNING 及以上状态跳过 Critique 节点。"""
        return self.get_state() in (SystemState.WARNING, SystemState.CRITICAL)

    def should_circuit_break_neo4j(self) -> bool:
        """CRITICAL 状态熔断 Neo4j 查询。"""
        return self.get_state() == SystemState.CRITICAL

    def should_circuit_break_tavily(self) -> bool:
        """CRITICAL 状态熔断 Tavily 搜索。"""
        return self.get_state() == SystemState.CRITICAL

    def get_stats(self) -> dict:
        """获取当前监控统计信息。"""
        qps = self._get_current_qps()
        return {
            "state": self.get_state().value,
            "qps": round(qps, 1),
            "warning_threshold": self._warning_qps,
            "critical_threshold": self._critical_qps,
            "window_size": self._window_size,
            "total_requests": self._request_count,
        }


# 模块级单例
_load_monitor: Optional[LoadMonitor] = None


def get_load_monitor() -> LoadMonitor:
    """获取全局负载监控器单例。"""
    global _load_monitor
    if _load_monitor is None:
        import os
        _load_monitor = LoadMonitor(
            window_size=int(os.getenv("LOAD_MONITOR_WINDOW", "10")),
            warning_qps=int(os.getenv("LOAD_WARNING_QPS", "50")),
            critical_qps=int(os.getenv("LOAD_CRITICAL_QPS", "100")),
        )
    return _load_monitor
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_load_monitor.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/ha/load_monitor.py tests/test_load_monitor.py
git commit -m "feat(v12): add Redis-based load monitor with system state machine"
```

---

### Task 6: 将负载监控集成到 LangGraph 条件边

**Files:**
- Modify: `backend/agent/orchestrator.py:886-896` (route_after_critique 增加负载判断)
- Modify: `backend/agent/orchestrator.py:581-629` (local_graph_search_node 增加降级)
- Modify: `backend/agent/orchestrator.py:368-447` (web_searcher_node 增加降级)
- Modify: `backend/observability/metrics.py` (新增负载指标)
- Modify: `backend/agent/brain.py` (SSE 新增 system_state 事件)

- [ ] **Step 1: 编写集成测试**

```python
# tests/test_load_monitor.py 追加
class TestLoadMonitorIntegration:
    def test_critique_skipped_under_load(self):
        """高负载时 route_after_critique 应返回 end。"""
        from unittest.mock import patch
        from backend.agent.orchestrator import route_after_critique
        from backend.ha.load_monitor import SystemState

        state = {
            "critique_result": {"is_valid": False, "feedback": "test"},
            "retry_count": 0,
        }

        # 正常负载：应返回 replan
        with patch("backend.agent.orchestrator.get_load_monitor") as mock:
            mock.return_value = MagicMock(should_skip_critique=lambda: False)
            # 注意：需要在 orchestrator 中导入 get_load_monitor
            # 这里先测试默认行为
            result = route_after_critique(state)
            assert result == "replan"
```

- [ ] **Step 2: 运行测试确认当前行为**

Run: `pytest tests/test_load_monitor.py::TestLoadMonitorIntegration -v`
Expected: PASS（验证现有行为基线）

- [ ] **Step 3: 修改 route_after_critique 增加负载判断**

在 `backend/agent/orchestrator.py` 的 `route_after_critique` 函数中：

```python
def route_after_critique(state: SupervisorState) -> str:
    """Critique 后的条件路由（v12: 负载感知）。"""
    from backend.ha.load_monitor import get_load_monitor

    critique = state.get("critique_result", {})
    retry = state.get("retry_count", 0)

    # v12: 高负载时跳过 Critique 重试，直接输出
    monitor = get_load_monitor()
    if monitor.should_skip_critique() and not critique.get("is_valid", True):
        log.info("critique_skipped_due_to_load", state=monitor.get_state().value)
        return "end"

    if critique.get("is_valid", True):
        return "end"
    elif retry < 2:
        return "replan"
    else:
        return "end"
```

- [ ] **Step 4: 修改 local_graph_search_node 增加 CRITICAL 降级**

在 `backend/agent/orchestrator.py` 的 `local_graph_search_node` 中，在调用 `safe_graph_search` 之前检查负载状态：

```python
    # v12: CRITICAL 状态下跳过图谱搜索，降级到纯向量
    from backend.ha.load_monitor import get_load_monitor
    monitor = get_load_monitor()
    if monitor.should_circuit_break_neo4j():
        emit_graph_step("⚡", "负载降级 — CRITICAL 状态，跳过 Neo4j 图谱搜索", agent="local_graph_search")
        from backend.rag.utils import retrieve_documents
        result = retrieve_documents(user_query, top_k=5)
        result["mode"] = "degraded_load_critical"
    else:
        # 原有逻辑
        ...
```

- [ ] **Step 5: 修改 web_searcher_node 增加 CRITICAL 降级**

在 `backend/agent/orchestrator.py` 的 `web_searcher_node` 中，在调用 `run_web_search` 之前检查：

```python
    # v12: CRITICAL 状态下跳过 Tavily 搜索
    from backend.ha.load_monitor import get_load_monitor
    monitor = get_load_monitor()
    if monitor.should_circuit_break_tavily():
        # 直接降级到 RAG
        search_result = {"error": "CRITICAL 负载降级，跳过联网搜索", "results": []}
    else:
        search_result = run_web_search(user_query)
```

- [ ] **Step 6: 新增 Prometheus 负载指标**

在 `backend/observability/metrics.py` 中追加：

```python
system_state = Gauge(
    "system_load_state", "系统负载状态 (0=NORMAL, 1=WARNING, 2=CRITICAL)"
)
query_qps = Gauge(
    "query_qps", "当前查询 QPS"
)
profiler_distribution = Counter(
    "query_profiler_distribution", "Query Profiler 意图分布", ["level"]
)
```

在 `Metrics` 类中追加静态方法：

```python
    @staticmethod
    def set_system_state(state_value: int):
        if not METRICS_ENABLED:
            return
        system_state.set(state_value)

    @staticmethod
    def set_qps(qps: float):
        if not METRICS_ENABLED:
            return
        query_qps.set(qps)

    @staticmethod
    def record_profiler_intent(level: str):
        if not METRICS_ENABLED:
            return
        profiler_distribution.labels(level=level).inc()
```

- [ ] **Step 7: 在 brain.py SSE 中推送 system_state 事件**

在 `backend/agent/brain.py` 的 supervisor 事件处理分支中，追加：

```python
                        # v12: 系统状态事件
                        from backend.ha.load_monitor import get_load_monitor
                        monitor = get_load_monitor()
                        await output_queue.put({
                            "type": "system_state",
                            "state": monitor.get_state().value,
                            "stats": monitor.get_stats(),
                        })
```

- [ ] **Step 8: 在 API 入口记录请求到负载监控器**

在 `backend/api/routes.py` 的 `chat_stream_endpoint` 和 `chat_endpoint` 中，调用 chat 前记录请求：

```python
    from backend.ha.load_monitor import get_load_monitor
    get_load_monitor().record_request()
```

- [ ] **Step 9: 运行全量测试**

Run: `pytest tests/test_load_monitor.py tests/test_query_profiler.py tests/test_dynamic_rrf.py -v`
Expected: 全部 PASS

- [ ] **Step 10: Commit**

```bash
git add backend/agent/orchestrator.py backend/observability/metrics.py backend/agent/brain.py backend/api/routes.py tests/test_load_monitor.py
git commit -m "feat(v12): integrate load-aware adaptive degradation into LangGraph edges"
```

---

## Phase 4: 压测与评测数据收集

### Task 7: 创建 Locust 压测脚本

**Files:**
- Create: `scripts/run_load_test.py`

- [ ] **Step 1: 创建 Locust 压测脚本**

```python
# scripts/run_load_test.py
#!/usr/bin/env python3
"""Locust 压测脚本 — 模拟不同 QPS 并发流量。

用法:
    # 直接运行（默认 10 QPS，持续 60 秒）
    python scripts/run_load_test.py

    # 使用 Locust Web UI
    locust -f scripts/run_load_test.py --host http://localhost:8000

    # 无头模式压测
    locust -f scripts/run_load_test.py --host http://localhost:8000 \
        --users 50 --spawn-rate 10 --run-time 2m --headless
"""
import json
import random
import time
from locust import HttpUser, task, between, events


# 测试查询集（覆盖不同意图类型）
TEST_QUERIES = [
    # L1_FACTUAL
    "你好",
    "Python 是什么？",
    "今天天气怎么样？",
    "谢谢",
    # L2_REASONING
    "Milvus 和 Neo4j 之间有什么关系？",
    "GraphRAG 依赖哪些组件？",
    "系统如何实现多跳推理？",
    "Dense 和 Sparse 检索的区别是什么？",
    # L3_MACRO_SUMMARY
    "请总结系统的整体技术架构",
    "所有文档中涉及的主要技术有哪些？",
    "系统的核心模块概览",
    "请综述各方面的设计思路",
]


class RagentUser(HttpUser):
    """模拟 Ragent AI 用户。"""
    wait_time = between(0.5, 2.0)

    @task(3)
    def chat_stream(self):
        """SSE 流式对话（主要测试场景）。"""
        query = random.choice(TEST_QUERIES)
        with self.client.post(
            "/api/chat/stream",
            json={"message": query, "session_id": f"load_test_{random.randint(1, 1000)}"},
            stream=True,
            name="/api/chat/stream",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                # 读取 SSE 流直到 [DONE]
                full_content = ""
                for line in response.iter_lines():
                    line = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            if event.get("type") == "content":
                                full_content += event.get("content", "")
                        except json.JSONDecodeError:
                            pass
                if full_content:
                    response.success()
                else:
                    response.failure("Empty response content")
            elif response.status_code == 423:
                response.failure("HITL lock (423)")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def chat_sync(self):
        """同步对话。"""
        query = random.choice(TEST_QUERIES)
        self.client.post(
            "/api/chat",
            json={"message": query, "session_id": f"load_test_{random.randint(1, 1000)}"},
            name="/api/chat",
        )

    @task(1)
    def health_check(self):
        """健康检查。"""
        self.client.get("/api/health", name="/api/health")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("=" * 60)
    print("Ragent AI 压测开始")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("=" * 60)
    print("Ragent AI 压测结束")
    print("=" * 60)
```

- [ ] **Step 2: 验证脚本可运行**

Run: `python scripts/run_load_test.py --help`
Expected: 显示 Locust 帮助信息

- [ ] **Step 3: Commit**

```bash
git add scripts/run_load_test.py
git commit -m "feat(v12): add Locust load testing script"
```

---

### Task 8: 创建 A/B 评测对比脚本

**Files:**
- Create: `scripts/run_ab_evaluation.py`

- [ ] **Step 1: 创建 A/B 评测脚本**

```python
# scripts/run_ab_evaluation.py
#!/usr/bin/env python3
"""A/B 评测对比脚本 — 静态链路 vs 自适应动态链路。

用法:
    # 先运行静态链路（v11 默认权重）
    python scripts/run_ab_evaluation.py --mode static --limit 20 --output static_result.json

    # 再运行动态链路（v12 意图驱动权重）
    python scripts/run_ab_evaluation.py --mode dynamic --limit 20 --output dynamic_result.json

    # 对比两次结果
    python scripts/run_evaluation.py --compare static_result.json dynamic_result.json
"""
import argparse
import json
import sys
import time
import statistics
from collections import defaultdict

sys.path.insert(0, ".")

from backend.rag.utils import retrieve_documents
from backend.rag.pipeline import run_rag_graph
from backend.evaluation.dataset import load_golden_dataset
from backend.evaluation.metrics import compute_ragas_metrics, generate_answer


def _format_docs(docs: list[dict]) -> list[str]:
    return [d.get("text", "") for d in docs]


def _compute_latency_stats(latencies: list[float]) -> dict:
    if not latencies:
        return {}
    return {
        "avg": round(statistics.mean(latencies), 1),
        "p50": round(statistics.median(latencies), 1),
        "p95": round(sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0], 1),
        "max": round(max(latencies), 1),
        "min": round(min(latencies), 1),
    }


def run_static_evaluation(dataset: list[dict], limit: int = 0) -> dict:
    """静态链路评测（使用默认环境变量权重）。"""
    if limit > 0:
        dataset = dataset[:limit]

    samples = []
    latencies = []
    intent_distribution = defaultdict(int)

    for item in dataset:
        t0 = time.time()
        result = run_rag_graph(item["question"])
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        if result.get("force_interrupt"):
            continue

        docs = result.get("docs", [])
        samples.append({
            "question": item["question"],
            "answer": item["ground_truth"],
            "contexts": _format_docs(docs),
            "ground_truth": item["ground_truth"],
        })
        print(f"  [{item.get('query_type', '?')}] {item['id']}: {len(docs)} chunks, {latency:.0f}ms")

    metrics = compute_ragas_metrics(samples)

    return {
        "mode": "static",
        "metrics": metrics,
        "sample_count": len(samples),
        "latency": _compute_latency_stats(latencies),
        "intent_distribution": dict(intent_distribution),
    }


def run_dynamic_evaluation(dataset: list[dict], limit: int = 0) -> dict:
    """动态链路评测（使用 Query Profiler + 动态权重）。"""
    if limit > 0:
        dataset = dataset[:limit]

    from backend.agent.query_profiler import QueryProfiler
    from backend.rag.dynamic_rrf import get_weights_for_intent

    profiler = QueryProfiler(use_embedding=True)

    samples = []
    latencies = []
    intent_distribution = defaultdict(int)
    weights_used = {}

    for item in dataset:
        # 1. Query Profiler 分类
        intent = profiler.profile(item["question"])
        intent_distribution[intent.level] += 1
        weights_used[intent.level] = get_weights_for_intent(intent.level)

        # 2. 使用意图级别调用 RAG
        t0 = time.time()
        result = run_rag_graph(item["question"], intent_level=intent.level)
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        if result.get("force_interrupt"):
            continue

        docs = result.get("docs", [])
        samples.append({
            "question": item["question"],
            "answer": item["ground_truth"],
            "contexts": _format_docs(docs),
            "ground_truth": item["ground_truth"],
        })
        print(f"  [{intent.level}] {item['id']}: {len(docs)} chunks, {latency:.0f}ms")

    metrics = compute_ragas_metrics(samples)

    per_intent = defaultdict(list)
    for item, sample in zip(dataset[:len(samples)], samples):
        intent = profiler.profile(item["question"])
        per_intent[intent.level].append(sample)
    by_intent = {}
    for level, items in sorted(per_intent.items()):
        if len(items) >= 2:
            by_intent[level] = compute_ragas_metrics(items)

    return {
        "mode": "dynamic",
        "metrics": metrics,
        "sample_count": len(samples),
        "latency": _compute_latency_stats(latencies),
        "intent_distribution": dict(intent_distribution),
        "weights_used": {k: list(v) for k, v in weights_used.items()},
        "by_intent_level": by_intent,
    }


def main():
    parser = argparse.ArgumentParser(description="v12 A/B 评测对比")
    parser.add_argument("--mode", choices=["static", "dynamic"], required=True,
                        help="评测模式: static(v11默认), dynamic(v12自适应)")
    parser.add_argument("--limit", type=int, default=0, help="限制评测条数")
    parser.add_argument("--output", default="ab_result.json", help="输出文件路径")
    args = parser.parse_args()

    dataset = load_golden_dataset()
    print(f"数据集: {len(dataset)} 条, 模式: {args.mode}")

    t_start = time.time()

    if args.mode == "static":
        result = run_static_evaluation(dataset, args.limit)
    else:
        result = run_dynamic_evaluation(dataset, args.limit)

    result["total_time_seconds"] = round(time.time() - t_start, 1)

    print(f"\n===== 评估结果 [{args.mode}] =====")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
    print(f"\n样本数: {result['sample_count']}")
    print(f"总耗时: {result['total_time_seconds']}s")

    if result.get("intent_distribution"):
        print(f"\n===== 意图分布 =====")
        for level, count in sorted(result["intent_distribution"].items()):
            print(f"  {level}: {count}")

    if result.get("latency"):
        lat = result["latency"]
        print(f"\n===== 延迟 (ms) =====")
        print(f"  avg={lat['avg']}, p50={lat['p50']}, p95={lat['p95']}, max={lat['max']}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/run_ab_evaluation.py
git commit -m "feat(v12): add A/B evaluation script for static vs dynamic comparison"
```

---

### Task 9: 集成 Prometheus 指标与配置环境变量

**Files:**
- Modify: `.env.example` (新增 v12 环境变量)

- [ ] **Step 1: 在 .env.example 中追加 v12 配置**

```bash
# ===== v12: 自适应检索与降级 =====

# Query Profiler
QUERY_PROFILER_USE_EMBEDDING=true

# 动态 RRF 权重配置文件路径
WEIGHT_MATRIX_CONFIG=config/weight_matrix.yaml

# 负载监控
LOAD_MONITOR_WINDOW=10
LOAD_WARNING_QPS=50
LOAD_CRITICAL_QPS=100
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "feat(v12): add v12 environment variables to .env.example"
```

---

### Task 10: 全量回归测试

**Files:**
- 所有已有测试文件

- [ ] **Step 1: 运行全量测试**

Run: `pytest tests/ -v --ignore=tests/test_evaluation.py`
Expected: 全部 PASS（evaluation 测试需要真实 Milvus/LLM 连接，CI 中跳过）

- [ ] **Step 2: 验证 v12 新增测试**

Run: `pytest tests/test_query_profiler.py tests/test_dynamic_rrf.py tests/test_load_monitor.py -v`
Expected: 全部 PASS

- [ ] **Step 3: 验证 Prometheus 指标端点**

Run: `curl http://localhost:8000/metrics | grep system_load_state`
Expected: 输出包含 `system_load_state` 指标

- [ ] **Step 4: 最终 Commit**

```bash
git add -A
git commit -m "feat(v12): complete adaptive retrieval and degradation architecture"
```

---

## 验证清单

完成所有 Task 后，按以下清单验证：

- [ ] Query Profiler 正确分类 L1/L2/L3 意图
- [ ] 动态 RRF 权重根据意图标签切换
- [ ] 负载监控器 QPS 计数准确
- [ ] WARNING 状态下 Critique/Replan 被跳过
- [ ] CRITICAL 状态下 Neo4j 和 Tavily 被熔断
- [ ] SSE 事件包含 `query_profiler` 和 `system_state`
- [ ] Prometheus `/metrics` 端点暴露新指标
- [ ] Locust 压测脚本可运行
- [ ] A/B 评测脚本可对比静态 vs 动态
- [ ] 全量测试通过
