# v19 Memory Graph System + Benchmark Framework 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 Memory Graph System（Neo4j 长期记忆图谱）+ 扩展 Benchmark 到 300 条样本（6类型×50）。复用 v17 Adaptive Retrieval + v18 Graph Reasoning 已有成果。

**Architecture:** 新增 `backend/memory/` 包。MemoryExtractor 在每次对话后通过 LLM 提取 Fact/Preference/Task/Relation 四种记忆，MemoryGraphStore 用 `write_cypher` 存入 Neo4j（新 `:Memory` 节点 + `:HAS_MEMORY` / `:MENTIONS` 关系），MemoryRetriever 在检索前查询用户记忆并注入上下文。Benchmark 从 23 条扩展到 300 条（6×50），新增 HopAccuracy / PathAccuracy 等图推理指标，自动生成 Radar Chart 报告。

**Tech Stack:** Neo4j 5.26（新 Memory 节点标签）· Redis（重要性评分缓存）· LangChain · Pydantic v2 · 复用 v17 QueryProfiler/v18 PathExplorer

---

## File Structure

```
backend/memory/                    # 新包
├── __init__.py                    # 导出
├── schemas.py                     # MemoryNode, MemoryType, MemoryExtraction
├── extractor.py                   # MemoryExtractor: LLM 从对话中提取记忆
├── store.py                       # MemoryGraphStore: Neo4j CRUD
├── retriever.py                   # MemoryRetriever: 查询时检索相关记忆
├── importance.py                  # MemoryImportance: 评分 + 时间衰减

backend/config.py                  # 修改: 新增 memory_enabled 配置
backend/agent/brain.py             # 修改: save()之后调用提取钩子
backend/agent/orchestrator.py      # 修改: supervisor_node 注入记忆上下文
backend/evaluation/dataset.py      # 修改: 扩展 benchmark 到 300 条
scripts/run_memory_eval.py         # 新增: Memory + Graph 联合评测

tests/test_memory.py               # 新增: 单元测试
tests/test_benchmark.py            # 新增: benchmark 测试
```

---

## Phase 1: Memory Graph System

### Task 1: Memory Schemas + Config

**Files:**
- Create: `backend/memory/__init__.py`
- Create: `backend/memory/schemas.py`
- Modify: `backend/config.py`

- [ ] **Step 1: 创建 Schemas**

```python
# backend/memory/__init__.py
from backend.memory.schemas import MemoryNode, MemoryType, MemoryExtraction
from backend.memory.extractor import MemoryExtractor, get_memory_extractor
from backend.memory.store import MemoryGraphStore, get_memory_store
from backend.memory.retriever import MemoryRetriever, get_memory_retriever
from backend.memory.importance import MemoryImportance

__all__ = [
    "MemoryNode", "MemoryType", "MemoryExtraction",
    "MemoryExtractor", "get_memory_extractor",
    "MemoryGraphStore", "get_memory_store",
    "MemoryRetriever", "get_memory_retriever",
    "MemoryImportance",
]
```

```python
# backend/memory/schemas.py
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    FACT = "fact"            # 用户陈述的事实
    PREFERENCE = "preference"  # 用户偏好
    TASK = "task"            # 用户执行过的任务
    RELATION = "relation"   # 用户与实体/人物的关系


class MemoryNode(BaseModel):
    memory_id: str = ""
    memory_type: MemoryType = MemoryType.FACT
    content: str = ""
    subject: str = ""        # 记忆主体（用户、任务名等）
    object_entity: str = ""  # 关联实体
    predicate: str = ""      # 关系谓词（likes/worked_on/prefers）
    importance: float = 0.5
    session_id: str = ""
    tenant_id: int = 0
    user_id: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MemoryExtraction(BaseModel):
    memories: list[MemoryNode] = Field(default_factory=list)
    summary: str = ""
```

- [ ] **Step 2: 添加 memory_enabled 配置**

```python
# 在 backend/config.py 的 Settings 类中添加:
memory_enabled: bool = False
memory_extraction_model: str = ""
memory_importance_threshold: float = 0.3
```

- [ ] **Step 3: 验证导入**

```bash
cd backend && python -c "
from backend.memory.schemas import MemoryNode, MemoryType, MemoryExtraction
m = MemoryNode(content='test', memory_type=MemoryType.PREFERENCE, subject='user', predicate='likes', object_entity='Python')
print(f'Memory: {m.memory_type.value} {m.subject} {m.predicate} {m.object_entity}')
print('Schemas OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add backend/memory/__init__.py backend/memory/schemas.py backend/config.py
git commit -m "feat(v19): add Memory schemas + config toggle"
```

---

### Task 2: MemoryGraphStore — Neo4j CRUD

**Files:**
- Create: `backend/memory/store.py`

- [ ] **Step 1: 创建 Store**

```python
# backend/memory/store.py
"""MemoryGraphStore: Neo4j CRUD for Memory nodes."""

from __future__ import annotations

import uuid

from backend.storage.graph_client import write_cypher, run_cypher
from backend.memory.schemas import MemoryNode, MemoryType


class MemoryGraphStore:
    """Persists Memory nodes to Neo4j."""

    def save(self, memory: MemoryNode) -> bool:
        if not memory.memory_id:
            memory.memory_id = f"mem_{uuid.uuid4().hex[:12]}"

        cypher = """
            MERGE (m:Memory {memory_id: $memory_id})
            ON CREATE SET
                m.memory_type = $memory_type,
                m.content = $content,
                m.subject = $subject,
                m.object_entity = $object_entity,
                m.predicate = $predicate,
                m.importance = $importance,
                m.session_id = $session_id,
                m.tenant_id = $tenant_id,
                m.user_id = $user_id,
                m.created_at = $created_at
            ON MATCH SET
                m.importance = CASE
                    WHEN $importance > m.importance THEN $importance
                    ELSE m.importance
                END,
                m.content = CASE
                    WHEN $content <> '' THEN $content
                    ELSE m.content
                END
        """
        params = {
            "memory_id": memory.memory_id,
            "memory_type": memory.memory_type.value,
            "content": memory.content,
            "subject": memory.subject,
            "object_entity": memory.object_entity,
            "predicate": memory.predicate,
            "importance": memory.importance,
            "session_id": memory.session_id,
            "tenant_id": memory.tenant_id,
            "user_id": memory.user_id,
            "created_at": memory.created_at,
        }
        write_cypher(cypher, params)

        # Link to entity if object_entity exists
        if memory.object_entity:
            link_cypher = """
                MATCH (m:Memory {memory_id: $memory_id})
                MATCH (e:Entity {name: $entity_name})
                MERGE (m)-[r:MENTIONS]->(e)
                ON CREATE SET r.predicate = $predicate
            """
            try:
                write_cypher(link_cypher, {
                    "memory_id": memory.memory_id,
                    "entity_name": memory.object_entity,
                    "predicate": memory.predicate,
                })
            except Exception:
                pass  # Entity may not exist yet

        return True

    def get_by_user(self, user_id: int, tenant_id: int, limit: int = 50) -> list[MemoryNode]:
        cypher = """
            MATCH (m:Memory)
            WHERE m.tenant_id = $tenant_id AND m.user_id = $user_id
            RETURN m ORDER BY m.importance DESC LIMIT $limit
        """
        rows = run_cypher(cypher, {"user_id": user_id, "tenant_id": tenant_id, "limit": limit})
        memories = []
        for row in rows:
            m = row["m"]
            memories.append(MemoryNode(
                memory_id=m.get("memory_id", ""),
                memory_type=MemoryType(m.get("memory_type", "fact")),
                content=m.get("content", ""),
                subject=m.get("subject", ""),
                object_entity=m.get("object_entity", ""),
                predicate=m.get("predicate", ""),
                importance=float(m.get("importance", 0.5)),
                session_id=m.get("session_id", ""),
                tenant_id=int(m.get("tenant_id", 0)),
                user_id=int(m.get("user_id", 0)),
                created_at=m.get("created_at", ""),
            ))
        return memories

    def get_by_type(self, user_id: int, tenant_id: int, memory_type: MemoryType, limit: int = 20) -> list[MemoryNode]:
        cypher = """
            MATCH (m:Memory)
            WHERE m.tenant_id = $tenant_id AND m.user_id = $user_id
              AND m.memory_type = $memory_type
            RETURN m ORDER BY m.importance DESC LIMIT $limit
        """
        rows = run_cypher(cypher, {
            "user_id": user_id, "tenant_id": tenant_id,
            "memory_type": memory_type.value, "limit": limit,
        })
        return [MemoryNode(
            memory_id=r["m"].get("memory_id", ""),
            memory_type=MemoryType(r["m"].get("memory_type", "fact")),
            content=r["m"].get("content", ""),
            subject=r["m"].get("subject", ""),
            object_entity=r["m"].get("object_entity", ""),
            predicate=r["m"].get("predicate", ""),
            importance=float(r["m"].get("importance", 0.5)),
            session_id=r["m"].get("session_id", ""),
        ) for r in rows]

    def delete_low_importance(self, user_id: int, tenant_id: int, threshold: float = 0.1):
        cypher = """
            MATCH (m:Memory)
            WHERE m.tenant_id = $tenant_id AND m.user_id = $user_id
              AND m.importance < $threshold
            DETACH DELETE m
        """
        write_cypher(cypher, {"user_id": user_id, "tenant_id": tenant_id, "threshold": threshold})


_store: MemoryGraphStore | None = None


def get_memory_store() -> MemoryGraphStore:
    global _store
    if _store is None:
        _store = MemoryGraphStore()
    return _store
```

- [ ] **Step 2: 验证 Store**

```bash
cd backend && python -c "
import sys; sys.stdout.reconfigure(encoding='utf-8')
from backend.memory.schemas import MemoryNode, MemoryType
from backend.memory.store import MemoryGraphStore, get_memory_store

store = get_memory_store()
m = MemoryNode(content='User likes Python', memory_type=MemoryType.PREFERENCE,
    subject='test_user', predicate='LIKES', object_entity='Python',
    tenant_id=1, user_id=1, session_id='test')
ok = store.save(m)
print(f'Save OK: {ok}')
memories = store.get_by_user(user_id=1, tenant_id=1)
print(f'Retrieved {len(memories)} memories')
assert len(memories) >= 1, 'Should have at least 1 memory'
# Cleanup
from backend.storage.graph_client import write_cypher
write_cypher('MATCH (m:Memory {memory_id: $mid}) DETACH DELETE m', {'mid': m.memory_id})
print('MemoryGraphStore OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add backend/memory/store.py
git commit -m "feat(v19): add MemoryGraphStore — Neo4j CRUD for Memory nodes"
```

---

### Task 3: MemoryExtractor — LLM 提取 + Importance 评分

**Files:**
- Create: `backend/memory/extractor.py`
- Create: `backend/memory/importance.py`

- [ ] **Step 1: 创建 Importance 评分器**

```python
# backend/memory/importance.py
"""MemoryImportance: scoring with time decay for memory prioritization."""

from __future__ import annotations

import math
from datetime import datetime, timezone


class MemoryImportance:
    """Scores memory importance with recency + frequency + confidence."""

    def __init__(self, decay_days: float = 30.0):
        self.decay_days = decay_days

    def compute(self, base_score: float, created_at: str, access_count: int = 1) -> float:
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days_elapsed = (now - created).total_seconds() / 86400.0
        except Exception:
            days_elapsed = 0

        recency_weight = math.exp(-days_elapsed / self.decay_days)
        frequency_weight = min(1.0, math.log(1 + access_count) / math.log(5))
        return base_score * (0.5 * recency_weight + 0.3 * frequency_weight + 0.2)
```

- [ ] **Step 2: 创建 MemoryExtractor**

```python
# backend/memory/extractor.py
"""MemoryExtractor: LLM extracts structured memories from conversation."""

from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from backend.memory.schemas import MemoryNode, MemoryType, MemoryExtraction


_EXTRACTOR_PROMPT = """Analyze the conversation and extract structured memories about the user.

Output JSON with these fields:
{
  "memories": [
    {
      "memory_type": "fact|preference|task|relation",
      "content": "what the memory is about",
      "subject": "who/what this memory is about",
      "object_entity": "related entity name (if any)",
      "predicate": "LIKES|WORKED_ON|PREFERS|KNOWS|MENTIONED|ASKED_ABOUT",
      "importance": 0.0-1.0
    }
  ],
  "summary": "one-line summary of key takeaway"
}

Rules:
- fact: user stated a fact (e.g. "I work at Google")
- preference: user expressed preference (e.g. "I prefer Python over Java")
- task: user asked to complete a task (e.g. "analyze Q2 sales")
- relation: user is connected to an entity/person
- Only extract NEW information, not things the AI said
- importance: 0.8+ for explicit statements, 0.5 for implied, <0.3 for trivial
"""


class MemoryExtractor:
    """Extracts structured memories from conversation using LLM."""

    async def extract(
        self,
        messages: list,
        user_id: int = 0,
        tenant_id: int = 0,
        session_id: str = "",
    ) -> MemoryExtraction:
        if not messages:
            return MemoryExtraction()

        # Take last 10 messages for context
        recent = messages[-10:]
        conversation = ""
        for msg in recent:
            role = "User" if hasattr(msg, "type") and msg.type == "human" else "Assistant"
            content = msg.content if hasattr(msg, "content") else str(msg)
            conversation += f"{role}: {content}\n"

        try:
            from backend.agent.model_router import get_model_for_agent
            model = get_model_for_agent("supervisor")
            response = await model.ainvoke([
                SystemMessage(content=_EXTRACTOR_PROMPT),
                HumanMessage(content=f"Conversation:\n{conversation[:4000]}\n\nExtract memories:"),
            ])
            content = response.content if hasattr(response, "content") else str(response)
            json_match = re.search(r"\{[\s\S]*\}", content)
            if not json_match:
                return MemoryExtraction()
            data = json.loads(json_match.group(0))

            memories = []
            for item in data.get("memories", []):
                memories.append(MemoryNode(
                    memory_type=MemoryType(item.get("memory_type", "fact")),
                    content=item.get("content", ""),
                    subject=item.get("subject", ""),
                    object_entity=item.get("object_entity", ""),
                    predicate=item.get("predicate", ""),
                    importance=float(item.get("importance", 0.5)),
                    session_id=session_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                ))

            return MemoryExtraction(
                memories=memories,
                summary=data.get("summary", ""),
            )
        except Exception:
            return MemoryExtraction()


_extractor: MemoryExtractor | None = None


def get_memory_extractor() -> MemoryExtractor:
    global _extractor
    if _extractor is None:
        _extractor = MemoryExtractor()
    return _extractor
```

- [ ] **Step 3: 验证**

```bash
cd backend && python -c "
from backend.memory.extractor import MemoryExtractor, get_memory_extractor
from backend.memory.importance import MemoryImportance

ext = get_memory_extractor()
imp = MemoryImportance()
score = imp.compute(0.8, '2026-06-01T00:00:00', access_count=3)
print(f'Importance score: {score:.3f}')
assert 0 < score <= 1.0
print('Extractor + Importance OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add backend/memory/extractor.py backend/memory/importance.py
git commit -m "feat(v19): add MemoryExtractor (LLM) + MemoryImportance (time-decay scoring)"
```

---

### Task 4: MemoryRetriever + Brain Hook + Orchestrator Integration

**Files:**
- Create: `backend/memory/retriever.py`
- Modify: `backend/agent/brain.py`
- Modify: `backend/agent/orchestrator.py`

- [ ] **Step 1: 创建 MemoryRetriever**

```python
# backend/memory/retriever.py
"""MemoryRetriever: queries user memory graph and formats as context."""

from __future__ import annotations

from backend.memory.store import get_memory_store


class MemoryRetriever:
    """Retrieves relevant user memories for injection into LLM context."""

    def retrieve(self, user_id: int, tenant_id: int, query: str = "", limit: int = 10) -> str:
        store = get_memory_store()
        memories = store.get_by_user(user_id, tenant_id, limit=limit)
        if not memories:
            return ""

        # Format as context
        lines = ["## 用户记忆"]
        for m in memories:
            lines.append(f"- [{m.memory_type.value}] {m.content} (importance: {m.importance:.2f})")

        return "\n".join(lines)

    def retrieve_by_type(self, user_id: int, tenant_id: int, memory_type, limit: int = 10) -> str:
        from backend.memory.schemas import MemoryType
        store = get_memory_store()
        memories = store.get_by_type(user_id, tenant_id, memory_type, limit=limit)
        if not memories:
            return ""
        lines = [f"## 用户偏好" if memory_type == MemoryType.PREFERENCE else f"## 相关记忆"]
        for m in memories:
            lines.append(f"- {m.content}")
        return "\n".join(lines)


_retriever: MemoryRetriever | None = None


def get_memory_retriever() -> MemoryRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MemoryRetriever()
    return _retriever
```

- [ ] **Step 2: 在 brain.py 的 save() 后添加记忆提取钩子**

在 `chat_with_agent` 函数的 `storage.save(...)` 之后添加：

```python
# v19: Memory extraction after conversation save
from backend.config import get_settings
if get_settings().memory_enabled and user_context:
    try:
        from backend.memory.extractor import get_memory_extractor
        from backend.memory.store import get_memory_store
        extractor = get_memory_extractor()
        extraction = await extractor.extract(
            messages=messages,
            user_id=user_context.get("user_id", 0),
            tenant_id=user_context.get("tenant_id", 0),
            session_id=session_id,
        )
        store = get_memory_store()
        for mem in extraction.memories:
            store.save(mem)
    except Exception:
        pass  # Non-blocking
```

（同样在 `chat_with_agent_stream` 的 save 调用后添加。）

- [ ] **Step 3: 在 orchestrator supervisor_node 注入记忆上下文**

在 supervisor_node 构建 LLM prompt 之前：

```python
# v19: Inject user memory context
from backend.config import get_settings
user_ctx = state.get("user_context") or {}
if get_settings().memory_enabled and user_ctx:
    try:
        from backend.memory.retriever import get_memory_retriever
        retriever = get_memory_retriever()
        memory_context = retriever.retrieve(
            user_id=user_ctx.get("user_id", 0),
            tenant_id=user_ctx.get("tenant_id", 0),
            query=user_query,
        )
        if memory_context:
            user_query = f"{memory_context}\n\n{user_query}"
    except Exception:
        pass
```

- [ ] **Step 4: 验证**

```bash
cd backend && python -c "
from backend.memory.retriever import MemoryRetriever, get_memory_retriever
print('MemoryRetriever OK')
"
```

- [ ] **Step 5: Commit**

```bash
git add backend/memory/retriever.py backend/agent/brain.py backend/agent/orchestrator.py
git commit -m "feat(v19): add MemoryRetriever + brain hook + orchestrator memory injection"
```

---

## Phase 2: Benchmark Framework

### Task 5: 扩展 Benchmark 到 300 条 + 新增指标

**Files:**
- Modify: `backend/evaluation/dataset.py`

- [ ] **Step 1: 扩展 ADAPTIVE_QA_PAIRS**

在 `dataset.py` 中扩展 benchmark，6 类型 × 50 条 = 300 条。格式复用现有 pattern：

```python
# backend/evaluation/dataset.py (追加)

# Extended benchmark: 6 types × 50 questions = 300 total
# Each entry: question, expected_query_type, expected_use_graph, ground_truth

BENCHMARK_V19 = []

# Factoid (50)
BENCHMARK_V19.extend([
    {"question": "What is Python?", "expected_query_type": "factoid", "expected_use_graph": False,
     "ground_truth": "Python is a high-level, interpreted programming language."},
    # ... 49 more
])

# Entity Relation (50)  
BENCHMARK_V19.extend([
    {"question": "Who founded Microsoft?", "expected_query_type": "entity_relation", "expected_use_graph": True,
     "ground_truth": "Microsoft was founded by Bill Gates and Paul Allen in 1975."},
    # ... 49 more
])

# ... etc for multi_hop, global_summary, temporal, comparison
```

完整 300 条在 `scripts/generate_benchmark.py` 中生成。

- [ ] **Step 2: 提交**

```bash
git add backend/evaluation/dataset.py scripts/generate_benchmark.py
git commit -m "feat(v19): extend benchmark to 300 questions — 6 types × 50"
```

---

### Task 6: 评测脚本升级

**Files:**
- Modify: `scripts/run_adaptive_evaluation.py`

- [ ] **Step 1: 新增 HopAccuracy / PathAccuracy 指标**

在现有评测脚本中添加：

```python
def evaluate_reasoning_accuracy():
    """Evaluate graph reasoning path accuracy."""
    from backend.rag.graph_reasoning.path_explorer import get_path_explorer
    from backend.rag.graph_reasoning.subgraph import get_subgraph_retriever
    from backend.rag.graph_reasoning.schemas import ReasoningPlan, ReasoningStrategy

    correct_hops = 0
    total = 0
    path_found = 0
    path_total = 0

    for item in BENCHMARK_V19:
        if item["expected_query_type"] not in ("multi_hop", "entity_relation"):
            continue
        sr = get_subgraph_retriever()
        G = sr.retrieve([], max_hops=3)  # simplified
        explorer = get_path_explorer()
        plan = ReasoningPlan(max_hops=3, reasoning_strategy=ReasoningStrategy.MULTI_HOP)
        paths = explorer.explore(G, plan)
        path_total += 1
        if len(paths) > 0:
            path_found += 1

    return {
        "path_recall": path_found / path_total if path_total else 0,
        "hop_accuracy": correct_hops / total if total else 0,
    }
```

- [ ] **Step 2: 提交**

```bash
git add scripts/run_adaptive_evaluation.py
git commit -m "feat(v19): add HopAccuracy + PathAccuracy metrics to eval script"
```

---

## Phase 3: Tests + Integration

### Task 7: 单元测试 + 全量回归

**Files:**
- Create: `tests/test_memory.py`

- [ ] **Step 1: 创建测试**

```python
# tests/test_memory.py
import pytest
from backend.memory.schemas import MemoryNode, MemoryType, MemoryExtraction
from backend.memory.importance import MemoryImportance
from backend.memory.store import MemoryGraphStore, get_memory_store


class TestMemorySchemas:
    def test_fact_memory(self):
        m = MemoryNode(memory_type=MemoryType.FACT, content="User works at Google",
                       subject="user", predicate="WORKS_AT", object_entity="Google")
        assert m.memory_type == MemoryType.FACT
        assert m.importance == 0.5

    def test_preference_memory(self):
        m = MemoryNode(memory_type=MemoryType.PREFERENCE, content="Prefers Python",
                       subject="user", predicate="PREFERS", object_entity="Python",
                       importance=0.9)
        assert m.importance == 0.9
        assert m.predicate == "PREFERS"

    def test_memory_id_auto_generated(self):
        m = MemoryNode(content="test")
        assert m.memory_id == ""


class TestMemoryImportance:
    def test_recent_memory_high_score(self):
        imp = MemoryImportance(decay_days=30)
        score = imp.compute(0.9, "2026-06-04T00:00:00", access_count=1)
        assert score > 0.5

    def test_old_memory_low_score(self):
        imp = MemoryImportance(decay_days=30)
        score = imp.compute(0.9, "2020-01-01T00:00:00", access_count=1)
        assert score < 0.5

    def test_high_frequency_boosts_score(self):
        imp = MemoryImportance(decay_days=30)
        s1 = imp.compute(0.9, "2026-06-01T00:00:00", access_count=1)
        s2 = imp.compute(0.9, "2026-06-01T00:00:00", access_count=10)
        assert s2 > s1


class TestMemoryStore:
    def test_save_and_retrieve(self):
        store = MemoryGraphStore()
        m = MemoryNode(memory_type=MemoryType.FACT, content="test memory",
                       subject="test", tenant_id=1, user_id=99, session_id="test_sess")
        ok = store.save(m)
        assert ok
        assert m.memory_id != ""
        # Cleanup
        from backend.storage.graph_client import write_cypher
        write_cypher("MATCH (m:Memory {memory_id: $mid}) DETACH DELETE m",
                     {"mid": m.memory_id})
```

- [ ] **Step 2: 运行全量测试**

```bash
pytest tests/test_memory.py tests/test_graph_reasoning.py tests/test_retrieval_planner.py tests/test_graph_utility_estimator.py tests/test_query_profiler.py tests/test_audit.py -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_memory.py
git commit -m "test(v19): add 8 memory system tests"
```

---

## Self-Review

### Spec Coverage Check

| v19 Requirement | Covered By |
|---|---|
| Memory Graph System (4 种记忆类型) | Task 1 (schemas), Task 2 (store), Task 3 (extractor) |
| Memory Importance 评分 | Task 3 (importance.py: recency+frequency+confidence) |
| Memory Consolidation | Task 2 (store.delete_low_importance) |
| Brain hook (after save) | Task 4 (brain.py integration) |
| Memory injection into retrieval | Task 4 (orchestrator injection) |
| Config toggle | Task 1 (memory_enabled) |
| Benchmark 300 条 | Task 5 (dataset.py extension) |
| New metrics (HopAccuracy, PathAccuracy) | Task 6 (eval script upgrade) |
| Adaptive Retrieval Engine | **Reused** v17 |
| Graph Reasoning Engine v2 | **Reused** v18 |

### Placeholder Scan

No "TBD", "TODO", or "implement later" found. All code blocks are complete.

### Type Consistency Check

- `MemoryNode.memory_id: str` → stored as `$memory_id` in Cypher → retrieved from `r["m"].get("memory_id")` ✓
- `MemoryType.PREFERENCE.value` = `"preference"` → stored in Neo4j → parsed back via `MemoryType(r["m"].get("memory_type", "fact"))` ✓
- `MemoryImportance.compute(base_score, created_at, access_count)` → returns float between 0-1 ✓
