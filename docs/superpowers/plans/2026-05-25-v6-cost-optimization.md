# Ragent AI v6.0 — 降本增效（Semantic Cache + Dynamic Routing + Cache Protection）升级计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为高频重复提问构建毫秒级语义缓存，按任务复杂度动态分配模型算力，实现缓存防击穿和生命周期管理。

**Architecture:** 请求入口处先查语义缓存（Milvus ANN + cosine ≥ 0.95）→ 命中直接返回（200ms, 0 Token）→ 未命中走正常 LangGraph 流程。Supervisor/闲聊用 qwen-turbo，复杂推理用 qwen-plus/max。Redis Singleflight 防缓存击穿。

**Tech Stack:** Milvus · MySQL · Redis · text-embedding-v1 · LLMLingua · Alembic · qwen-turbo/qwen-plus/qwen-max

---

## 文件结构概览

```
新增文件 (7):
  backend/cache/__init__.py                 # 缓存模块入口
  backend/cache/semantic_cache.py           # 语义缓存层（Milvus ANN + MySQL + cosine）
  backend/cache/singleflight.py             # Redis Singleflight 防击穿
  backend/cache/invalidation.py             # 文档删除 → 缓存失效事件
  backend/agent/model_router.py             # 动态模型路由策略引擎
  scripts/run_benchmark.py                  # 并发压测 + Token 对比脚本
  alembic/versions/001_query_cache_store.py # 数据库迁移脚本

修改文件 (7):
  backend/agent/orchestrator.py             # Supervisor/DirectAnswer → qwen-turbo; 动态模型选择
  backend/agent/brain.py                    # chat_with_agent_stream 入口加缓存查询
  backend/storage/models.py                 # +QueryCacheStore 表
  backend/milvus/client.py                  # +semantic_cache_collection 初始化
  backend/storage/doc_lifecycle.py          # 软删除触发缓存失效
  backend/storage/cache.py                  # +semantic_cache 读写方法
  pyproject.toml                            # +llmlingua, alembic
  .env.example                              # +缓存/env 配置
```

---

## Phase 1: 语义缓存层

### Task 1.1: MySQL QueryCacheStore 表 + Milvus 缓存集合

**Files:**
- Modify: `backend/storage/models.py` (新增 QueryCacheStore)
- Modify: `backend/milvus/client.py` (初始化 semantic_cache_collection)

- [ ] **Step 1: 新增 QueryCacheStore ORM 模型**

在 `backend/storage/models.py` 末尾:

```python
class QueryCacheStore(Base):
    """语义缓存存储表 — 存储高频问题的 LLM 回答。"""
    __tablename__ = "query_cache_store"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    vector_id: Mapped[str] = mapped_column(String(64), nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_doc: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    hit_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    ttl_seconds: Mapped[int] = mapped_column(Integer, default=86400, nullable=False)
```

- [ ] **Step 2: Milvus 初始化缓存集合**

在 `backend/milvus/client.py` 的 `MilvusManager` 中添加:

```python
CACHE_COLLECTION = "semantic_cache_collection"

def init_cache_collection(self):
    client = self._client()
    if not client.has_collection(self.CACHE_COLLECTION):
        schema = client.create_schema(auto_id=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)
        schema.add_field("query_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("query_text", DataType.VARCHAR, max_length=2000)
        schema.add_field("source_doc", DataType.VARCHAR, max_length=255)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE")
        client.create_collection(collection_name=self.CACHE_COLLECTION, schema=schema, index_params=index_params)
        client.load_collection(self.CACHE_COLLECTION)

def search_cache(self, query_vector: list[float], top_k: int = 3) -> list[dict]:
    client = self._client()
    results = client.search(
        collection_name=self.CACHE_COLLECTION,
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=top_k,
        output_fields=["query_hash", "query_text", "source_doc"],
    )
    return results[0] if results else []

def insert_cache(self, query_vector, query_hash, query_text, source_doc=""):
    client = self._client()
    data = [{
        "embedding": query_vector,
        "query_hash": query_hash,
        "query_text": query_text[:2000],
        "source_doc": source_doc,
    }]
    return client.insert(self.CACHE_COLLECTION, data)

def delete_cache_by_source(self, source_doc: str) -> int:
    client = self._client()
    res = client.delete(collection_name=self.CACHE_COLLECTION, filter=f'source_doc == "{source_doc}"')
    return res.get("delete_count", 0) if isinstance(res, dict) else 0
```

- [ ] **Step 3: 验证表创建 + 集合初始化**

```bash
uv run python -c "
from backend.storage.database import engine, Base
from backend.storage.models import *
Base.metadata.create_all(engine)
from backend.milvus.client import MilvusManager
m = MilvusManager()
m.init_cache_collection()
print('Cache collection ready')
"
```

- [ ] **Step 4: Commit**

```bash
git add backend/storage/models.py backend/milvus/client.py
git commit -m "feat: add QueryCacheStore table and Milvus semantic_cache_collection"
```

---

### Task 1.2: 语义缓存层核心逻辑

**Files:**
- Create: `backend/cache/__init__.py`
- Create: `backend/cache/semantic_cache.py`

- [ ] **Step 1: 实现语义缓存匹配与存储**

```python
# backend/cache/semantic_cache.py
"""语义缓存层：Milvus ANN + cosine 匹配, MySQL 存储。"""
import os
import hashlib
import numpy as np

from backend.milvus.client import MilvusManager
from backend.embedding.service import EmbeddingService
from backend.storage.database import SessionLocal
from backend.storage.models import QueryCacheStore

CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

_milvus = MilvusManager()
_embedding = EmbeddingService()


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def query_cache(query: str) -> dict | None:
    """查询语义缓存。返回缓存命中结果或 None。"""
    query_vector = _embedding.get_embeddings([query])[0]

    # Step 1: Milvus ANN 检索
    _milvus.init_cache_collection()
    candidates = _milvus.search_cache(query_vector, top_k=3)

    # Step 2: cosine ≥ 阈值检查
    best_score = 0.0
    best_hash = None
    for c in candidates:
        emb = np.array(c.get("embedding", c.get("entity", {}).get("embedding", [])))
        if len(emb) == 0:
            score = c.get("distance", 0)
        else:
            score = np.dot(query_vector, emb) / (
                np.linalg.norm(query_vector) * np.linalg.norm(emb)
            )
        if score > best_score:
            best_score = score
            best_hash = c.get("query_hash") or c.get("entity", {}).get("query_hash", "")

    if best_score < CACHE_SIMILARITY_THRESHOLD:
        return None  # 未命中

    # Step 3: MySQL 读取缓存响应 + 更新 hit_count
    with SessionLocal() as session:
        entry = session.query(QueryCacheStore).filter_by(query_hash=best_hash).first()
        if entry:
            entry.hit_count += 1
            session.commit()
            return {
                "response": entry.response_text,
                "source_doc": entry.source_doc,
                "similarity": round(float(best_score), 4),
                "hit_count": entry.hit_count,
                "cached": True,
            }
    return None


def write_cache(query: str, response: str, source_doc: str = "") -> dict:
    """写入语义缓存（Milvus + MySQL）。"""
    query_vector = _embedding.get_embeddings([query])[0]
    query_hash = _hash_query(query)

    # Step 1: Milvus 向量写入
    _milvus.init_cache_collection()
    insert_result = _milvus.insert_cache(query_vector, query_hash, query, source_doc)
    vector_id = str(insert_result.get("ids", [None])[0]) if insert_result else ""

    # Step 2: MySQL 文本写入（upsert）
    with SessionLocal() as session:
        existing = session.query(QueryCacheStore).filter_by(query_hash=query_hash).first()
        if existing:
            existing.response_text = response
            existing.hit_count = 1
            existing.updated_at = QueryCacheStore.updated_at.default.arg()
        else:
            session.add(QueryCacheStore(
                query_hash=query_hash,
                vector_id=vector_id,
                response_text=response,
                source_doc=source_doc,
            ))
        session.commit()

    return {"query_hash": query_hash, "vector_id": vector_id, "status": "cached"}
```

- [ ] **Step 2: 创建 __init__.py**

```python
# backend/cache/__init__.py
from .semantic_cache import query_cache, write_cache
from .singleflight import with_singleflight
from .invalidation import invalidate_by_filename
```

- [ ] **Step 3: 验证缓存逻辑**

```bash
uv run python -c "
from backend.cache.semantic_cache import query_cache, write_cache
result = query_cache('这是一个测试问题')
print(f'Cache miss: {result}')  # None expected
write_cache('测试', '这是测试响应')
result2 = query_cache('测试')
print(f'Cache hit: {result2}')
"
```

Expected: 第一次 miss (None), 第二次 hit (返回缓存响应)。

- [ ] **Step 4: Commit**

```bash
git add backend/cache/
git commit -m "feat: semantic cache layer — Milvus ANN + cosine + MySQL store"
```

---

### Task 1.3: 请求入口集成语义缓存

**Files:**
- Modify: `backend/agent/brain.py` (chat_with_agent_stream 入口)

- [ ] **Step 1: 在 SSE 流入口处查询缓存**

在 `chat_with_agent_stream` 函数起始处（加载历史消息后、LangGraph 执行前）插入:

```python
async def chat_with_agent_stream(session_id: str, user_message: str):
    # ... 现有逻辑（加载历史、HITL 检查等）...

    # --- v6.0 语义缓存查询 ---
    from backend.cache import query_cache, write_cache
    cache_result = query_cache(user_message)
    if cache_result:
        trace_data = {
            "type": "cache_hit",
            "similarity": cache_result["similarity"],
            "hit_count": cache_result["hit_count"],
        }
        yield f"data: {json.dumps(trace_data)}\n\n"
        yield f"data: {json.dumps({'type': 'content', 'content': cache_result['response']})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return
    # --- 缓存未命中，继续正常流程 ---
```

- [ ] **Step 2: 回答生成后写入缓存**

在 LangGraph 图执行完成后、StreamingResponse 返回前:

```python
    # 原流程结束，full_response 已生成
    if full_response and not interrupt_info:
        try:
            write_cache(user_message, full_response)
        except Exception:
            pass  # 缓存写入失败不阻塞主流程
```

- [ ] **Step 3: 验证端到端缓存**

```bash
uv run python -c "
# 模拟两次相同问题
from backend.agent.brain import chat_with_agent_stream
# ... 测试逻辑
"
```

- [ ] **Step 4: Commit**

```bash
git add backend/agent/brain.py
git commit -m "feat: integrate semantic cache into chat stream entry point"
```

---

## Phase 2: 动态模型路由

### Task 2.1: 模型路由策略引擎

**Files:**
- Create: `backend/agent/model_router.py`

- [ ] **Step 1: 实现三级模型路由**

```python
# backend/agent/model_router.py
"""动态模型路由：按任务复杂度分配算力。"""
import os

# 三级模型配置
MODEL_TURBO = os.getenv("MODEL_TURBO", "qwen-turbo")
MODEL_PLUS = os.getenv("MODEL", "qwen-plus")
MODEL_MAX = os.getenv("MODEL_MAX", "qwen-max")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("ARK_API_KEY")

ROUTE_MAP = {
    # 轻量任务 → turbo
    "supervisor": MODEL_TURBO,
    "direct_answer": MODEL_TURBO,
    # 中等任务 → plus
    "web_searcher": MODEL_PLUS,
    "rag_specialist": MODEL_PLUS,
    "synthesize": MODEL_PLUS,
    # 重推理任务 → max (可回退到 plus)
    "data_analyst": os.getenv("DATA_ANALYST_MODEL", MODEL_PLUS),
    "local_graph_search": MODEL_PLUS,
    "global_graph_search": MODEL_PLUS,
    # 复杂多跳推理
    "complex_graph_reasoning": MODEL_MAX,
}


def get_model_for_agent(agent_name: str):
    """根据 Agent 角色获取对应模型实例。"""
    from langchain_openai import ChatOpenAI

    model_name = ROUTE_MAP.get(agent_name, MODEL_PLUS)
    return ChatOpenAI(
        model=model_name,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.0,
    )


def is_lightweight_task(routes: list[str]) -> bool:
    """判断是否轻量任务（闲聊、简单问答）。"""
    return set(routes).issubset({"direct_answer"})
```

- [ ] **Step 2: 修改 orchestrator 使用动态模型**

在 `orchestrator.py` 中:

```python
# 替换 _get_supervisor_model
from backend.agent.model_router import get_model_for_agent

def _get_supervisor_model():
    global _supervisor_model
    if _supervisor_model is None:
        _supervisor_model = get_model_for_agent("supervisor")
    return _supervisor_model

# Direct Answer 节点也降级
def _get_worker_model():
    global _worker_model
    if _worker_model is None:
        _worker_model = get_model_for_agent("rag_specialist")  # 默认 plus
    return _worker_model
```

- [ ] **Step 3: Commit**

```bash
git add backend/agent/model_router.py backend/agent/orchestrator.py
git commit -m "feat: dynamic model routing — turbo for lightweight, plus/max for heavy"
```

---

### Task 2.2: Token 消耗截断（Context Pruning）

**Files:**
- Modify: `backend/agent/orchestrator.py` (synthesize 节点前)

- [ ] **Step 1: 集成 LLMLingua 压缩**

在 synthesize 节点组装 context 后、发送 LLM 前:

```python
def _compress_context(text: str, target_ratio: float = 0.6) -> str:
    """使用 LLMLingua 压缩上下文，保留核心语义。"""
    try:
        from llmlingua import PromptCompressor
        compressor = PromptCompressor(
            model_name=os.getenv("EMBEDDER", "text-embedding-v1"),
            use_llmlingua2=True,
        )
        compressed = compressor.compress_prompt(
            text,
            rate=target_ratio,
            force_tokens=["!", "?", "\n"],
        )
        return compressed.get("compressed_prompt", text)
    except Exception:
        return text  # 压缩失败返回原文
```

- [ ] **Step 2: 在 synthesize 节点调用压缩**

```python
def synthesize_node(state: SupervisorState) -> dict:
    # 收集所有 worker 输出
    combined = _collect_worker_outputs(state.get("worker_outputs", {}))

    # --- v6.0 Context Pruning ---
    if len(combined) > 2000:
        combined = _compress_context(combined, target_ratio=0.6)
    # ---

    # ... 原有 LLM 生成逻辑
```

- [ ] **Step 3: Commit**

```bash
git add backend/agent/orchestrator.py pyproject.toml
git commit -m "feat: add LLMLingua context pruning in synthesize node"
```

---

## Phase 3: 缓存防击穿与生命周期

### Task 3.1: Redis Singleflight 防击穿

**Files:**
- Create: `backend/cache/singleflight.py`

- [ ] **Step 1: 实现 Singleflight 模式**

```python
# backend/cache/singleflight.py
"""Redis Singleflight：防缓存击穿。"""
import time
from backend.storage.cache import cache

SINGLEFLIGHT_TTL = 30  # 等待锁超时


def with_singleflight(key: str, ttl: int = SINGLEFLIGHT_TTL):
    """装饰器：相同 key 的并发请求只穿透一次。"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            lock_key = f"singleflight:{key}"
            result_key = f"singleflight_result:{key}"

            # 尝试获取执行锁
            if cache.acquire_lock(lock_key, ttl=ttl):
                try:
                    result = func(*args, **kwargs)
                    # 写入结果缓存
                    cache.set_json(result_key, result, ttl=60)
                    return result
                finally:
                    cache.release_lock(lock_key)
            else:
                # 等待第一个请求完成
                for _ in range(ttl * 2):
                    cached = cache.get_json(result_key)
                    if cached is not None:
                        return cached
                    time.sleep(0.5)
                # 超时，直接执行
                return func(*args, **kwargs)
        return wrapper
    return decorator
```

- [ ] **Step 2: 应用到缓存写入**

在 `backend/cache/semantic_cache.py` 的 `write_cache` 函数上:

```python
from .singleflight import with_singleflight

@with_singleflight("semantic_cache_write")
def write_cache(query: str, response: str, source_doc: str = "") -> dict:
    # ... 原有逻辑
```

- [ ] **Step 3: Commit**

```bash
git add backend/cache/singleflight.py backend/cache/semantic_cache.py
git commit -m "feat: Redis Singleflight pattern to prevent cache stampede"
```

---

### Task 3.2: 缓存失效 — 文档删除联动

**Files:**
- Create: `backend/cache/invalidation.py`
- Modify: `backend/storage/doc_lifecycle.py` (mark_document_deleted 末尾触发失效)

- [ ] **Step 1: 实现缓存失效函数**

```python
# backend/cache/invalidation.py
"""文档删除 → 语义缓存失效。"""
from backend.milvus.client import MilvusManager
from backend.storage.database import SessionLocal
from backend.storage.models import QueryCacheStore


def invalidate_by_filename(filename: str) -> dict:
    """级联清除与文档相关的所有语义缓存。"""
    milvus = MilvusManager()
    milvus.init_cache_collection()

    # 1. Milvus 缓存向量删除
    milvus_deleted = milvus.delete_cache_by_source(filename)

    # 2. MySQL 缓存条目删除
    with SessionLocal() as session:
        entries = session.query(QueryCacheStore).filter_by(source_doc=filename).all()
        mysql_deleted = len(entries)
        for e in entries:
            session.delete(e)
        session.commit()

    return {"milvus_deleted": milvus_deleted, "mysql_deleted": mysql_deleted}
```

- [ ] **Step 2: mark_document_deleted 末尾触发失效**

在 `backend/storage/doc_lifecycle.py` 的 `mark_document_deleted` 函数 return 之前:

```python
    # v6.0: 缓存失效
    try:
        from backend.cache.invalidation import invalidate_by_filename
        invalidate_result = invalidate_by_filename(filename)
        result["cache_invalidated"] = invalidate_result
    except Exception:
        pass
```

- [ ] **Step 3: Commit**

```bash
git add backend/cache/invalidation.py backend/storage/doc_lifecycle.py
git commit -m "feat: event-driven cache invalidation on document soft-delete"
```

---

### Task 3.3: TTL 过期缓存自动清理

**Files:**
- Modify: `backend/cache/semantic_cache.py` (添加定期清理逻辑)

- [ ] **Step 1: 添加 TTL 过期检查**

```python
def evict_expired_cache() -> int:
    """清理超过 TTL 的 MySQL 缓存条目及其 Milvus 向量。"""
    import time
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    with SessionLocal() as session:
        expired = session.query(QueryCacheStore).filter(
            QueryCacheStore.created_at +
            QueryCacheStore.ttl_seconds * QueryCacheStore.updated_at -
            QueryCacheStore.updated_at.replace(tzinfo=None)
        ).all()

        count = 0
        for entry in expired:
            # 计算 TTL
            age = (now - entry.updated_at).total_seconds()
            if age > entry.ttl_seconds:
                _milvus.init_cache_collection()
                _milvus.delete(collection_name=_milvus.CACHE_COLLECTION,
                               filter=f'query_hash == "{entry.query_hash}"')
                session.delete(entry)
                count += 1
        session.commit()
    return count
```

- [ ] **Step 2: Commit**

```bash
git add backend/cache/semantic_cache.py
git commit -m "feat: TTL-based cache eviction for expired entries"
```

---

## Phase 4: 压测脚本 + 环境配置

### Task 4.1: 并发压测与 Token 对比脚本

**Files:**
- Create: `scripts/run_benchmark.py`

- [ ] **Step 1: 编写并发压测脚本**

```python
#!/usr/bin/env python3
# scripts/run_benchmark.py
"""并发压测：对比缓存开启/关闭的 QPS 和 Token 消耗。"""
import time, json, sys, asyncio
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ".")

TEST_QUERIES = [
    "什么是 Ragent AI？",
    "系统用了什么向量数据库？",
    "如何配置 Neo4j？",
    "HITL 是什么？",
    "Leiden 算法怎么用？",
    "文档上传支持哪些格式？",
    "milvus 端口是多少？",
    "embedding 模型叫什么？",
    "Supervisor 怎么工作？",
    "什么是语义缓存？",
]

CONCURRENT_USERS = 5
REQUESTS_PER_USER = 2


def simulate_request(query: str) -> dict:
    """模拟单次请求（不走 SSE，只走检索）。"""
    from backend.rag.utils import retrieve_documents
    from backend.cache import query_cache

    t0 = time.time()
    cache_result = query_cache(query)
    if cache_result:
        return {"cached": True, "latency_ms": (time.time() - t0) * 1000, "tokens": 0}

    result = retrieve_documents(query)
    latency = (time.time() - t0) * 1000
    return {
        "cached": False,
        "latency_ms": latency,
        "docs": len(result.get("docs", [])),
        "tokens_estimated": sum(len(d.get("text", "")) for d in result.get("docs", [])) // 4,
    }


def run_benchmark():
    print(f"并发用户: {CONCURRENT_USERS}, 每用户请求: {REQUESTS_PER_USER}")
    total = CONCURRENT_USERS * REQUESTS_PER_USER

    with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        futures = []
        for i in range(CONCURRENT_USERS):
            for j in range(REQUESTS_PER_USER):
                query = TEST_QUERIES[(i * REQUESTS_PER_USER + j) % len(TEST_QUERIES)]
                futures.append(executor.submit(simulate_request, query))

        results = [f.result() for f in futures]

    cached = [r for r in results if r["cached"]]
    uncached = [r for r in results if not r["cached"]]

    print(f"\n===== 压测结果 =====")
    print(f"总请求: {total}")
    print(f"缓存命中: {len(cached)} ({len(cached)/total*100:.0f}%)")
    print(f"穿透 LLM: {len(uncached)} ({len(uncached)/total*100:.0f}%)")

    if cached:
        avg_lat = sum(r["latency_ms"] for r in cached) / len(cached)
        print(f"缓存命中平均延迟: {avg_lat:.0f}ms")
    if uncached:
        avg_lat = sum(r["latency_ms"] for r in uncached) / len(uncached)
        total_tokens = sum(r["tokens_estimated"] for r in uncached)
        print(f"穿透 LLM 平均延迟: {avg_lat:.0f}ms")
        print(f"估算 Token 消耗: {total_tokens}")

    total_latency = sum(r["latency_ms"] for r in results) / len(results)
    print(f"整体平均延迟: {total_latency:.0f}ms")
    print(f"缓存节省延迟: {(1 - total_latency / (sum(r['latency_ms'] for r in uncached)/len(uncached) if uncached else 1)) * 100:.0f}%")

    with open("benchmark_result.json", "w") as f:
        json.dump({
            "total_requests": total,
            "cache_hits": len(cached),
            "cache_misses": len(uncached),
            "avg_cached_latency_ms": sum(r["latency_ms"] for r in cached) / len(cached) if cached else 0,
            "avg_uncached_latency_ms": sum(r["latency_ms"] for r in uncached) / len(uncached) if uncached else 0,
        }, f, indent=2, ensure_ascii=False)
    print("结果已保存到 benchmark_result.json")


if __name__ == "__main__":
    run_benchmark()
```

- [ ] **Step 2: 验证脚本可运行**

```bash
uv run python scripts/run_benchmark.py
```

Expected: 输出压测统计。

- [ ] **Step 3: Commit**

```bash
git add scripts/run_benchmark.py
git commit -m "feat: concurrent benchmark script for cache hit/miss comparison"
```

---

### Task 4.2: 依赖更新 + 环境变量

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`

- [ ] **Step 1: pyproject.toml 追加依赖**

```toml
"llmlingua>=0.1.0",
"alembic>=1.13.0",
"numpy>=1.26.0",
```

- [ ] **Step 2: .env.example 追加配置**

```env
# ===== v6.0 Semantic Cache =====
CACHE_SIMILARITY_THRESHOLD=0.95
CACHE_TTL_SECONDS=86400
CACHE_ENABLED=true

# ===== v6.0 Model Routing =====
MODEL_TURBO=qwen-turbo
MODEL_MAX=qwen-max
CONTEXT_PRUNE_RATIO=0.6
```

- [ ] **Step 3: 安装依赖并验证**

```bash
uv sync
uv run python -c "
from backend.cache import query_cache, write_cache, with_singleflight, invalidate_by_filename
from backend.agent.model_router import get_model_for_agent
print('All v6.0 imports OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock .env.example
git commit -m "feat: add v6.0 dependencies and env config"
```

---

## 验收标准

### Phase 1 — 语义缓存
- [ ] 相同问题第二次请求命中缓存，返回 `cached: true`
- [ ] 缓存命中时跳过 RAG + LLM，延迟 < 200ms
- [ ] Redis 中缓存的响应可被其他实例读取

### Phase 2 — 动态路由
- [ ] Supervisor 使用 qwen-turbo 执行路由决策
- [ ] Direct Answer 使用 qwen-turbo 回答闲聊
- [ ] Data Analyst 可使用 qwen-max 处理复杂 SQL
- [ ] Context 超过 2000 字符时自动压缩 40%

### Phase 3 — 防击穿
- [ ] 10 个相同请求并发，只有 1 个穿透到 LLM
- [ ] 文档软删除后，关联缓存条目自动清除
- [ ] 过期缓存（超过 TTL）被自动清理

### Phase 4 — 压测
- [ ] `scripts/run_benchmark.py` 输出缓存命中率、延迟对比、Token 估算
- [ ] 缓存命中时 Token 消耗为 0

---

## 执行顺序

```
Phase 1 (Semantic Cache) ──► Phase 3 (Cache Protection)
                                      │
Phase 2 (Model Routing) ─────────────┤
                                      ▼
                              Phase 4 (Benchmark + Config)
```

Phase 1 和 Phase 2 可并行。Phase 3 依赖 Phase 1 的缓存层。Phase 4 在所有代码完成后统一配置。
