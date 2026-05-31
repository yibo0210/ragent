# Ragent AI 项目面试准备文档 — AI 应用开发实习

## 目录

1. [项目概述与定位](#1-项目概述与定位)
2. [技术架构全景](#2-技术架构全景)
3. [RAG 检索增强生成](#3-rag-检索增强生成)
4. [GraphRAG 知识图谱检索](#4-graphrag-知识图谱检索)
   - [4.6 本体约束图谱抽取 (v10)](#46-本体约束图谱抽取-v10)
   - [4.7 增量更新与异步管线 (v11)](#47-增量更新与异步管线-v11)
5. [多智能体编排系统](#5-多智能体编排系统)
6. [向量数据库 Milvus](#6-向量数据库-milvus)
7. [图数据库 Neo4j](#7-图数据库-neo4j)
8. [缓存与高可用](#8-缓存与高可用)
9. [可观测性体系](#9-可观测性体系)
10. [评测体系 (RAGAS)](#10-评测体系-ragas)
11. [自适应推理与自纠错 (v8)](#11-自适应推理与自纠错-v8)
12. [SSE 流式响应](#12-sse-流式响应)
13. [文档处理与层次化切片](#13-文档处理与层次化切片)
14. [HITL 人机协同](#14-hitl-人机协同)
15. [常见面试问题与回答](#15-常见面试问题与回答)

---

## 1. 项目概述与定位

### 1.1 一句话介绍

Ragent AI 是一个**企业级多智能体 GraphRAG 知识库助手**，基于 LangGraph Supervisor-Workers 架构，融合向量检索、知识图谱检索、联网搜索、数据分析等多种能力，通过 SSE 实时流式响应，支持 HITL 人机协同和自适应推理纠错。

### 1.2 解决了什么问题

传统 RAG 系统存在几个核心痛点：

1. **单一检索路径**：只能做向量相似度检索，无法处理多跳推理（"A 和 B 有什么关系？"）
2. **幻觉问题**：LLM 生成的回答可能编造文档中没有的信息，缺乏自检机制
3. **复杂查询拆解不足**：用户问一个需要综合多个文档的问题，系统无法自动拆解为多步执行
4. **缺乏可观测性**：检索过程是黑盒，无法追踪回答的来源和推理路径
5. **无评测闭环**：改了检索策略后无法量化评估效果变化

Ragent AI 通过以下方式解决：
- **GraphRAG**：向量检索 + 知识图谱外扩，支持多跳推理
- **Critique 自纠错**：LLM 交叉验证回答与检索依据，检测幻觉并触发重新检索
- **Planner 任务拆解**：复杂查询自动拆解为多步执行计划
- **全链路 Trace**：SSE 实时推送每一步检索/路由/生成事件，前端可视化
- **RAGAS 评测体系**：4 个量化指标 + A/B 对比 + HTML 报告

### 1.3 核心数据

- 6 个专用 Worker Agent
- 7 层 LangGraph 图节点（supervisor → planner → workers → synthesize → critique → replan → END）
- 3 层层次化切片（L1 1200字 / L2 600字 / L3 300字）
- 4 通道 RRF 融合（Dense + Sparse + Graph + Visual）
- 11 种本体实体类型，12 种关系谓词，70+ 条三元组规则（v10）
- 80 条 Golden Dataset，7 种查询类型
- 4 个 RAGAS 评测指标
- 5 种评测模式（retrieval/pipeline/e2e/graph/graph_compare）
- 10 个 Docker 服务一键拉起，含 API + Worker 双进程（v11）
- 文档 Hash 指纹 + 增量更新 + 异步队列（v11）

---

## 2. 技术架构全景

### 2.1 技术栈选型与理由

| 层级 | 技术 | 选型理由 |
|------|------|----------|
| Web 框架 | FastAPI | 异步原生支持，SSE 流式响应天然友好，自动 OpenAPI 文档 |
| Agent 编排 | LangGraph | 基于状态图的 Agent 编排，支持条件路由、并行 fan-out、中断恢复 |
| LLM | 通义千问 (DashScope) | 国内访问快，OpenAI 兼容接口，成本低 |
| 向量数据库 | Milvus 2.5 | 支持稠密+稀疏混合检索，动态 schema，生产级稳定性 |
| 图数据库 | Neo4j 5.26 | Cypher 查询语言，天然适合实体关系存储和多跳查询 |
| 关系数据库 | MySQL 8.0 | 会话/消息/文档索引持久化 |
| 缓存 | Redis 7.0 | 分布式锁（HITL）、对话缓存、父块热缓存 |
| 前端 | Vue 3 (CDN) | 零构建工具，单文件部署，响应式 |
| 可观测性 | OTel + Jaeger + Prometheus + Grafana | 全链路追踪 + 指标采集 + 看板展示 |
| 评测 | RAGAS + matplotlib | 自动化 RAG 质量评估 + 可视化报告 |

### 2.2 整体架构

```
用户提问
  │
  ▼
FastAPI (SSE 流式)
  │
  ▼
LangGraph Supervisor-Workers
  │
  ├── Supervisor (意图路由)
  │     ├── 直接路由 → Worker
  │     └── 复杂查询 → Planner → 多步 Worker
  │
  ├── Workers (并行 fan-out)
  │     ├── RAG Specialist (知识库检索)
  │     ├── Local Graph Search (图谱局部检索)
  │     ├── Global Graph Search (图谱全局检索)
  │     ├── Web Searcher (联网搜索)
  │     ├── Data Analyst (Text-to-SQL)
  │     └── Direct Answer (直接回答)
  │
  ├── Synthesize (多 Worker 结果聚合)
  │
  └── Critique (事实核查)
        ├── 通过 → END
        └── 驳回 → Replan → 重新路由 (max 2 retries)
```

### 2.3 数据存储架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Milvus    │    │    Neo4j    │    │    MySQL    │
│  向量数据库  │    │  图数据库    │    │  关系数据库  │
│             │    │             │    │             │
│ L3 叶子块   │    │ 实体节点    │    │ 会话表      │
│ Dense 向量  │    │ RELATES_TO  │    │ 消息表      │
│ Sparse 向量 │    │ source_chunks│   │ 父块表      │
│ 社区摘要    │    │ community_id│    │ 文档索引    │
│ 语义缓存    │    │ temporal    │    │ 图检查点    │
└─────────────┘    └─────────────┘    └─────────────┘
        │                │                  │
        └────────────────┼──────────────────┘
                         │
                    ┌─────────┐
                    │  Redis  │
                    │ 缓存/锁 │
                    └─────────┘
```

---

## 3. RAG 检索增强生成

### 3.1 RAG Pipeline 架构

RAG 流水线是一个独立的 LangGraph 状态机：

```
retrieve_initial → grade_documents → [conditional]
                                          │
                     ┌────────────────────┤
                     ▼                    ▼
              generate_answer      rewrite_question
                 (END)                   │
                                         ▼
                                 retrieve_expanded
                                         │
                                         ▼
                                 grade_after_expansion
                                         │
                                    ┌────┤
                                    ▼    ▼
                                  END   END (force_interrupt)
```

**关键设计思想**：
- **两轮评分**：初次检索后评分，不通过则查询重写 + 扩展检索 + 二次评分
- **HITL 兜底**：两次评分都不通过 → 触发人工介入中断
- **查询扩展策略**：Step-Back（退步提问）和 HyDE（假设性文档生成），由 LLM 自动选择

### 3.2 混合检索 (Hybrid Retrieval)

**三通道 RRF 融合**：

```python
RRF_Score = w1/(k+rank_dense) + w2/(k+rank_sparse) + w3/(k+rank_graph) + w4/(k+rank_visual)
```

- **Dense 向量**：Qwen text-embedding-v1 (1536 维)，语义相似度
- **Sparse 向量**：BM25 算法，关键词匹配
- **Graph 三元组**：Neo4j 实体关系检索
- **Visual 向量**：图片描述语义检索（v7 多模态）

权重通过环境变量配置，支持网格搜索优化。

**为什么需要混合检索？**
- 纯向量检索：语义理解强，但精确匹配弱（搜"Milvus 端口"可能返回不相关结果）
- 纯 BM25：关键词匹配强，但语义理解弱（搜"向量数据库"可能漏掉"embedding 存储"）
- 混合：互补优势，RRF 融合排序

### 3.3 Rerank 重排

检索后调用 qwen3-rerank API 对候选文档按相关性重排：

```python
# 自动检测：qwen 开头 → 原生 API，其他 → OpenAI 兼容端点
if RERANK_MODEL.startswith("qwen"):
    endpoint = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
else:
    endpoint = f"{RERANK_BINDING_HOST}/v1/rerank"
```

**设计思想**：Rerank 是检索质量的最后一道关卡。向量检索返回 Top-20 候选，Rerank 精排后取 Top-5，显著提升精确率。

### 3.4 自动合并 (Auto-Merging)

L3 叶子块检索后，如果多个 L3 块属于同一个 L2 父块，自动合并为 L2 父块返回：

```
L1 (1200字) ← 自动合并
  └── L2 (600字) ← 自动合并
        └── L3 (300字) ← 向量索引
```

**解决的问题**：碎片化检索。如果检索到一个 L2 块下的 3 个 L3 块，不如直接返回 L2 父块，上下文更完整。

### 3.5 查询扩展 (Query Expansion)

两种策略，由 LLM 自动选择：

1. **Step-Back Prompting**：将具体问题抽象为更高层次的问题
   - 用户问："AnomalyCLIP 的 zero-shot 检测精度是多少？"
   - Step-Back："AnomalyCLIP 的核心方法和性能表现如何？"
   - 扩大检索范围，提高召回率

2. **HyDE (Hypothetical Document Embedding)**：生成一个假设性文档
   - 用户问："什么是 RAG？"
   - HyDE 生成一段关于 RAG 的描述文本
   - 用这段文本做向量检索，语义对齐更精确

3. **Complex**：同时使用 Step-Back + HyDE，结果去重

---

## 4. GraphRAG 知识图谱检索

### 4.1 为什么需要 GraphRAG？

传统 RAG 的局限：
- **无法处理多跳推理**：问"A 和 B 有什么关系？"，需要先找到 A，再找 A 的关系边，再找 B
- **缺乏全局视角**：问"有哪些主要技术方向？"，需要综合所有文档的实体关系
- **上下文碎片化**：相关实体分散在不同文档的不同片段中

GraphRAG 通过知识图谱解决这些问题。

### 4.2 知识图谱构建流程

```
文档上传
  │
  ▼
层次化切片 (L1/L2/L3)
  │
  ▼
L2 中等粒度文本 → LLM 实体抽取
  │
  ▼
提取三元组 (subject, predicate, object)
  │
  ▼
Neo4j MERGE (实体 + 关系 + source_chunks)
  │
  ▼
离线: Leiden 社区聚类 → 社区摘要 → Milvus 索引
```

**实体抽取 Prompt 设计**：
- 从文本中提取所有实体（人名、组织、技术、概念）
- 提取实体间关系（"使用"、"依赖"、"创新"、"改进"）
- 每个关系绑定 source_chunks（L3 叶子块 ID），实现溯源

**Neo4j 存储结构**：
```cypher
// 实体节点
(:Entity {name: "AnomalyCLIP", type: "Model", description: "..."})

// 关系边
(:Entity {name: "AnomalyCLIP"}) -[:INNOVATES {weight: 1.0, source_chunks: ["chunk_001", "chunk_002"]}]-> (:Entity {name: "Zero-shot"})

// 社区
(:Entity {name: "AnomalyCLIP"}) -[:BELONGS_TO]-> (:Community {id: "c1"})
```

### 4.3 局部图谱检索 (Local Graph Search)

```
用户问题 → Milvus 向量检索 Top-K → 提取关联实体 → Neo4j 1-hop 外扩邻居
  │
  ▼
合并：向量检索文本 + 图谱三元组 → 上下文
```

**适用场景**：实体间关系查询、多跳推理
- "AnomalyCLIP 和 CLIP 有什么关系？"
- "Milvus 依赖哪些组件？"

### 4.4 全局图谱检索 (Global Graph Search)

```
用户问题 → Milvus 社区摘要向量匹配 → 返回最相关的社区综述
```

**适用场景**：总结性、全局性提问
- "知识库中有哪些主要技术方向？"
- "整体架构是怎样的？"

### 4.5 时序路由 (Temporal Routing)

Supervisor 检测到时间敏感查询时，设置 `is_temporal=true` 和 `temporal_year`，传递给 local_graph_search_node：

```cypher
// Cypher 查询中过滤时序
MATCH (e:Entity)-[r:RELATES_TO]->(t:Entity)
WHERE r.valid_from <= $year AND r.valid_to >= $year
RETURN e, r, t
```

**适用场景**："2023年的技术进展"、"最新的研究"

### 4.6 本体约束图谱抽取 (v10)

**问题**：自由抽取导致图谱冗余——同一概念多种类型（"BERT" 可能是 Model 也可能是 Technology），关系谓词不统一（"使用" vs "采用" vs "依赖"），孤岛节点多。

**解决方案**：引入领域本体约束层，将 LLM 发散抽取收敛为结构化填空。

**核心组件**：

```python
# backend/ontology/schema.py — 唯一真实来源
ENTITY_TYPES = ["Person", "Organization", "Technology", "Concept", "Model",
                "Method", "Data", "Product", "Event", "Document", "Metric"]  # 11 种

RELATION_PREDICATES = ["DEPENDS_ON", "CONTAINS", "CITES", "USES", "PART_OF",
                       "PROPOSES", "EVALUATES", "CAUSES", "IMPLEMENTS",
                       "BELONGS_TO", "COMPETES_WITH", "RELATED_TO"]  # 12 种

RELATION_RULES = [
    ("Technology", "DEPENDS_ON", "Technology"),
    ("Organization", "PROPOSES", "Model"),
    ("*", "CITES", "*"),  # 通配规则
    # ... 共 70+ 条
]
```

**两层防护机制**：

1. **Pydantic field_validator**（第一层）：`EntityInfo.type` 和 `RelationInfo.predicate` 在模型构建时自动归一化
   - `"company"` → `"Organization"`，`"tool"` → `"Technology"`，`"algorithm"` → `"Method"`
   - 兜底：未知类型 → `"Concept"`，未知谓词 → `"USES"`

2. **`_validate_extraction()` 拦截器**（第二层）：LLM 输出后、写入 Neo4j 前
   - 过滤 type 不在白名单的实体
   - 过滤 subject/object 不在本批实体中的关系
   - 调用 `is_valid_relation(type_s, predicate, type_o)` 校验三元组合法性

**Qwen 兼容性处理**：DashScope 的 `with_structured_output` 返回 `source`/`target` 而非 `subject`/`object`，解决方案是手动 JSON 解析 + 字段名映射。

**效果对比**：

| 指标 | 自由抽取 | 受控抽取 |
|------|---------|---------|
| 孤岛节点 | 23.5% | 0% |
| 平均度 | 1.18 | 1.62 |
| 类型一致性 | 混乱（中英文混杂） | 100% 白名单 |
| 谓词一致性 | 100+ 种自由谓词 | 12 种标准谓词 |

**面试话术**：
> "我在 v10 引入了本体约束层来解决图谱质量问题。核心思路是定义一个包含实体类型、关系谓词和三元组规则的 Schema，然后通过 Pydantic 验证器和后置拦截器两层防护，确保 LLM 输出符合预定义的结构。这把图谱从'自由抽取'变成了'受控填空'，孤岛率从 23.5% 降到了 0%。"

### 4.7 增量更新与异步管线 (v11)

**问题**：原来每次上传文档都走「全量删除 → 重新切片 → 重新向量化 → 重新抽取图谱」，即使内容没变也要等几十秒。大文件上传时 HTTP 请求超时。

**解决方案**：三层改造——文档指纹跳过、图谱增量清理、异步队列解耦。

**1. 文档指纹 (SHA-256)**

```python
# backend/documents/fingerprint.py
def compute_file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

上传时先算 Hash，查 `document_index` 表：
- Hash 一致 → 跳过，秒级返回 `status: "unchanged"`
- Hash 不同 → 标记 `is_deleted=False`，更新版本号，走增量管线

**2. 图谱增量清理**

```python
# 重新上传时，先清理该文件产生的旧图谱数据
cleanup_by_filename(filename)
# → strip_chunk_from_edges (从边的 source_chunks 中移除旧 chunk ID)
# → remove_empty_edges (删除 source_chunks 为空的边)
# → remove_orphan_entities (删除无任何边的孤岛节点)
```

**3. 异步任务队列 (arq + Redis)**

```
用户上传 → FastAPI 计算 Hash → 返回 HTTP 202 (job_id)
                ↓
        Redis 队列 (arq)
                ↓
        Worker 进程 (start_worker.py)
          ├── 清理旧数据
          ├── 切片 + 向量化
          ├── 图谱抽取 + 写入
          └── 更新 DocumentIndex
```

- Worker 独立于 HTTP 进程，有自己的 DB 初始化
- Redis 挂了自动降级为同步模式（不影响可用性）
- Worker 内存限制 4G（Docker Compose 资源限制）

**面试话术**：
> "v11 解决的是'更新效率'问题。我在三个层面做了改造：第一层是文档指纹，SHA-256 Hash 相同直接跳过，秒级返回；第二层是图谱增量清理，重新上传时先按文件名清理旧的边和孤岛节点再重建，避免图谱膨胀；第三层是用 arq + Redis 做异步队列，上传请求立即返回，后台 Worker 执行重活，Redis 挂了自动降级回同步。整个系统从'每次全量重建'变成了'有变化才更新'。"

---

## 5. 多智能体编排系统

### 5.1 为什么用多智能体？

单 Agent 的局限：
- 一个 LLM 要处理所有类型的查询（闲聊、检索、数据分析、联网搜索）
- Prompt 越来越长，效果越来越差
- 无法并行执行多种检索

多 Agent 的优势：
- **专业分工**：每个 Agent 专注一种能力，Prompt 更精准
- **并行执行**：多个 Worker 可以同时检索，减少总耗时
- **灵活路由**：Supervisor 根据意图选择最合适的 Worker

### 5.2 LangGraph 状态图

LangGraph 的核心思想是**用状态图描述 Agent 工作流**：

```python
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 对话历史
    next_worker: str           # 路由目标
    next_workers: list[str]    # 并行路由目标
    user_query: str            # 用户原始问题
    worker_outputs: dict       # 各 Worker 结果
    query_plan: Optional[dict] # Planner 计划
    critique_result: Optional[dict]  # Critique 结果
    retry_count: int           # 重试次数
    draft_answer: str          # 草稿答案
```

**状态图 vs 链式调用**：
- 链式调用（LangChain Chain）：A → B → C，线性，无法循环
- 状态图（LangGraph）：节点之间可以有条件路由、循环、并行，更灵活

### 5.3 Supervisor 路由

Supervisor 是整个系统的"大脑"，负责意图识别和路由分发：

```python
SUPERVISOR_SYSTEM_PROMPT = """你是一个智能路由调度员。
## 路由规则
- rag_specialist：知识库文档相关
- local_graph_search：实体关系、多跳推理
- global_graph_search：总结性、全局性提问
- web_searcher：实时信息、天气、新闻
- direct_answer：闲聊、通用知识
- data_analyst：数据查询、统计分析
"""
```

**路由实现**：手动 JSON 解析（不用 with_structured_output）
- 原因：Qwen 模型的 function_calling 与 LangChain 的 with_structured_output 不兼容
- 方案：LLM 输出 JSON → 正则提取 → json.loads 解析

### 5.4 Send 并行 Fan-Out

当 Supervisor 决定路由到多个 Worker 时，使用 LangGraph 的 `Send` 实现并行：

```python
def route_supervisor(state):
    workers = state["next_workers"]
    if len(workers) == 1:
        return workers[0]  # 单路由
    return [Send(worker, state) for worker in workers]  # 并行 fan-out
```

**适用场景**：用户问"AnomalyCLIP 的核心创新和联网搜索最新进展"
- Supervisor 同时路由到 rag_specialist + web_searcher
- 两个 Worker 并行执行
- synthesize 节点聚合结果

### 5.5 Synthesize 结果聚合

多 Worker 结果通过 LLM 聚合：

```python
synthesis_prompt = f"""你是一个信息整合专家。以下是多个智能体对同一问题的不同回答，
请将它们整合为一个条理清晰、内容完整的回答。

规则：
- 融合各来源信息，互补而非重复
- 如果各来源有矛盾，指出差异并给出综合判断
"""
```

### 5.6 Worker 实现示例

**RAG Specialist**：
```python
def rag_specialist_node(state):
    user_query = state["user_query"]
    rag_result = run_rag_graph(user_query)  # 调用 RAG Pipeline
    docs = rag_result["docs"]
    context = rag_result["context"]
    
    # HITL 中断检查
    if rag_result.get("force_interrupt"):
        interrupt({"type": "hitl_rag_grade", "query": user_query, ...})
    
    # LLM 生成回答
    prompt = f"{RAG_SPECIALIST_PROMPT}\n\n## 检索到的文档\n\n{context}\n\n## 用户问题\n\n{user_query}"
    answer = _stream_answer(model, [HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=answer)], "rag_trace": rag_result["rag_trace"]}
```

---

## 6. 向量数据库 Milvus

### 6.1 为什么选 Milvus？

| 对比维度 | Milvus | FAISS | Pinecone | Weaviate |
|----------|--------|-------|----------|----------|
| 稀疏向量 | 原生支持 | 不支持 | 支持 | 支持 |
| 混合检索 | 原生支持 | 需要自建 | 支持 | 支持 |
| 动态 schema | 支持 | 不支持 | 支持 | 支持 |
| 分布式 | 支持 | 不支持 | 云服务 | 支持 |
| 开源 | 是 | 是 | 否 | 是 |
| 生产稳定性 | 高 | 中 | 高 | 中 |

选择 Milvus 的核心原因：**原生支持稠密+稀疏混合检索**，无需自建 RRF 逻辑。

### 6.2 Collection 设计

```python
collection_schema = {
    "fields": [
        {"name": "id", "type": "VARCHAR", "is_primary": True},
        {"name": "text", "type": "VARCHAR"},
        {"name": "filename", "type": "VARCHAR"},
        {"name": "file_type", "type": "VARCHAR"},
        {"name": "chunk_id", "type": "VARCHAR"},
        {"name": "chunk_level", "type": "INT64"},
        {"name": "dense_vector", "type": "FLOAT_VECTOR", "dim": 1536},
        {"name": "sparse_vector", "type": "SPARSE_FLOAT_VECTOR"},
    ],
    "enable_dynamic_field": True  # 支持动态字段
}
```

**关键设计**：
- `enable_dynamic_field=True`：社区摘要和文档块共用同一个 Collection，通过动态字段区分
- 稠密+稀疏双索引：HNSW（稠密）+ SPARSE_INVERTED_INDEX（稀疏）

### 6.3 混合检索实现

```python
def hybrid_retrieve(dense_vec, sparse_vec, top_k=20):
    results = client.search(
        collection_name="ragent",
        data=[dense_vec, sparse_vec],
        anns_field=["dense_vector", "sparse_vector"],
        search_params={"metric_type": "IP"},
        limit=top_k,
        output_fields=["text", "filename", "chunk_id"],
    )
    return results
```

### 6.4 gRPC 重连机制

Milvus 的 gRPC 连接可能断开，需要重连保护：

```python
def _ensure_connected(self):
    try:
        self.client.get_load_state("ragent")
    except Exception:
        self.client = MilvusClient(uri=f"http://{HOST}:{PORT}")
```

---

## 7. 图数据库 Neo4j

### 7.1 为什么用图数据库？

关系型数据库存储实体关系的局限：
- 多跳查询需要多次 JOIN，性能差
- 关系类型不灵活，新增关系需要改表结构
- 无法天然支持图算法（社区发现、中心性分析）

Neo4j 的优势：
- Cypher 查询语言，多跳查询一行搞定
- 灵活的 schema，随时新增节点类型和关系类型
- 内置图算法库

### 7.2 Cypher 查询示例

**实体外扩（1-hop）**：
```cypher
MATCH (e:Entity)-[r:RELATES_TO]-(neighbor:Entity)
WHERE e.name = $entity_name
RETURN e.name, r.predicate, neighbor.name, r.weight
ORDER BY r.weight DESC
LIMIT 10
```

**带 source_chunks 溯源**：
```cypher
MATCH (e:Entity)-[r:RELATES_TO]->(t:Entity)
WHERE e.name IN $entity_names
RETURN e.name, r.predicate, t.name, r.source_chunks
```

**时序过滤**：
```cypher
MATCH (e:Entity)-[r:RELATES_TO]->(t:Entity)
WHERE r.valid_from <= $year AND r.valid_to >= $year
RETURN e, r, t
```

### 7.3 MERGE 去重策略

```cypher
// 实体去重：name 唯一约束
MERGE (e:Entity {name: $name})
ON CREATE SET e.type = $type, e.description = $description
ON MATCH SET e.description = CASE 
    WHEN size($description) > size(e.description) THEN $description 
    ELSE e.description END

// 关系 upsert：weight 取最大值
MERGE (s:Entity {name: $subject})-[r:RELATES_TO {predicate: $predicate}]->(o:Entity {name: $object})
ON CREATE SET r.weight = $weight, r.source_chunks = $chunks
ON MATCH SET r.weight = CASE WHEN $weight > r.weight THEN $weight ELSE r.weight END
```

### 7.4 级联清理

文档删除时需要清理 Neo4j 中的相关数据：

```python
def full_cascade_cleanup(chunk_ids):
    # 1. 剥离边上的 source_chunks 引用
    strip_chunk_from_edges(chunk_ids)
    # 2. 删除空边（source_chunks 为空）
    remove_empty_edges()
    # 3. 删除孤立实体（无任何关系）
    remove_orphan_entities()
```

---

## 8. 缓存与高可用

### 8.1 语义缓存

**思想**：如果用户问的问题与之前某个问题语义相似（cosine ≥ 0.95），直接返回缓存答案，跳过 RAG 流程。

```python
def query_cache(query):
    query_vec = embedding_service.get_embeddings([query])[0]
    results = milvus.search(collection_name="semantic_cache", data=[query_vec], limit=1)
    if results[0]["distance"] >= CACHE_SIMILARITY_THRESHOLD:
        return {"response": results[0]["entity"]["response_text"], "similarity": results[0]["distance"]}
    return None
```

**存储**：Milvus 存向量 + MySQL 存原文（TTL 过期、命中计数）

### 8.2 Singleflight 防缓存击穿

**问题**：高并发下，同一个 query 可能同时穿透缓存，导致多个 RAG 请求同时执行。

**解决**：Redis 分布式锁，同一 key 只允许一个请求执行，其他请求等待结果：

```python
def with_singleflight(key_prefix):
    def decorator(func):
        def wrapper(*args, **kwargs):
            lock_key = f"singleflight:{key_prefix}:{hash(args)}"
            if redis.set(lock_key, "1", nx=True, ex=30):
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    redis.delete(lock_key)
            else:
                # 等待其他请求完成
                time.sleep(0.5)
                return cache.get(result_key)
        return wrapper
    return decorator
```

### 8.3 熔断器

**思想**：如果 LLM API 连续失败 3 次，熔断 60 秒，期间直接返回降级响应。

```python
class CircuitBreaker:
    def __init__(self, name, failure_threshold=3, recovery_timeout=60):
        self.state = "CLOSED"  # CLOSED → OPEN → HALF_OPEN
        self.failure_count = 0
    
    def call(self, func, fallback):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                return fallback()
        try:
            result = func()
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise
```

### 8.4 降级策略

Neo4j 查询超时 → 自动降级为纯向量检索：

```python
def safe_graph_search(query):
    try:
        return local_graph_search(query)
    except Exception:  # Neo4j 超时
        logger.warning("Neo4j timeout, falling back to dense+sparse")
        return retrieve_documents(query)  # 降级
```

### 8.5 重试机制

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def call_llm(prompt):
    return model.invoke(prompt)
```

指数退避：1s → 2s → 4s

---

## 9. 可观测性体系

### 9.1 三层可观测性

| 层级 | 工具 | 用途 |
|------|------|------|
| 链路追踪 | OpenTelemetry + Jaeger | 请求全链路 Span，定位慢节点 |
| 指标采集 | Prometheus | LLM Token 用量、路由分布、延迟直方图、熔断器状态 |
| 日志 | structlog | 结构化 JSON 日志，ELK/Loki 友好 |

### 9.2 手动 Span

```python
tracer = get_tracer("ragent.agent")

def local_graph_search_node(state):
    with tracer.start_as_current_span("agent.local_graph_search") as span:
        span.set_attribute("query", user_query[:200])
        result = safe_graph_search(user_query)
        span.set_attribute("triples_count", len(result["graph_triples"]))
```

**为什么手动 Span 而不是自动？**
- FastAPI 自动 Span 无法覆盖 LangGraph 内部节点
- 需要在每个 Agent 节点、Milvus 查询、Neo4j Cypher 调用处手动埋点

### 9.3 Prometheus 指标

```python
class Metrics:
    _llm_tokens = Counter("llm_tokens_total", "Total LLM tokens", ["model", "direction"])
    _routing_count = Counter("agent_routing_total", "Agent routing count", ["agent"])
    _vector_latency = Histogram("vector_search_latency_seconds", "Vector search latency")
    _circuit_breaker_state = Gauge("circuit_breaker_state", "Circuit breaker state", ["name"])
```

---

## 10. 评测体系 (RAGAS)

### 10.1 为什么需要评测？

改了检索策略（RRF 权重、Rerank 模型、切片参数）后，如何量化效果变化？
- 主观评估："感觉更好了"不可靠
- 自动化评测：RAGAS 指标量化，A/B 对比

### 10.2 RAGAS 四个指标

| 指标 | 衡量什么 | 计算方式 |
|------|----------|----------|
| Context Precision | 检索到的上下文有多少是有用的 | LLM 判断每个检索片段是否相关 |
| Context Recall | 标准答案需要的信息是否被检索到 | LLM 判断标准答案的每个声明是否在上下文中 |
| Faithfulness | 生成的回答是否忠于检索到的上下文 | LLM 拆解回答为声明，逐条验证 |
| Answer Relevancy | 回答与问题的相关程度 | LLM 反向生成问题，计算语义相似度 |

### 10.3 三种评测模式

```bash
# 模式 1: 仅检索质量（answer = ground_truth，不调 LLM 生成）
python scripts/run_evaluation.py --mode retrieval --limit 10

# 模式 2: 完整 RAG Pipeline（走 run_rag_graph，answer = ground_truth）
python scripts/run_evaluation.py --mode pipeline --limit 10

# 模式 3: 端到端（LLM 真实生成 answer + 路由准确率 + 延迟统计）
python scripts/run_evaluation.py --mode e2e --limit 10
```

### 10.4 Golden Dataset

80 条 QA 对，覆盖 7 种查询类型：

| 类型 | 数量 | 期望 Agent | 示例 |
|------|------|-----------|------|
| conceptual | 10 | rag_specialist | "什么是 GraphRAG？" |
| detail | 40 | rag_specialist | "Milvus 端口是多少？" |
| cross_doc | 8 | local_graph_search | "Milvus 和 Neo4j 的关系？" |
| global_summary | 7 | global_graph_search | "系统有哪些技术栈？" |
| realtime | 5 | web_searcher | "今天天气怎么样？" |
| chat | 5 | direct_answer | "你好" |
| data_query | 5 | data_analyst | "有多少条会话？" |

### 10.5 A/B 对比

```bash
# 两次评测结果对比
python scripts/run_evaluation.py --compare baseline.json experiment.json
```

输出 diff 表格，标注每个指标的提升/下降百分比。

---

## 11. 自适应推理与自纠错 (v8)

### 11.1 设计思想

传统 RAG 是**单向执行**：检索 → 生成 → 输出。如果生成的回答有幻觉（编造了文档中没有的信息），系统不会自检。

v8 引入**自省能力**：
1. **Planner**（前置推理）：复杂查询自动拆解为多步执行计划
2. **Critique**（后置反思）：LLM 交叉验证回答与检索依据，检测幻觉
3. **Replan**（自纠错）：Critique 驳回后，提取缺失信息，重新检索

### 11.2 Planner 节点

```python
PLANNER_PROMPT = """你是一个任务规划专家。分析用户问题，判断是否需要多步执行。
- 简单问题：返回 is_complex=false
- 复杂问题：拆解为 2-4 个步骤，每个步骤指定 agent 和子查询
- 可用 agent: rag_specialist, local_graph_search, global_graph_search, web_searcher
"""
```

**路由逻辑**：
- Supervisor 判断为复杂查询 → 路由到 Planner
- Planner 生成 QueryPlan JSON → 按步骤 Send 到各 Worker
- 简单查询 → 直接路由到 Worker，跳过 Planner

### 11.3 Critique 节点

```python
CRITIQUE_PROMPT = """你是一个严格的事实核查专家。检查以下回答是否完全基于提供的上下文。
- 逐条检查回答中的事实声明
- 每个声明必须能在上下文中找到直接依据
- 如果有声明无法验证，标记 is_valid=false
"""
```

**输出**：`CritiqueResult {is_valid, missing_information, feedback, confidence}`

### 11.4 自纠错循环

```
synthesize → critique
                │
        ┌───────┤
        ▼       ▼
      valid   invalid, retry<2
        │       │
        ▼       ▼
       END    replan → supervisor (重新路由)
```

**防死循环**：最大重试 2 次。超过后降级输出当前最佳答案。

### 11.5 direct_answer 跳过 Critique

闲聊回答没有检索上下文，Critique 必然判"依据不足"，触发无意义重试。因此 `direct_answer` 直接到 END。

---

## 12. SSE 流式响应

### 12.1 为什么用 SSE 而不是 WebSocket？

| 对比 | SSE | WebSocket |
|------|-----|-----------|
| 方向 | 服务端 → 客户端（单向） | 双向 |
| 协议 | HTTP | 独立协议 |
| 复杂度 | 低 | 高 |
| 重连 | 浏览器自动重连 | 需要手动实现 |
| 适用场景 | 服务端推送（流式响应） | 实时双向通信 |

SSE 天然适合 LLM 流式输出场景：服务端逐 token 推送，客户端实时渲染。

### 12.2 SSE 事件协议

```python
# 事件类型
"agent_start"       # Agent 开始执行
"agent_done"        # Agent 执行完成
"routing"           # 路由决策
"rag_step"          # RAG 检索步骤
"graph_expand"      # 图谱外扩
"community_match"   # 社区摘要匹配
"content"           # 回答内容（逐 token）
"worker_content"    # Worker 完成内容
"plan_generated"    # Planner 生成计划 (v8)
"critique_feedback" # Critique 反馈 (v8)
"self_correction"   # 自纠错循环 (v8)
"trace"             # RAG 追踪
"agent_trace"       # Agent 追踪
"hitl_interrupt"    # HITL 中断
"error"             # 错误
"[DONE]"            # 结束
```

### 12.3 实现方式

```python
# 后端：asyncio.Queue + StreamingResponse
async def chat_with_agent_stream(user_text, session_id):
    output_queue = asyncio.Queue()
    
    async def _graph_worker():
        async for event in graph.astream(..., stream_mode="updates"):
            for node_name, update in event.items():
                await output_queue.put({"type": "...", ...})
        await output_queue.put(None)  # 哨兵
    
    agent_task = asyncio.create_task(_graph_worker())
    
    while True:
        event = await output_queue.get()
        if event is None:
            break
        yield f"data: {json.dumps(event)}\n\n"
    
    yield "data: [DONE]\n\n"
```

```javascript
// 前端：fetch + ReadableStream
const reader = response.body.getReader();
while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // 解析 SSE 事件...
}
```

---

## 13. 文档处理与层次化切片

### 13.1 为什么需要层次化切片？

固定长度切片的问题：
- 切片太小：上下文碎片化，检索到的片段缺乏完整语义
- 切片太大：向量检索精度下降，噪声多

层次化切片的解决方案：
- L3 (300字)：向量索引，精确检索
- L2 (600字)：图谱抽取，中等粒度
- L1 (1200字)：自动合并，完整上下文

### 13.2 切片参数

| 层级 | 字符数 | 重叠 | 用途 |
|------|--------|------|------|
| L1 | 1200 | 240 | 根级块，自动合并目标 |
| L2 | 600 | 120 | 中级块，图谱抽取 |
| L3 | 300 | 60 | 叶子块，向量索引 |

### 13.3 支持的文档格式

PDF、Word (docx/doc)、Excel (xlsx/xls)、Markdown、图片 (png/jpg/gif/webp/bmp)

---

## 14. HITL 人机协同

### 14.1 两种触发场景

**场景 A：RAG 低置信度**
- RAG Pipeline 两次评分都不通过
- 触发 interrupt()，等待人工决策
- 选项：批准（使用当前结果）/ 修改查询 / 终止

**场景 B：非 SELECT SQL**
- Data Analyst 生成了 INSERT/UPDATE/DELETE 语句
- 触发 interrupt()，等待人工审批
- 选项：批准执行 / 终止

### 14.2 实现机制

```python
# 中断
from langgraph.types import interrupt
interrupt({"type": "hitl_rag_grade", "query": user_query, ...})

# 恢复
from langgraph.types import Command
graph.invoke(Command(resume={"action": "approve"}), config=config)
```

**并发控制**：Redis 分布式锁，中断期间阻止同一 session 的新请求（HTTP 423）。

---

## 15. 常见面试问题与回答

### Q1: 介绍一下你的项目？

**回答**：Ragent AI 是一个企业级多智能体 GraphRAG 知识库助手。核心架构是 LangGraph Supervisor-Workers 模式，Supervisor 做意图路由，6 个 Worker Agent 并行执行不同任务。检索层融合了向量检索（Milvus）、知识图谱检索（Neo4j）、联网搜索（Tavily）三种能力，通过 RRF 融合排序。v8 版本新增了 Planner（复杂查询拆解）和 Critique（事实核查自纠错），使系统具备自省能力。全链路 SSE 流式响应，前端实时展示推理过程。

### Q2: RAG 和 GraphRAG 的区别？你为什么选择 GraphRAG？

**回答**：
- **传统 RAG**：向量检索 → LLM 生成。优点是简单高效，缺点是无法处理多跳推理和全局性问题。
- **GraphRAG**：在 RAG 基础上增加知识图谱检索。通过实体关系网络支持多跳推理（"A 和 B 什么关系？"）和全局摘要（"有哪些主要技术？"）。

选择 GraphRAG 的原因：企业知识库中的信息往往是关联的（技术依赖、人物关系、因果链），纯向量检索无法捕捉这些关联。GraphRAG 通过 Neo4j 存储实体关系，支持 1-hop 外扩和社区摘要，补足了向量检索的盲区。

### Q3: 你的 RAG Pipeline 是怎么设计的？

**回答**：
1. **检索**：Dense + Sparse + Graph 三通道 RRF 融合，Rerank 精排，Auto-Merging 合并碎片
2. **评分**：LLM 判断检索结果与问题的相关性（binary yes/no）
3. **重写**：评分不通过 → Step-Back/HyDE 查询扩展 → 扩展检索 → 二次评分
4. **兜底**：两次评分都不通过 → HITL 人工介入

关键设计思想：**两轮评分 + 查询扩展**，确保检索质量。如果初次检索不理想，通过查询重写扩大检索范围。

### Q4: 多智能体是怎么编排的？为什么用 LangGraph？

**回答**：
- **编排方式**：LangGraph StateGraph，状态机描述 Agent 工作流
- **路由**：Supervisor LLM 做意图识别，返回路由目标列表
- **并行**：多个 Worker 通过 LangGraph `Send` fan-out 并行执行
- **聚合**：Synthesize 节点 LLM 聚合多 Worker 结果

选择 LangGraph 的原因：
1. **支持循环**：v8 的 Critique → Replan → Supervisor 循环，LangChain Chain 无法实现
2. **状态管理**：TypedDict 定义全局状态，节点之间通过状态通信
3. **中断恢复**：原生支持 interrupt() + Command(resume=...)，HITL 实现简单
4. **可视化**：LangGraph Studio 可以直观查看图拓扑

### Q5: Milvus 混合检索是怎么实现的？

**回答**：
Milvus 原生支持稠密+稀疏双通道检索：
- 稠密向量：Qwen text-embedding-v1 (1536维)，HNSW 索引
- 稀疏向量：BM25 算法，SPARSE_INVERTED_INDEX

检索时两个通道同时查询，Milvus 内部做 RRF 融合排序。外部还可以叠加 Graph 通道（Neo4j 三元组）和 Visual 通道（图片描述），通过 `rrf_fusion_three_channel` 函数做 4 通道加权融合。

### Q6: 知识图谱是怎么构建的？

**回答**：
1. 文档上传 → 层次化切片（L1/L2/L3）
2. L2 中等粒度文本 → LLM 实体/关系抽取（v10 后为受控抽取）
3. 提取三元组 (subject, predicate, object)
4. Neo4j MERGE：实体按 name 去重，关系按 (subject, predicate, object) 去重，weight 取最大值
5. 每个关系绑定 source_chunks（L3 叶子块 ID），实现从图谱到原文的溯源
6. 离线：Leiden 社区聚类 → LLM 生成社区摘要 → 向量化存入 Milvus

**v10 改进**：引入本体约束层解决图谱质量问题。定义 11 种实体类型和 12 种关系谓词的白名单，通过 Pydantic 验证器自动归一化 LLM 输出（如 "company" → "Organization"），再用后置拦截器校验三元组合法性（如 "Person CAUSES Metric" 会被过滤）。实体消歧也增加了类型约束，只在同类型实体间比较编辑距离，避免 "Apple"(公司) 和 "apple"(水果) 误合并。改造后图谱孤岛率从 23.5% 降到 0%。

### Q7: 如何解决 LLM 幻觉问题？

**回答**：
1. **Prompt 约束**：RAG Specialist Prompt 明确要求"基于提供的上下文回答，不要编造"
2. **RAG Pipeline 评分**：两轮评分确保检索质量
3. **v8 Critique**：LLM 交叉验证回答与检索依据，逐条检查事实声明是否有上下文支撑
4. **HITL 兜底**：低置信度时触发人工介入

### Q8: 缓存策略是怎么设计的？

**回答**：
三层缓存：
1. **语义缓存**（Milvus）：query 向量相似度 ≥ 0.95 → 直接返回缓存答案，跳过 RAG
2. **对话缓存**（Redis）：会话消息历史缓存，减少 MySQL 查询
3. **父块热缓存**（Redis）：自动合并时频繁访问的 L1/L2 块缓存

防击穿：Singleflight 模式，同一 query 只允许一个请求执行 RAG，其他等待结果。

### Q9: 如何保证系统的高可用？

**回答**：
1. **熔断器**：LLM API 连续失败 3 次 → 熔断 60 秒 → 返回降级响应
2. **重试**：tenacity 指数退避（1s→2s→4s），最多 3 次
3. **降级**：Neo4j 超时 → 自动降级为纯向量检索
4. **超时控制**：每个 LLM 调用设置 60 秒超时

### Q10: 评测体系是怎么做的？

**回答**：
基于 RAGAS 框架，4 个指标：context_precision、context_recall、faithfulness、answer_relevancy。
- Golden Dataset：80 条 QA 对，7 种查询类型
- 三种评测模式：retrieval（仅检索）、pipeline（完整 RAG）、e2e（端到端）
- A/B 对比：调参前后两次评测结果 diff
- 路由准确率：Supervisor 路由 vs expected_agent 对比
- 延迟统计：per-question / p50 / p95 / max

### Q11: v8 的自纠错循环是怎么实现的？

**回答**：
1. Synthesize 生成草稿答案，保存到 state.draft_answer
2. Critique 节点提取草稿和检索上下文，调用 LLM 逐条验证事实声明
3. 如果 is_valid=true → END
4. 如果 is_valid=false 且 retry<2 → Replan 节点提取 missing_information，构建补充查询，重新路由到 Supervisor
5. Supervisor 根据补充查询重新选择 Worker 执行
6. 最大重试 2 次，超过后降级输出当前最佳答案

### Q12: SSE 流式响应是怎么实现的？

**回答**：
- 后端：LangGraph astream() 生成事件 → asyncio.Queue 缓存 → StreamingResponse yield SSE
- 前端：fetch + ReadableStream 读取 → 按 \n\n 分割 → JSON.parse 解析 → 按 type 分发渲染
- 事件类型：routing、agent_start/done、rag_step、content（逐 token）、plan_generated、critique_feedback 等

### Q12.5: v10 的本体约束抽取是怎么设计的？

**回答**：

**问题背景**：自由抽取的图谱存在三个问题——类型混乱（"BERT" 可能是 Model 也可能是 Technology）、谓词不统一（"使用" vs "采用"）、孤岛节点多。

**解决方案**：三层防护

1. **Prompt 约束**：在 EXTRACTION_PROMPT 中明确列出所有 11 种实体类型和 12 种关系谓词，指令"只能使用上述类型，不得发明新类型"

2. **Pydantic 验证器**：`EntityInfo.type` 添加 `@field_validator`，内置归一化映射表：
   - `"company"` / `"公司"` → `"Organization"`
   - `"tool"` / `"工具"` → `"Technology"`
   - `"algorithm"` / `"算法"` → `"Method"`
   - 未知类型兜底 → `"Concept"`

3. **后置拦截器**：`_validate_extraction()` 在 LLM 输出后、Neo4j 写入前执行：
   - 过滤 type 不在白名单的实体
   - 过滤 subject/object 不在本批实体中的关系（防止幻觉实体）
   - 调用 `is_valid_relation(s_type, predicate, o_type)` 校验三元组合法性

**Qwen 兼容性**：DashScope 的 `with_structured_output` 返回 `source`/`target` 而非 `subject`/`object`，改用手动 JSON 解析 + 字段映射。

**效果**：孤岛率 23.5% → 0%，平均度 1.18 → 1.62，所有类型和谓词 100% 在白名单内。

### Q12.6: v11 的增量更新是怎么实现的？

**回答**：

**问题背景**：原来每次上传文档都走「全量删除 → 重新切片 → 重新向量化 → 重新抽取图谱」，即使内容没变也要等几十秒。大文件上传时 HTTP 请求超时。

**三层改造**：

1. **文档指纹跳过**：上传时先算 SHA-256 Hash，查 `document_index` 表。Hash 一致直接返回 `status: "unchanged"`，秒级完成。这一步省掉了 90% 的重复计算。

2. **图谱增量清理**：内容变化时，先按文件名清理旧图谱数据——从边的 `source_chunks` 数组中移除旧 chunk ID，删除变空的边，删除无连接的孤岛节点。然后重建新数据。这避免了图谱随重复上传不断膨胀。

3. **异步队列解耦**：用 `arq`（Redis-backed）把重活（切片、向量化、图谱抽取）推到后台 Worker 进程，HTTP 请求立即返回 `status: "queued"`。Worker 有自己的 DB 初始化，不依赖 FastAPI 生命周期。Redis 挂了自动降级回同步模式。

**效果**：同文件重传从几十秒降到秒级，图谱不再因重复上传膨胀，HTTP 请求不再因大文件超时。

### Q13: 如果让你优化这个系统，你会怎么做？

**回答**（参考计划文档）：
1. **检索优化**：引入 RAPTOR（递归摘要树）提升长文档检索质量
2. **生成优化**：使用 Chain-of-Thought 或 Tree-of-Thought 提升复杂推理能力
3. **多模态**：图片/表格理解，视觉问答
4. **个性化**：用户画像 + 检索偏好学习
5. **部署优化**：模型量化、KV Cache、推测解码
6. **评测增强**：更多评测指标（answer correctness、hallucination rate）、自动化回归测试

### Q14: 你在这个项目中遇到的最大挑战是什么？

**回答**：
1. **DashScope API 兼容性**：RAGAS 的 prompt 格式与 DashScope 不兼容，部分指标返回 NaN。解决：降级到 RAGAS 0.2.x，核心指标可用
2. **Qwen with_structured_output 不兼容**：LangChain 的结构化输出与 Qwen 的 thinking 模式冲突。解决：手动 JSON 正则解析
3. **Critique 对闲聊过度校验**：direct_answer 没有检索上下文，Critique 必然判"依据不足"。解决：direct_answer 跳过 Critique
4. **Milvus gRPC 连接断开**：长时间空闲后连接失效。解决：_ensure_connected() 每次查询前检查连接状态

### Q15: 你对 RAG 的理解是什么？未来趋势？

**回答**：
RAG = Retrieval Augmented Generation，检索增强生成。核心思想是**让 LLM 基于外部知识回答问题**，而不是依赖参数记忆。

**RAG 演进**：
1. **Naive RAG**：简单检索 + 生成
2. **Advanced RAG**：查询重写、混合检索、Rerank、Self-RAG
3. **Modular RAG**：模块化设计，可插拔组件
4. **GraphRAG**：知识图谱 + 向量检索，支持多跳推理
5. **Agentic RAG**：多智能体协作，自适应推理（本项目）

**未来趋势**：
- 长上下文模型（如 100K+ tokens）可能替代部分 RAG 场景
- 多模态 RAG（图片、视频、音频）
- 端到端训练的 RAG 模型（如 RETRO）
- Agent-native RAG：RAG 作为 Agent 的工具，而非独立系统

---

## 附录：项目亮点总结（面试用）

1. **完整的 RAG 系统**：混合检索 + Rerank + Auto-Merging + 查询扩展，不是简单 demo
2. **GraphRAG 创新**：向量 + 知识图谱融合，支持多跳推理和全局检索
3. **本体约束抽取**：领域 Schema + 两层防护（Pydantic 验证器 + 拦截器），图谱孤岛率 23.5% → 0%
4. **增量更新管线**：SHA-256 指纹跳过 + 图谱增量清理 + 异步队列，从全量重建到有变化才更新
5. **多智能体编排**：LangGraph Supervisor-Workers，支持并行 fan-out 和条件路由
6. **自纠错机制**：Planner + Critique + Replan，LLM 自省能力
7. **全链路可观测**：OTel + Prometheus + Grafana，不是黑盒
8. **自动化评测**：RAGAS + Golden Dataset + A/B 对比 + 图谱拓扑统计，用数据说话
9. **生产级设计**：熔断、降级、重试、缓存、HITL、异步队列，不是 toy project
10. **容器化部署**：Docker Compose 10 服务一键拉起，API + Worker 双进程，资源限制
