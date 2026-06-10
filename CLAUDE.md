# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Ragent AI — Project Context

Enterprise Multi-Agent GraphRAG Knowledge Base Assistant.

### Stack

FastAPI · LangChain · LangGraph · Pydantic · SQLAlchemy · Vue 3 (CDN)
Milvus 2.5 · Neo4j 5.26 · MySQL 8.0 · Redis 7.0
Qwen (DashScope OpenAI-compatible) · text-embedding-v1 (1536-dim) · BM25
qwen3-rerank (DashScope native API, model name starts with "qwen" auto-switches endpoint)
NetworkX · python-louvain
MCP (Model Context Protocol) · Echarts · aiohttp

### Architecture

**Supervisor-Workers** (LangGraph): supervisor routes to 6 agents — `rag_specialist`, `local_graph_search`, `global_graph_search`, `web_searcher`, `data_analyst`, `direct_answer`. Multi-worker via `Send` fan-out, merged by `synthesize`. v8: `planner` (前置推理拆解复杂查询) → workers → `synthesize` → `critique` (事实核查) → END/replan (自纠错循环, max 2 retries). `direct_answer` 和 `data_analyst` 直接到 END，跳过 Critique（闲聊无检索上下文、SQL 查询结果是结构化数据，均非 RAG 检索上下文）。v9: `data_analyst` 支持 MCP 外部数据源，自动发现并调用 MCP 工具。

**RAG Pipeline** (separate LangGraph): `retrieve → grade → [rewrite → retrieve_expanded → grade_v2]`. L1(1200)/L2(600)/L3(300) chunking. Leaf-only Milvus indexing. Auto-merge L3→L2→L1.

**GraphRAG**: Upload → L2 chunks → LLM extraction → Neo4j MERGE (entity + relation + source_chunks). Offline: `scripts/run_community_clustering.py` → Leiden → summaries → Milvus + MySQL.

**SSE Streaming**: `routing`, `agent_start/done`, `rag_step`, `graph_expand`, `community_match`, `content`, `trace`, `agent_trace`, `hitl_interrupt`, `error`. v8 新增: `plan_generated`, `critique_feedback`, `self_correction`. v9 新增: `mcp_tool_call`, `mcp_tool_result`. v12 新增: `query_profiler`, `system_state`.

v13: 增量图聚类引擎 — 文档摄入后自动触发局部补丁（新节点归入邻居多数社区）或子图重构（桥接多社区时仅对局部子图运行 Louvain），替代全量 Louvain 重算。脏位驱动的定向摘要生成（is_dirty 标记），只对受影响社区重新生成 LLM 摘要，Token 成本降 90%+。Redis Streams 消息总线（doc_ingest → graph_extract → vector_sync 三阶段管线）替代 arq 单队列。

**HITL**: LangGraph `interrupt()` (scenario A: low confidence RAG, scenario B: non-SELECT SQL). Redis lock → HTTP 423 during pending. Resume via `Command(resume=...)`.

**Multi-Tenant RBAC** (v14): OAuth2/JWT 鉴权中间件提取 `tenant_id` + `user_id` + `role`。`SupervisorState.user_context` 透传给所有 Worker。Milvus pre-filtering (`expr = "tenant_id == X"`) 实现向量通道硬隔离。Neo4j MERGE key 扩展为 `{name, tenant_id}` + Cypher 子图约束实现图通道隔离。MySQL `tenant_id` FK 实现行级隔离。Data Analyst SQL 生成注入 tenant 约束 + `execute_sql` 安全检查双重防护。

**Agent Workflow Platform** (v16): 新增 Workflow 子系统（独立 LangGraph），支持 Goal → Plan → Execute → Deliver 完整闭环。WorkflowPlanner 通过 LLM 将自然语言目标拆解为 DAG 执行计划，WorkflowExecutor 按依赖关系串行/并行执行步骤。6 个现有 Agent 统一抽象为 WorkflowTool，通过 ToolRegistry 注册。Artifact 系统产出 Report/Excel/Chart/CSV 等交付物并持久化到 MySQL。前端新增任务工作流面板，支持目标输入、DAG 可视化、进度轮询、产物查看。工作流状态通过 MySQL Checkpointer 持久化，支持断点续跑。

**Adaptive GraphRAG** (v17): 在检索层之上引入 Query-Aware 检索决策层。QueryProfiler 从 3 级扩展到 6 种查询类型，RetrievalPlanner 根据查询类型输出通道选择+图深度+融合策略，weight_matrix 扩展为 6 类型独立 RRF 权重。GraphUtilityEstimator 用 5 维启发式特征预测图检索价值，低分跳过 Neo4j。Orchestrator graph nodes 通过 intent 动态条件跳过图检索。50 测试全绿。

**Graph Reasoning Engine** (v18): 五阶段图推理管线——ReasoningPlanner 将 NL 转为结构化 ReasoningPlan，SubgraphRetriever 通过多跳 Cypher 抽取 Neo4j 子图为 NetworkX DiGraph，PathExplorer 用 BFS+Beam Search 发现候选推理路径，PathRanker 用 4 维加权排序（语义+置信度+时序+长度），ReasoningVerifier 用 LLM 验证答案是否被路径支持（SUPPORTED/PARTIAL/UNSUPPORTED）。修复 local_graph_search 支持真正 n-hop 循环扩展。47 测试全绿。

**Memory Graph System** (v19): 新增 `backend/memory/` 包。MemoryExtractor 在每次对话保存后通过 LLM 提取 Fact/Preference/Task/Relation 四种用户记忆，MemoryGraphStore 用 `MERGE (m:Memory)` 存入 Neo4j 并通过 `:MENTIONS` 关系链接知识图谱 Entity。MemoryImportance 用时间衰减（30天半衰期）+ 访问频次三维评分。MemoryRetriever 在 supervisor_node 检索前将用户记忆格式化为上下文注入 LLM prompt。通过 `memory_enabled` 配置开关控制，默认关闭。57 测试全绿。

**Deep Research Engine** (v20): 新增 `backend/research/` 包。ResearchPlanner 将研究目标通过 LLM 拆解为 DAG 执行计划（3~6 子任务，含依赖关系），ResearchExecutor 按依赖关系串行/并行调度 4 个 ResearchAgent（Web/Graph/Data/Internal KB），所有 Agent 输出统一为 Evidence 存入 EvidenceStore。ResearchReviewer 用 4 维加权评分（覆盖率 35% + 多样性 20% + 引用 25% + 置信度 20%）评估证据充分性，GapAnalyzer 自动生成补充检索查询，形成 Collect→Review→Gap→Collect 循环（max 3 rounds）。最终由 ResearchReportGenerator 生成证据驱动中文研究报告（每条结论绑定 Evidence ID）。前端新增研究工作区标签页（进度实时监控 + 证据卡片 + 报告阅读 + 历史回溯）。使用 qwen-turbo + max_tokens=1024 优化 LLM 响应速度。16 测试全绿。

### Key Files


| File | Purpose |
|------|---------|
| `backend/agent/orchestrator.py` | Supervisor graph: 6 agents + synthesize + planner + critique + replan + temporal routing |
| `backend/agent/brain.py` | SSE streaming, conversation storage, HITL resume |
| `backend/agent/tools.py` | emit_rag_step, emit_graph_step, token queue, MCP tool adapter |
| `backend/agent/model_router.py` | Dynamic LLM routing: turbo/plus/max by task |
| `backend/agent/data_analyst.py` | Text-to-SQL + MCP multi-data-source query |
| `backend/agent/mcp_client.py` | MCP connection manager: SSE/stdio transport, tools/list, tools/call |
| `backend/agent/chart_generator.py` | Echarts chart generation: type detection + config + markdown format |
| `backend/agent/tool_retriever.py` | MCP tool semantic retriever: Milvus index + top-k recall |
| `backend/agent/web_searcher.py` | Tavily API web search + fallback to RAG |
| `backend/agent/multimodal_specialist.py` | Visual retrieval: image/table description + Milvus search |
| `backend/rag/pipeline.py` | RAG LangGraph (retrieve→grade→rewrite→retrieve_expanded→grade_v2) |
| `backend/rag/utils.py` | Hybrid retrieval, rerank, auto-merge, query expansion (Step-Back/HyDE), 4-channel RRF |
| `backend/rag/graph_retriever.py` | local_graph_search, global_graph_search (+ time_filter) |
| `backend/rag/visual_retriever.py` | Visual retrieval: text-to-image-description semantic search |
| `backend/documents/loader.py` | Hierarchical chunking (PDF/Word/Excel/MD/Image) |
| `backend/documents/graph_extractor.py` | LLM entity/relation extraction (+ valid_from/valid_to, ontology-controlled v10) |
| `backend/documents/fingerprint.py` | SHA-256 file/chunk content fingerprinting for incremental updates |
| `backend/ontology/schema.py` | Domain ontology: 11 entity types, 12 predicates, 70+ relation rules, validation |
| `backend/pipeline/task_queue.py` | arq Redis task queue configuration (async ingestion) |
| `backend/pipeline/ingestion_worker.py` | Async document ingestion worker (full pipeline outside HTTP) |
| `backend/documents/layout_analyzer.py` | PDF layout analysis (text/image/table separation) |
| `backend/documents/media_extractor.py` | Image/table extraction + MinIO upload |
| `backend/documents/vlm_descriptor.py` | Qwen-VL chart/table description generation |
| `backend/storage/graph_client.py` | Neo4j driver (run_cypher/write_cypher) |
| `backend/storage/graph_ingestion.py` | MERGE entities + relations (+ temporal fields) |
| `backend/storage/graph_cleanup.py` | Neo4j cascade cleanup: strip edges, remove orphans |
| `backend/storage/doc_lifecycle.py` | Document lifecycle: soft-delete, chunk ID query, DocumentIndex upsert (hash/version) |
| `backend/storage/graph_schema.py` | Neo4j constraints and indexes |
| `backend/storage/models.py` | ORM: sessions, messages, chunks, CommunitySummary, DocumentIndex, QueryCacheStore, checkpoints |
| `backend/config.py` | Pydantic BaseSettings — centralized env validation, no hardcoded fallback secrets |
| `backend/storage/database.py` | SQLAlchemy engine + session factory (pool_size=10, max_overflow=20) |
| `backend/storage/checkpointer.py` | LangGraph MySQL checkpointer for state persistence |
| `alembic/` | Database migration tooling (alembic revision --autogenerate) |
| `backend/storage/cache.py` | Redis wrapper: get/set/lock/json |
| `backend/storage/parent_chunk_store.py` | MySQL parent chunk (L1/L2) store for auto-merge |
| `backend/graph/community.py` | Leiden clustering, summaries, Milvus indexing |
| `backend/graph/entity_resolution.py` | Two-stage entity dedup: edit-distance + LLM + Cypher merge |
| `backend/graph/incremental_clustering.py` | Incremental graph clustering: local patching + subgraph re-clustering |
| `backend/pipeline/stream_queue.py` | Redis Streams message queue: XADD/XREADGROUP/XACK with consumer groups |
| `backend/pipeline/stream_consumer.py` | Three-stage stream consumer: doc_ingest → graph_extract → vector_sync |
| `backend/pipeline/summary_updater.py` | Dirty-flag-driven targeted summary regeneration |
| `backend/milvus/client.py` | Milvus hybrid search + delete_by_chunk_ids + is_deleted filter |
| `backend/milvus/writer.py` | Batch write documents to Milvus with progress callback |
| `backend/embedding/service.py` | Dense (Qwen API) + Sparse (BM25) |
| `backend/evaluation/dataset.py` | Golden dataset loader (80 QA pairs, 7 query types, expected_agent) |
| `backend/evaluation/metrics.py` | Ragas metrics + generate_answer + routing accuracy + critique_pass_rate |
| `backend/schemas.py` | Pydantic: Chat*, Document*, HITL*, GraphEntity, GraphRelation, QueryPlan, CritiqueResult |
| `backend/cache/semantic_cache.py` | Milvus ANN + cosine + MySQL semantic cache |
| `backend/cache/singleflight.py` | Redis singleflight anti-stampede |
| `backend/cache/invalidation.py` | Doc delete → cache eviction |
| `backend/observability/tracing.py` | OTel init + ConsoleSpanExporter + get_tracer |
| `backend/observability/metrics.py` | Prometheus metrics: tokens, routing, latency, circuit breaker |
| `backend/observability/logging.py` | structlog JSON logging configuration |
| `backend/ha/circuit_breaker.py` | Circuit breaker state machine + LLM/Tavily protection |
| `backend/ha/retry.py` | tenacity exponential backoff retry decorator |
| `backend/ha/degradation.py` | Neo4j timeout → Dense+Sparse fallback |
| `backend/ha/load_monitor.py` | Redis sliding-window QPS monitor + NORMAL/WARNING/CRITICAL state machine |
| `backend/auth/__init__.py` | Auth package init |
| `backend/auth/models.py` | Tenant, User, Role SQLAlchemy models |
| `backend/auth/jwt_handler.py` | JWT encode/decode (PyJWT), password hash (passlib bcrypt) |
| `backend/auth/dependencies.py` | UserContext dataclass, get_current_user FastAPI dependency |
| `backend/auth/routes.py` | /auth/register, /auth/token endpoints |
| `tests/test_privilege_escalation.py` | v14 integration tests: auth, tenant isolation, privilege escalation |
| `tests/test_tenant_isolation_mysql.py` | MySQL tenant_id FK tests |
| `tests/test_tenant_isolation_milvus.py` | Milvus tenant_id schema tests |
| `tests/test_tenant_isolation_neo4j.py` | Neo4j tenant_id MERGE tests |
| `backend/billing/__init__.py` | Billing package init |
| `backend/billing/models.py` | TokenUsageLog, RateLimitRule, AuditLog SQLAlchemy models |
| `backend/billing/token_tracker.py` | Per-request token usage recording + summary queries |
| `backend/billing/rate_limiter.py` | Per-tenant Redis sliding-window rate limiter + get_tenant_rule |
| `backend/billing/audit.py` | Audit log writer + AuditContext context manager |
| `backend/billing/middleware.py` | FastAPI rate-limit middleware (429 on exceeded) |
| `backend/billing/routes.py` | /billing/usage + /billing/audit API endpoints |
| `backend/workflow/__init__.py` | Workflow 子系统入口：Planner, Executor, ArtifactGenerator 统一导出 |
| `backend/workflow/models.py` | WorkflowDefinition, WorkflowExecution, WorkflowArtifact ORM 模型 |
| `backend/workflow/schemas.py` | WorkflowStep, WorkflowPlan, ExecutionStatus, ArtifactType Pydantic |
| `backend/workflow/planner.py` | WorkflowPlanner: LLM 将自然语言目标拆解为 DAG 执行计划 |
| `backend/workflow/executor.py` | WorkflowExecutor: LangGraph DAG 执行引擎（串行+并行） |
| `backend/workflow/tool_runtime.py` | WorkflowTool 统一抽象 + ToolRegistry 注册中心 |
| `backend/workflow/agent_tools.py` | 将 6 个 Agent 注册为 WorkflowTool（轻量 LLM 调用） |
| `backend/workflow/artifact.py` | ArtifactGenerator: Report/Excel/Chart/CSV 交付物生成 |
| `backend/workflow/routes.py` | /workflows/plan, /execute, /status, /artifacts API |
| `backend/rag/retrieval_planner.py` | RetrievalPlanner: 查询类型→检索策略（通道+图深度+融合） |
| `backend/rag/graph_utility_estimator.py` | GraphUtilityEstimator: 5维启发式预测图检索价值，低分跳过 Neo4j |
| `scripts/run_adaptive_evaluation.py` | Adaptive GraphRAG 评测：分类准确率 + Plan 决策 + Utility 预测 |
| `backend/rag/graph_reasoning/schemas.py` | ReasoningPlan, ReasoningPath, VerificationResult Pydantic |
| `backend/rag/graph_reasoning/planning.py` | ReasoningPlanner: NL → 结构化推理计划 |
| `backend/rag/graph_reasoning/subgraph.py` | SubgraphRetriever: Neo4j 多跳 Cypher → NetworkX DiGraph |
| `backend/rag/graph_reasoning/path_explorer.py` | PathExplorer: BFS + Beam Search 路径发现 |
| `backend/rag/graph_reasoning/path_ranker.py` | PathRanker: 4 维加权路径排序 |
| `backend/rag/graph_reasoning/verifier.py` | ReasoningVerifier: LLM 答案-路径交叉验证 |
| `backend/memory/schemas.py` | MemoryNode, MemoryType (fact/preference/task/relation), MemoryExtraction |
| `backend/memory/extractor.py` | MemoryExtractor: LLM 从对话末尾10条消息提取结构化记忆 |
| `backend/memory/store.py` | MemoryGraphStore: Neo4j `:Memory` 节点 CRUD + `:MENTIONS` 关系链接 |
| `backend/memory/retriever.py` | MemoryRetriever: 查询用户记忆并格式化为 LLM 上下文 |
| `backend/memory/importance.py` | MemoryImportance: 时间衰减(30天)+频次三维评分 |
| `backend/research/schemas.py` | ResearchPlan, Evidence, ReviewResult, GapAnalysis Pydantic |
| `backend/research/models.py` | ORM: ResearchExecution, ResearchEvidence, ResearchReportRecord |
| `backend/research/planner.py` | ResearchPlanner: LLM goal→DAG plan decomposition (qwen-turbo) |
| `backend/research/executor.py` | ResearchExecutor: DAG 执行 + 审核循环 + 实时进度持久化 |
| `backend/research/evidence_store.py` | EvidenceStore: 证据持久化 + 多维查询统计 |
| `backend/research/research_agents.py` | 4 ResearchAgent 封装 (web/graph/data/internal_kb) |
| `backend/research/reviewer.py` | ResearchReviewer: 4 维证据评分 (coverage/diversity/citation/confidence) |
| `backend/research/gap_analyzer.py` | GapAnalyzer: LLM 缺失分析 → 补充检索查询 |
| `backend/research/report_generator.py` | ResearchReportGenerator: 证据驱动中文报告生成 |
| `backend/research/routes.py` | /research/* API 端点 (create/status/evidence/report/cancel/list) |
| `tests/test_research.py` | v20 研究引擎 16 个单元测试 |
| `tests/test_token_tracker.py` | Token usage + rate limiter tests |
| `tests/test_rate_limiter.py` | Rate limiter + SLA rule tests |
| `tests/test_audit.py` | Audit logger + context manager tests |
| `tests/test_billing_integration.py` | v15 billing/audit integration tests |
| `backend/agent/query_profiler.py` | Lightweight intent classifier: keyword + embedding → L1/L2/L3 |
| `backend/rag/dynamic_rrf.py` | Intent-driven RRF weight matrix (loads from config/weight_matrix.yaml) |
| `config/weight_matrix.yaml` | RRF weight config: L1 Dense 70%, L2 Graph 65%, L3 balanced |
| `scripts/run_ab_evaluation.py` | A/B evaluation: static vs dynamic chain comparison |
| `scripts/run_load_test.py` | Locust load test with L1/L2/L3 query coverage |
| `tests/test_query_profiler.py` | Query Profiler unit + integration tests |
| `tests/test_load_monitor.py` | Load Monitor unit tests (mocked Redis) |
| `tests/test_dynamic_rrf.py` | Dynamic RRF weight matrix tests |
| `scripts/run_community_clustering.py` | Offline: graph→cluster→summarize→index |
| `scripts/run_entity_resolution.py` | Offline: entity dedup pipeline |
| `scripts/run_evaluation.py` | RAG eval: 5 modes (retrieval/pipeline/e2e/graph/graph_compare) + latency + A/B compare |
| `scripts/graph_topology_stats.py` | Graph topology metrics: nodes, edges, orphans, type/predicate distribution |
| `scripts/grid_search_rrf.py` | RRF weight grid search (composite score, graph channel, all RAGAS metrics) |
| `scripts/generate_report.py` | HTML evaluation report: radar chart + bar chart + routing matrix + latency |
| `scripts/ci_evaluation.sh` | CI threshold check: context_precision ≥ 0.6, faithfulness ≥ 0.7, answer_relevancy ≥ 0.6 |
| `scripts/run_benchmark.py` | Concurrent cache benchmark |
| `scripts/benchmark_incremental.py` | Benchmark: full Louvain vs incremental clustering comparison |
| `tests/test_incremental_clustering.py` | Incremental clustering unit tests (mocked Neo4j) |
| `prometheus.yml` | Prometheus scrape config (targets app :8000) |
| `tests/test_doc_lifecycle.py` | Document soft-delete unit tests |
| `tests/test_evaluation.py` | Evaluation unit tests (golden dataset, RRF fusion, metrics signatures) |
| `tests/test_fingerprint.py` | Document fingerprint SHA-256 unit tests |
| `tests/test_incremental_upload.py` | Incremental upload integration tests (DocumentIndex, hash skip, cleanup) |
| `tests/test_v10_ontology.py` | v10 ontology schema + extraction validation tests |
| `start_worker.py` | arq async ingestion worker entrypoint |
| `frontend/index.html` | Vue 3 SPA: chat, trace canvas, HITL modal, settings |
| `frontend/script.js` | Vue 3 app logic: SSE handler, trace panel, HITL modal, session management |
| `frontend/style.css` | Gemini-inspired dual-theme (Light/Dark) styles |

### Patterns

- **Lazy model init**: `_get_supervisor_model()`, `_get_worker_model()` — global singletons
- **call_soon_threadsafe**: `emit_rag_step`/`emit_token` use loop-safe cross-thread scheduling
- **Rerank auto-detect**: model starts with "qwen" → native API, else OpenAI-compatible endpoint
- **Structured output fallback**: grade nodes try `with_structured_output()`, fall back to raw invoke + parse
- **Neo4j MERGE**: entities deduplicated by name constraint; relations upserted with weight=max
- **source_chunks**: every Neo4j edge stores referencing L3 chunk IDs for provenance
- **Milvus dynamic schema**: `enable_dynamic_field=True`, community summaries coexist with doc chunks
- **Graph events in SSE**: `_RagStepProxy` detects graph agents and auto-emits `graph_expand`/`community_match`
- **Soft-delete cascade**: DELETE endpoint → MySQL `is_deleted` → Milvus `delete_by_chunk_ids` → Neo4j `strip_chunk_from_edges` → `remove_empty_edges` → `remove_orphan_entities`
- **Temporal routing**: Supervisor detects time-sensitive queries (`is_temporal`/`temporal_year`) → `local_graph_search_node` passes `time_filter` → Cypher filters by `valid_from`/`valid_to`
- **Entity resolution**: `find_candidates_in_community` (edit-distance) → `resolve_entities_batch` (LLM confirm) → `merge_entity_pair` (Cypher DETACH DELETE + edge inheritance)
- **DocumentIndex**: tracks filename-level version/state; `mark_document_deleted` bumps version and sets `is_deleted` on both ParentChunk and DocumentIndex
- **RRF weights**: configurable via `RRF_WEIGHT_DENSE/SPARSE/GRAPH/VISUAL` env vars; `rrf_fusion_three_channel` supports 3 or 4 weights, grid-searchable via composite score
- **Evaluation 3 modes**: `retrieval` (initial retrieval only), `pipeline` (full RAG pipeline), `e2e` (LLM generates answer + routing accuracy + latency)
- **Golden dataset**: 84 QA pairs with `expected_agent` field for routing accuracy eval; 8 query types: conceptual, detail, cross_doc, global_summary, realtime, chat, data_query, privilege_escalation (v14)
- **RAGAS 4 metrics**: context_precision, context_recall, faithfulness, answer_relevancy; composite = 0.4*prec + 0.3*faith + 0.3*rel. Uses ragas 0.2.15 (0.4.x incompatible with DashScope API format). `context_precision` and `faithfulness` work reliably; `answer_relevancy` and `context_recall` may return NaN due to DashScope prompt format rejection (400 error)
- **Routing accuracy**: `evaluate_routing_accuracy()` calls Supervisor LLM directly (no Worker execution) and compares against `expected_agent`
- **EvaluationResult handling**: ragas 0.2.x returns dict directly; code also handles 0.4.x `EvaluationResult` objects via `_scores_dict`/`to_pandas` fallback chain
- **Chart NaN safety**: `generate_charts` and `generate_report` convert NaN to 0 before matplotlib rendering to prevent polar plot crashes
- **v8 Planner**: `planner_node` sits between supervisor and workers; complex queries get decomposed into multi-step plans (`QueryPlan` JSON); simple queries bypass planner entirely
- **v8 Critique**: `critique_node` sits after synthesize; validates draft answer against retrieved contexts via LLM cross-verification; outputs `CritiqueResult` with `is_valid`/`missing_information`/`feedback`. `direct_answer` bypasses Critique (goes directly to END) —闲聊没有检索上下文，Critique 必然判"依据不足"导致无效重试
- **v8 Self-correction loop**: `route_after_critique` → if invalid and retry<2 → `replan_node` (injects missing_info as supplement query) → supervisor re-routes; max 2 retries prevents infinite loops
- **v8 draft_answer**: `synthesize_node` always saves `draft_answer` to state (single worker: extract from worker_outputs; multi worker: LLM synthesis result)
- **v9 MCP client**: `MCPConnectionManager` manages connections to multiple MCP Servers (SSE/stdio); `get_mcp_manager()` global singleton; `connect()` → `get_available_tools()` → `call_tool()` lifecycle
- **v9 Dynamic tool registration**: `mcp_tools_to_langchain_tools()` converts MCP `tools/list` Schema to LangChain `StructuredTool`; `get_dynamic_tools()` returns tools for a specific agent
- **v9 Tool retriever**: `ToolRetriever` indexes MCP tool descriptions into Milvus; `retrieve_tools(query, top_k)` semantic search prevents context window explosion when 100+ tools exist
- **v9 Echarts**: `chart_generator.py` detects chart type via LLM, generates Echarts JSON config; frontend `configureMarked()` intercepts ````echarts` code blocks and renders via `echarts.init()`
- **v9 Data Analyst multi-source**: `get_mcp_data_sources()` discovers database-type MCP tools; `generate_mcp_query()` LLM generates query params from tool schema; `execute_mcp_query()` calls MCP tool
- **OTel manual spans**: `get_tracer("ragent.xxx")` + `tracer.start_as_current_span()` on Agent nodes, Milvus queries, Neo4j Cypher — no FastAPI auto-instrument
- **Prometheus /metrics**: `init_metrics(app)` registers `/metrics` endpoint; `Metrics.record_*()` methods called from spans
- **structlog**: `init_logging()` in `create_app()` configures JSON logging globally; use `get_logger("name")` for structured logging
- **Circuit breaker**: `CircuitBreaker(name, failure_threshold=3, recovery_timeout=60)` state machine; `llm_breaker`/`tavily_breaker` global instances; `with_circuit_breaker(breaker, fallback)` decorator
- **Degradation**: `safe_graph_search()` wraps `local_graph_search` with try/except → fallback to `retrieve_documents` (Dense+Sparse only)
- **Retry**: `with_retry(max_attempts=3)` uses tenacity exponential backoff (1s→2s→4s) for ConnectionError/TimeoutError
- **Semantic cache**: `query_cache(query)` checks Milvus ANN + cosine ≥ threshold before RAG; `write_cache(query, response)` stores on generation complete; `invalidate_by_filename(filename)` on document delete
- **Model routing**: `get_model_for_agent(agent_name)` returns ChatOpenAI with model from `ROUTE_MAP`; Supervisor/DirectAnswer use qwen-turbo, heavy tasks use qwen-plus/max
- **Singleflight**: `with_singleflight(key_prefix)` decorator wraps `write_cache` to prevent cache stampede under high concurrency
- **Supervisor manual JSON parsing**: Qwen `with_structured_output` incompatible with LangChain (json_mode needs "json" in prompt, function_calling conflicts with thinking mode). Fix: use `model.invoke()` + regex JSON extraction from response text.
- **v10 Ontology-controlled extraction**: `backend/ontology/schema.py` defines ENTITY_TYPES (11), RELATION_PREDICATES (12), RELATION_RULES (70+ triples with `*` wildcard). `graph_extractor.py` uses field_validator + `_normalize_entity_type()`/`_normalize_predicate()` + `_validate_extraction()` interceptor. DashScope `with_structured_output` returns `source`/`target` instead of `subject`/`object` — manual JSON parsing with field name mapping handles this.
- **v10 Entity resolution type filter**: `entity_resolution.py` Cypher adds `AND a.type = b.type` — only same-type entities compared for dedup, reduces false merges.
- **v10 Graph topology stats**: `scripts/graph_topology_stats.py` queries Neo4j for node/edge/orphan counts, type/predicate distributions, degree percentiles. Used for A/B comparison of extraction quality. Neo4j 5.26 requires `COUNT {}` instead of `size()` for pattern expressions.
- **v11 Incremental pipeline**: `fingerprint.py` computes SHA-256 per file. `doc_lifecycle.upsert_document_index()` tracks hash/version in `DocumentIndex` table. Upload endpoint checks hash → skip if unchanged. Changed files trigger: Milvus delete by filename → parent_chunk_store delete → `graph_cleanup.cleanup_by_filename()` → re-extract → re-insert.
- **v11 Async task queue**: `arq` (Redis-based) dispatches `run_ingestion_task` to `backend/pipeline/ingestion_worker.py`. Upload returns HTTP 202 immediately. Worker initializes its own DB/services. Fallback to sync if Redis unavailable.
- **v11 Milvus is_deleted fix**: `writer.py` now sets `is_deleted: False` on insert. Previously the field existed in schema and was filtered on retrieval but never set — worked by accident.
- **v12 Query Profiler**: `QueryProfiler.profile(query)` returns `QueryIntent(level, complexity_score, matched_keywords, embedding_similarity)`. Keyword matching 60% + Embedding cosine similarity 40%. Short queries (<5 chars) forced to L1. Module-level `_prototype_embeddings` cache for prototype query embeddings. `warmup()` pre-loads embeddings at startup (single API call for all 12 prototypes). L1 fast route: if `level==L1_FACTUAL` and `complexity_score<0.3`, skip Supervisor LLM and route directly to `direct_answer`.
- **v12 Model routing optimization**: `direct_answer_node` uses `get_model_for_agent("direct_answer")` → qwen-turbo (3-5s), not qwen-plus (15-20s). Combined with L1 fast route, simple queries complete in ~13s vs ~48s before optimization.
- **v12 Dynamic RRF**: `get_weights_for_intent(intent_level)` loads from `config/weight_matrix.yaml` (YAML hot-reload via `reload_weight_matrix()`). L1: Dense 70%, L2: Graph 65%, L3: balanced. Passed through `run_rag_graph(intent_level=...)` → `retrieve_documents(intent_level=...)`.
- **v12 Load Monitor**: `LoadMonitor` uses Redis INCR+EXPIRE per-second counters, `mget` sliding window. `get_state()` cached 1s. `should_skip_critique()` (WARNING+), `should_circuit_break_neo4j()` / `should_circuit_break_tavily()` (CRITICAL only). Module-level singleton `get_load_monitor()`.
- **v12 Adaptive degradation**: `_get_tenant_degradation(state)` helper extracts `tenant_id` → `get_tenant_rule()` for SLA tier → `monitor.get_tenant_degradation(tier)` for per-tenant degradation level. Used in `route_after_critique` (`skip_critique`/`cache_only` → skip replan), `local_graph_search_node` (`cache_only` → fallback to `retrieve_documents`), and `web_searcher_node` (`cache_only` → skip Tavily). Replaces the old non-tier-aware `should_skip_critique()`, `should_circuit_break_neo4j()`, `should_circuit_break_tavily()` direct calls. DB failure defaults to `free` tier.
- **v12 SSE events**: `query_profiler` event (intent level, score, keywords) emitted after supervisor routing. `system_state` event (normal/warning/critical, qps, thresholds) emitted per request.
- **v13 Incremental clustering**: `patch_new_node(node_name)` checks 1-hop neighbors' community_id; if >60% share one community → direct assignment (zero cost); otherwise `recluster_subgraph(affected_communities)` extracts local subgraph and runs Louvain only on it. `incremental_cluster_after_ingest(filename)` orchestrates the full flow.
- **v13 Dirty-flag summaries**: `CommunitySummary` MySQL table has `is_dirty` boolean. `mark_communities_dirty(cids)` sets flag on affected communities. `update_dirty_summaries()` scans dirty rows, regenerates only those, resets flag. Token savings 80-100%.
- **v13 Redis Streams**: `StreamQueue` wraps XADD/XREADGROUP/XACK. Three streams (doc_ingest, graph_extract, vector_sync) with consumer groups. `ack_and_publish()` chains stages. Dead letter after 3 retries. Falls back to arq if Redis unavailable.
- **v14 JWT auth**: `OAuth2PasswordBearer(tokenUrl="/auth/token")` + `get_current_user` dependency returns `UserContext` dataclass. PyJWT HS256 encoding, passlib bcrypt hashing. Token carries `user_id`, `tenant_id`, `tenant_name`, `role`, `access_level`.
- **v14 Tenant isolation (MySQL)**: `tenant_id` FK with `server_default="1"` on DocumentIndex, ChatSession, ParentChunk, QueryCacheStore. Existing data gets tenant 1 automatically.
- **v14 Tenant isolation (Milvus)**: `tenant_id` INT64 field in collection schema. `retrieve_documents(tenant_id=X)` appends `&& (tenant_id == X)` to `filter_expr`. Pre-filtering happens before ANN search — database-level enforcement.
- **v14 Tenant isolation (Neo4j)**: Entity MERGE key changed from `{name}` to `{name, tenant_id}`. Cypher queries add `AND a.tenant_id = $tenant_id AND b.tenant_id = $tenant_id` dynamically. Backward compatible (None = no constraint).
- **v14 user_context propagation**: `SupervisorState.user_context` dict flows from JWT → routes → brain → graph → all workers. Each worker extracts `tenant_id` from `state["user_context"]` and passes to retrieval functions.
- **v14 Data Analyst SQL isolation**: `generate_sql(tenant_id=X)` injects tenant constraint into LLM prompt. `execute_sql` has defense-in-depth: blocks queries on tenant-scoped tables without `tenant_id` in SQL text.
- **v14 Ingestion propagation**: `run_ingestion_task(tenant_id, access_level)` passes through to Milvus writer (per-doc dict), Neo4j ingestion, DocumentIndex upsert, and ParentChunk store.
- **v15 Token metering**: `record_token_usage(db, tenant_id, user_id, model_name, prompt_tokens, completion_tokens, agent_name, request_type)` writes to `token_usage_logs` table after each LLM call. `get_usage_summary(db, tenant_id, days)` aggregates for billing API.
- **v15 Rate limiting**: `TenantRateLimiter(redis_client, window=10)` uses Redis sliding-window per `tenant_id`. `check_rate_limit(tenant_id, qps_limit)` returns `{allowed, current_count, limit, retry_after}`. Fail-open on Redis errors.
- **v15 SLA tiers**: `RateLimitRule` table stores per-tenant `tier` (free/standard/premium/enterprise), `qps_limit`, `daily_token_limit`. `get_tenant_rule(db, tenant_id)` returns default free tier if no rule exists.
- **v15 SLA degradation**: `LoadMonitor.get_tenant_degradation(tier)` maps system load + tenant tier to degradation level: enterprise=full under CRITICAL, premium=skip_critique, free=cache_only. Integrated into orchestrator via `_get_tenant_degradation(state)` helper that resolves tenant_id → SLA tier → degradation level at each decision point.
- **v15 Audit logging**: `log_audit_event(db, tenant_id, user_id, action, target, arguments, result_summary, risk_level)` writes to `audit_logs` table. `AuditContext` context manager auto-logs exceptions as `risk_level="high"`.
- **v15 Billing API**: `GET /billing/usage?days=30` returns token consumption summary. `GET /billing/audit?action=mcp_tool_call&limit=50` returns paginated audit logs. Both tenant-scoped via JWT.
- **v15 HITL webhook**: `_notify_hitl_webhook(tenant_id, interrupt_data)` sends POST to `HITL_WEBHOOK_URL` env var on interrupt events. Non-blocking `asyncio.create_task`, 5s timeout, silent no-op if unset.
- **Frontend auth**: Login/register UI in `index.html`; `authToken` stored in `localStorage('ragent-token')`; all API calls use `_authFetch(url, options)` wrapper that injects `Authorization: Bearer <token>`; 401 on chat triggers auto-logout.
- **ChatRequest validation**: `message: str = Field(..., min_length=1)` rejects empty messages with 422 before LLM call.
- **Document listing**: `GET /documents` now queries MySQL `document_index` via `list_active_documents(tenant_id)` for tenant isolation, not raw Milvus.
- **Document delete isolation**: `mark_document_deleted(filename, tenant_id)` filters `DocumentIndex` by tenant_id — cross-tenant delete returns 404.
- **Session persistence**: `storage.save(..., tenant_id=...)` is mandatory — sessions without tenant_id are invisible to `list_session_infos(tenant_id)`. `delete_session(session_id, tenant_id)` also filters by tenant_id.
- **Session listing optimization**: `list_session_infos` uses single-pass batch queries (`GROUP BY`, `func.count`, `func.min`) instead of N+1 per-session fetches — 3 queries total regardless of session count.
- **Config management**: `backend/config.py` — Pydantic `BaseSettings` loads from `.env` with `extra="ignore"`; `get_settings()` singleton. No hardcoded fallback secrets — JWT_SECRET, DATABASE_URL raise `RuntimeError` if unset.
- **Error handling**: all `except Exception: pass` replaced with `log.warning("operation_failed", error=str(e))`; rate-limiter fail-open and SSE queue close are intentional but documented with inline comments.
- **SQL execution safety**: `data_analyst.execute_sql` has four-layer defense: (1) `startswith("SELECT")` check, (2) multi-statement `;` rejection, (3) `SET TRANSACTION READ ONLY`, (4) tenant-scoped table `tenant_id` filter enforcement.
- **Upload validation**: file size capped at `upload_max_size_mb` (default 50MB) via `backend/config.py`; oversized uploads rejected with 400 before processing.
- **Alembic migrations**: `alembic/` initialized, `env.py` wired to `backend/storage/database.py:Base.metadata` and `SQLALCHEMY_DATABASE_URL`. Use `alembic revision --autogenerate -m "..."` for schema changes.
- **OTel OTLP support**: `tracing.py` auto-detects `OTEL_EXPORTER_OTLP_ENDPOINT` env var — if set, uses `OTLPSpanExporter` (gRPC); falls back to `ConsoleSpanExporter` for development.
- **CORS**: origins read from `CORS_ORIGINS` env var (comma-separated), defaults to `*` for development.
- **Checkpointer pending_writes**: `_load_writes` must return `(task_id, channel, value)` triples for LangGraph 0.2+ compatibility. Old format `(channel, value)` causes "not enough values to unpack (expected 3, got 2)".
- **Milvus pymilvus 2.5 API**: `client.search()` uses `search_params` not `param`. Silent 0-result failure with old param name.
- **Milvus gRPC reconnect**: `_ensure_connected()` calls `get_load_state()` before each query; resets client on failure to prevent "closed channel" errors.
- **v16 Workflow 独立 LangGraph**: WorkflowExecutor 使用独立 StateGraph（与 Supervisor 图并行），通过 MySQL Checkpointer 持久化，支持断点续跑。节点：init → execute_step ⇄ finalize/error。
- **v16 WorkflowTool 统一抽象**: 6 个 Agent（rag_specialist 等）通过 `WorkflowTool.from_agent()` 包装，MCP 工具通过 `from_mcp()` 包装。ToolRegistry 单例管理，Executor 按 step.tool 名称查找并调用。
- **v16 DAG 依赖解析**: `_execute_step_node` 每轮找出所有依赖已满足的步骤并执行，独立步骤自动并行。依赖通过 `step.dependencies`（step_id 列表）声明，前序结果通过 `previous_results` 传递。
- **v16 轻量 Agent 调用**: `agent_tools.py` 使用直接 LLM 调用（不走完整 agent node 函数），每个 step 通过 `model.ainvoke()` 快速获取结果，避免 agent node 的复杂状态依赖和长耗时。
- **v16 Artifact 持久化**: `_run_workflow_background` 执行完成后调用 `ArtifactGenerator` 生成 Report，并将内容写入 `workflow_artifacts` 表。前端通过轮询 `/status` 获取进度，完成后通过 `/artifacts` 获取产物。
- **v16 前端 Workflow 面板**: Vue 3 组件实现 Goal 输入 → Plan DAG 可视化 → Execute 进度轮询 → Artifact 查看。历史记录列表自动加载，点击可回溯查看过往执行。
- **v16 upsert_document_index 异步管线修复**: 相同 hash 但 chunk_count 不同时（流式管线先写 0 后更新实际值），不再跳过而是更新 chunk_count。Stream consumer 注入 tenant_id 到 doc dicts 避免 FK 约束失败。
- **v17 6-Type Query Classification**: QueryProfiler 新增 `_classify_query_type()` 方法，关键词匹配 6 种类型（含 temporal 优先逻辑），`_TYPE_PROTOTYPES` 4 条/类型 Embedding 原型，`_type_prototype_embeddings` 独立缓存，warmup() 双批次预热。
- **v17 RetrievalPlanner**: `STRATEGY_MAP` 字典映射查询类型→RetrievalPlan（Pydantic 模型），`plan(intent=dict)` 支持 graph_hops/graph_skip 覆盖。factoid: 无图 0-hop，multi_hop: graph_first 3-hop，global_summary: community 优先。
- **v17 Adaptive weight_matrix**: 6 种查询类型独立权重（factoid graph=0, multi_hop graph=0.85, global_summary community=0.80）。`get_weights_for_intent(intent_level, query_type="")` query_type 优先查找，回退 L1/L2/L3。
- **v17 Orchestrator Adaptive Graph**: `local_graph_search_node` 读取 `state["query_intent"]` 传入 RetrievalPlanner，`skip_graph` 时直接调用 `retrieve_documents` 跳过 Neo4j。`safe_graph_search` 新增 `graph_hops` 参数透传。`global_graph_search_node` 非 community 类型时返回空跳过。
- **v17 GraphUtilityEstimator**: 5 维特征（entity_density 30% + rel_score 30% + reason_score 25% + time_score 15%），阈值 0.35。`score >= 0.55 → 3-hop`, `>= 0.25 → 1-hop`, `< 0.25 → 0-hop`。纯启发式零 LLM 调用。
- **v18 SubgraphRetriever**: `MATCH p = (a:Entity)-[:RELATES_TO*1..max_hops]->(b:Entity)` 多跳 Cypher，UNWIND 展开路径为三元组，构建 NetworkX DiGraph。空结果返回空图不抛异常。
- **v18 PathExplorer**: BFS 用 `nx.all_simple_paths(source, target, cutoff=max_hops)` 穷举，Beam Search 每跳保留 top beam_width 路径按 path_score 截断。支持跳过图中不存在的起始实体。
- **v18 PathRanker**: 4 维加权 `score = 0.30*semantic + 0.25*confidence + 0.20*temporal + 0.25*length_penalty`。semantic: 关键词重叠率，confidence: edges/hops 比率，length_penalty: 1/(1+log(1+hops))。
- **v18 ReasoningVerifier**: System prompt 输出 JSON `{verdict, confidence, explanation, supporting_paths}`，LLM 交叉验证答案与推理路径的一致性。
- **v18 Multi-hop Fix**: `graph_retriever.local_graph_search` 的邻居扩展从单次改为 `for hop in range(1, graph_hops)` 循环，`NOT other.name IN $names` 防止重复扩展，上限 50 实体。
- **v19 Memory Graph Store**: `MERGE (m:Memory {memory_id})` 创建 `:Memory` 节点，`ON MATCH` 用 `CASE WHEN $importance > m.importance` 保留最高分。`MATCH (m:Memory)-[r:MENTIONS]->(e:Entity)` 链接到知识图谱实体。
- **v19 Memory Importance**: `score = base_score * (0.5*exp(-days/30) + 0.3*log(1+freq)/log(5) + 0.2)`。时间衰减 30 天半衰期，访问频次对数归一化。
- **v19 Memory Extraction Hook**: `chat_with_agent` 和 `chat_with_agent_stream` 在 `storage.save()` 之后通过 `asyncio.create_task` 异步调用 `MemoryExtractor.extract()`，非阻塞失败静默。
- **v19 Memory Injection**: `supervisor_node` 在 profiler 之后通过 `MemoryRetriever.retrieve(user_id, tenant_id)` 获取记忆上下文，格式化为 `## 用户记忆` 注入 user_query。
- **v20 Research Planner**: `ResearchPlanner.plan()` 调用 `init_chat_model("qwen-turbo", max_tokens=1024)` 生成精简 JSON 计划。独立模型实例，不复用 model_router 全局配置。
- **v20 Research Executor**: `_execute_all_tasks` 按依赖关系分批执行——无依赖任务并行 (`asyncio.gather`)，有依赖任务等待前序完成。每批完成后立即 `_update_execution_record()` 持久化进度到 MySQL。
- **v20 Research Agents**: 4 个 agent 函数（`run_web_research`/`run_graph_research`/`run_data_research`/`run_internal_kb_research`）通过 `AGENT_MAP` 字典调度，统一返回 `tuple[str, list[Evidence]]`。每个 agent 独立创建 `init_chat_model` 实例（qwen-turbo, max_tokens=1024, timeout=60）。
- **v20 Evidence Store**: `EvidenceStore.save_batch()` 批量写入 MySQL `research_evidence` 表。`get_stats()` 返回来源分布 + 置信度分布 + 覆盖率统计。
- **v20 Reviewer**: 纯启发式评分（零 LLM 调用）——覆盖率 = 有证据的 task 占比，多样性 = 来源种类/4，引用率 = 有 citation 的 evidence 占比，置信度 = high=1.0/medium=0.6/low=0.3 均值。`overall >= 0.70 AND coverage >= 0.60` 判定通过。
- **v20 Gap Analyzer**: `GapAnalyzer.analyze()` 先尝试 LLM 分析生成补充查询，失败则 fallback 到启发式（取第一个 gap 作为查询）。`is_sufficient=True` 时直接返回空。
- **v20 Report Generator**: `ResearchReportGenerator.generate()` 构建 evidence_by_task 索引 + task_results 摘要，调用 LLM 生成中文 Markdown 报告。`_extract_summary()` 用正则提取"摘要"段落。每条结论必须引用 Evidence ID。
- **v20 Async Execution**: routes.py 使用 `asyncio.create_task()` 而非 FastAPI `BackgroundTasks` 启动后台研究任务——前者在当前 event loop 中可靠调度，后者在某些 Starlette 版本中可能不执行 async 任务。
- **v20 Progress Persistence**: Executor 在 `_execute_all_tasks` 的 while 循环内每批任务完成后调用 `_update_execution_record()`，前端 3 秒轮询 `/research/{id}` 立即感知进度变化。
- **v20 Model Optimization**: Research 模块使用独立 `init_chat_model("qwen-turbo", max_tokens=1024, timeout=60)` 而非全局 model_router (qwen-plus, max_tokens=8192, timeout=120)。Planner 74s → 6.6s (11x 提升)。Agent prompt 精简为中文短提示词。
