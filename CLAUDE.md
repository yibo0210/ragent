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
| `backend/storage/database.py` | SQLAlchemy engine + session factory |
| `backend/storage/checkpointer.py` | LangGraph MySQL checkpointer for state persistence |
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
- **Golden dataset**: 80 QA pairs with `expected_agent` field for routing accuracy eval; 7 query types: conceptual, detail, cross_doc, global_summary, realtime, chat, data_query
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
- **v12 Query Profiler**: `QueryProfiler.profile(query)` returns `QueryIntent(level, complexity_score, matched_keywords, embedding_similarity)`. Keyword matching 60% + Embedding cosine similarity 40%. Short queries (<5 chars) forced to L1. Module-level `_prototype_embeddings` cache for prototype query embeddings.
- **v12 Dynamic RRF**: `get_weights_for_intent(intent_level)` loads from `config/weight_matrix.yaml` (YAML hot-reload via `reload_weight_matrix()`). L1: Dense 70%, L2: Graph 65%, L3: balanced. Passed through `run_rag_graph(intent_level=...)` → `retrieve_documents(intent_level=...)`.
- **v12 Load Monitor**: `LoadMonitor` uses Redis INCR+EXPIRE per-second counters, `mget` sliding window. `get_state()` cached 1s. `should_skip_critique()` (WARNING+), `should_circuit_break_neo4j()` / `should_circuit_break_tavily()` (CRITICAL only). Module-level singleton `get_load_monitor()`.
- **v12 Adaptive degradation**: `route_after_critique` checks `monitor.should_skip_critique()` — WARNING+ skips replan. `local_graph_search_node` checks `should_circuit_break_neo4j()` — CRITICAL falls back to `retrieve_documents`. `web_searcher_node` checks `should_circuit_break_tavily()` — CRITICAL skips Tavily, triggers existing RAG fallback.
- **v12 SSE events**: `query_profiler` event (intent level, score, keywords) emitted after supervisor routing. `system_state` event (normal/warning/critical, qps, thresholds) emitted per request.
- **v13 Incremental clustering**: `patch_new_node(node_name)` checks 1-hop neighbors' community_id; if >60% share one community → direct assignment (zero cost); otherwise `recluster_subgraph(affected_communities)` extracts local subgraph and runs Louvain only on it. `incremental_cluster_after_ingest(filename)` orchestrates the full flow.
- **v13 Dirty-flag summaries**: `CommunitySummary` MySQL table has `is_dirty` boolean. `mark_communities_dirty(cids)` sets flag on affected communities. `update_dirty_summaries()` scans dirty rows, regenerates only those, resets flag. Token savings 80-100%.
- **v13 Redis Streams**: `StreamQueue` wraps XADD/XREADGROUP/XACK. Three streams (doc_ingest, graph_extract, vector_sync) with consumer groups. `ack_and_publish()` chains stages. Dead letter after 3 retries. Falls back to arq if Redis unavailable.
- **Checkpointer pending_writes**: `_load_writes` must return `(task_id, channel, value)` triples for LangGraph 0.2+ compatibility. Old format `(channel, value)` causes "not enough values to unpack (expected 3, got 2)".
- **Milvus pymilvus 2.5 API**: `client.search()` uses `search_params` not `param`. Silent 0-result failure with old param name.
- **Milvus gRPC reconnect**: `_ensure_connected()` calls `get_load_state()` before each query; resets client on failure to prevent "closed channel" errors.
