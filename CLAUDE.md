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

### Architecture

**Supervisor-Workers** (LangGraph): supervisor routes to 6 agents — `rag_specialist`, `local_graph_search`, `global_graph_search`, `web_searcher`, `data_analyst`, `direct_answer`. Multi-worker via `Send` fan-out, merged by `synthesize`.

**RAG Pipeline** (separate LangGraph): `retrieve → grade → [rewrite → retrieve_expanded → grade_v2]`. L1(1200)/L2(600)/L3(300) chunking. Leaf-only Milvus indexing. Auto-merge L3→L2→L1.

**GraphRAG**: Upload → L2 chunks → LLM extraction → Neo4j MERGE (entity + relation + source_chunks). Offline: `scripts/run_community_clustering.py` → Leiden → summaries → Milvus + MySQL.

**SSE Streaming**: `routing`, `agent_start/done`, `rag_step`, `graph_expand`, `community_match`, `content`, `trace`, `agent_trace`, `hitl_interrupt`, `error`.

**HITL**: LangGraph `interrupt()` (scenario A: low confidence RAG, scenario B: non-SELECT SQL). Redis lock → HTTP 423 during pending. Resume via `Command(resume=...)`.

### Key Files


| File | Purpose |
|------|---------|
| `backend/agent/orchestrator.py` | Supervisor graph: 6 agents + synthesize + temporal routing |
| `backend/agent/brain.py` | SSE streaming, conversation storage, HITL resume |
| `backend/agent/tools.py` | emit_rag_step, emit_graph_step, token queue, weather/search |
| `backend/rag/pipeline.py` | RAG LangGraph (retrieve→grade→rewrite) |
| `backend/rag/utils.py` | Hybrid retrieval, rerank, auto-merge, 3-channel RRF (configurable weights) |
| `backend/rag/graph_retriever.py` | local_graph_search, global_graph_search (+ time_filter) |
| `backend/documents/loader.py` | Hierarchical chunking (PDF/Word/Excel/MD/Image) |
| `backend/documents/graph_extractor.py` | LLM entity/relation extraction (+ valid_from/valid_to) |
| `backend/storage/graph_client.py` | Neo4j driver (run_cypher/write_cypher) |
| `backend/storage/graph_ingestion.py` | MERGE entities + relations (+ temporal fields) |
| `backend/storage/graph_cleanup.py` | Neo4j cascade cleanup: strip edges, remove orphans |
| `backend/storage/doc_lifecycle.py` | Document lifecycle: soft-delete, chunk ID query |
| `backend/graph/community.py` | Leiden clustering, summaries, Milvus indexing |
| `backend/graph/entity_resolution.py` | Two-stage entity dedup: edit-distance + LLM + Cypher merge |
| `backend/evaluation/dataset.py` | Golden dataset loader (50 QA pairs) |
| `backend/evaluation/metrics.py` | Ragas metrics: faithfulness, relevancy, precision |
| `backend/milvus/client.py` | Milvus hybrid search + delete_by_chunk_ids + is_deleted filter |
| `backend/embedding/service.py` | Dense (Qwen API) + Sparse (BM25) |
| `backend/storage/models.py` | ORM: sessions, messages, chunks, CommunitySummary, DocumentIndex, checkpoints (+ soft-delete fields) |
| `backend/schemas.py` | Pydantic: Chat*, Document*, HITL*, GraphEntity, GraphRelation, DocumentStatus |
| `scripts/run_community_clustering.py` | Offline: graph→cluster→summarize→index |
| `scripts/run_entity_resolution.py` | Offline: entity dedup pipeline |
| `scripts/run_evaluation.py` | Automated RAG eval + matplotlib charts |
| `scripts/grid_search_rrf.py` | RRF weight grid search optimization |
| `backend/observability/tracing.py` | OTel init + ConsoleSpanExporter + get_tracer |
| `backend/observability/metrics.py` | Prometheus metrics: tokens, routing, latency, circuit breaker |
| `backend/observability/logging.py` | structlog JSON logging configuration |
| `backend/ha/circuit_breaker.py` | Circuit breaker state machine + LLM/Tavily protection |
| `backend/ha/retry.py` | tenacity exponential backoff retry decorator |
| `backend/ha/degradation.py` | Neo4j timeout → Dense+Sparse fallback |
| `prometheus.yml` | Prometheus scrape config (targets app :8000) |
| `backend/cache/semantic_cache.py` | Milvus ANN + cosine + MySQL semantic cache |
| `backend/cache/singleflight.py` | Redis singleflight anti-stampede |
| `backend/cache/invalidation.py` | Doc delete → cache eviction |
| `backend/agent/model_router.py` | Dynamic LLM routing: turbo/plus/max by task |
| `scripts/run_benchmark.py` | Concurrent cache benchmark |
| `frontend/script.js` | Vue 3: SSE handler, trace panel, HITL modal |

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
- **RRF weights**: configurable via `RRF_WEIGHT_DENSE/SPARSE/GRAPH` env vars, grid-searchable via `scripts/grid_search_rrf.py`
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
- **Checkpointer pending_writes**: `_load_writes` must return `(task_id, channel, value)` triples for LangGraph 0.2+ compatibility. Old format `(channel, value)` causes "not enough values to unpack (expected 3, got 2)".
- **Milvus pymilvus 2.5 API**: `client.search()` uses `search_params` not `param`. Silent 0-result failure with old param name.
- **Milvus gRPC reconnect**: `_ensure_connected()` calls `get_load_state()` before each query; resets client on failure to prevent "closed channel" errors.
