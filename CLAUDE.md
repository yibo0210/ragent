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
| `backend/agent/orchestrator.py` | Supervisor graph: 6 agents + synthesize |
| `backend/agent/brain.py` | SSE streaming, conversation storage, HITL resume |
| `backend/agent/tools.py` | emit_rag_step, emit_graph_step, token queue, weather/search |
| `backend/rag/pipeline.py` | RAG LangGraph (retrieve→grade→rewrite) |
| `backend/rag/utils.py` | Hybrid retrieval, rerank, auto-merge, 3-channel RRF |
| `backend/rag/graph_retriever.py` | local_graph_search, global_graph_search |
| `backend/documents/loader.py` | Hierarchical chunking (PDF/Word/Excel/MD/Image) |
| `backend/documents/graph_extractor.py` | LLM entity/relation extraction |
| `backend/storage/graph_client.py` | Neo4j driver (run_cypher/write_cypher) |
| `backend/storage/graph_ingestion.py` | MERGE entities + relations into Neo4j |
| `backend/graph/community.py` | Leiden clustering, summaries, Milvus indexing |
| `backend/milvus/client.py` | Milvus hybrid search (dense+sparse RRF) |
| `backend/embedding/service.py` | Dense (Qwen API) + Sparse (BM25) |
| `backend/storage/models.py` | ORM: sessions, messages, chunks, CommunitySummary, checkpoints |
| `backend/schemas.py` | Pydantic: Chat*, Document*, HITL*, GraphEntity, GraphRelation |
| `scripts/run_community_clustering.py` | Offline: graph→cluster→summarize→index |
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
