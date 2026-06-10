<div align="center">

# Ragent AI

**Enterprise Multi-Agent GraphRAG Knowledge Base Assistant**

A full-stack Retrieval-Augmented Generation platform built on LangGraph Supervisor-Workers architecture with GraphRAG semantic network capabilities. Features multi-agent collaboration, hybrid (vector + graph) retrieval, Human-in-the-Loop (HITL) interrupt/resume, and real-time streaming responses.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)
![Vue.js](https://img.shields.io/badge/Vue.js-3-4FC08D?style=flat&logo=vuedotjs&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Milvus](https://img.shields.io/badge/Milvus-2.5-00A1E0?style=flat&logo=milvus&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-5.26-4581C3?style=flat&logo=neo4j&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

<br/>

<img src="docs/img.png" width="100%" alt="Ragent AI Interface" />

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Roadmap](#roadmap)

---

## Overview

Ragent AI is a production-ready **multi-agent GraphRAG platform** that orchestrates specialized AI workers to answer user questions by combining private document retrieval, web search, structured data analysis, and **knowledge graph traversal**. Built on **LangGraph Supervisor-Workers architecture**, it delivers accurate, source-attributed answers with full audit traceability.

**Core Capabilities:**

- **Multi-Agent Collaboration** — Supervisor-Workers model with 6 specialized agents: RAG Specialist, Local Graph Search, Global Graph Search, Web Searcher, Data Analyst, and Direct Answer — intelligently routed with support for parallel dispatch
- **GraphRAG Semantic Network** — LLM-powered entity/relation extraction during document ingestion, Neo4j graph storage, Leiden community clustering, and hierarchical community summarization
- **Hybrid Graph-Vector Retrieval** — Dense (Qwen text-embedding-v1, 1536-dim) + Sparse (BM25) + Graph triples three-channel fusion, with local search (vector → graph expansion) and global search (community summary matching) modes
- **Three-Level Hierarchical Chunking** — L1 (1200 chars) / L2 (600 chars) / L3 (300 chars) sliding-window chunking with auto-merging retriever; L2 chunks feed graph extraction while L3 chunks are vector-indexed
- **Human-in-the-Loop (HITL)** — LangGraph `interrupt()` mechanism for low-confidence RAG retrieval and risky SQL review; resume graph execution with human-approved inputs
- **State Persistence** — MySQL-based LangGraph checkpointer enables graph state persistence across sessions, supporting long-running interrupt/resume cycles
- **Real-Time Streaming** — SSE-based token streaming with live agent status visualization (Trace Canvas) and interleaved RAG/graph step tracking
- **Premium Gemini-Inspired UI** — Clean, modern dual-theme (Light/Dark) interface with real-time multi-agent trace panel and HITL approval modal

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                       Frontend (Vue 3 SPA)                             │
│   Chat UI · Session Management · Knowledge Base Upload                │
│   Trace Canvas (Agent State + Timeline) · HITL Approval Modal         │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ SSE / HTTP
┌──────────────────────────────▼─────────────────────────────────────────┐
│                     FastAPI Application Layer                           │
│  ┌────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │  api/routes.py          │  │  schemas.py                          │  │
│  │  REST + SSE endpoints   │  │  Pydantic Request/Response Models    │  │
│  └───────────┬─────────────┘  └─────────────────────────────────────┘  │
│              │                                                          │
│  ┌───────────▼──────────────────────────────────────────────────────┐  │
│  │            LangGraph Supervisor-Workers Orchestrator (v8)        │  │
│  │                                                                   │  │
│  │                     ┌──────────────┐                              │  │
│  │                     │  Supervisor  │  (Intent Routing)            │  │
│  │                     └──────┬───────┘                              │  │
│  │         ┌──────────────────┼─────────────────────────────────┐    │  │
│  │         │                  │                                 │    │  │
│  │  ┌──────▼──────┐  ┌───────▼──────┐  ┌─────────▼─────────┐   │    │  │
│  │  │ RAG         │  │ Local Graph  │  │ Global Graph      │   │    │  │
│  │  │ Specialist  │  │ Search       │  │ Search            │   │    │  │
│  │  └──────┬──────┘  └───────┬──────┘  └─────────┬─────────┘   │    │  │
│  │         │                  │                   │             │    │  │
│  │  ┌──────▼──────┐  ┌───────▼──────┐  ┌─────────▼─────────┐   │    │  │
│  │  │ Web Searcher│  │ Data Analyst │  │ Direct Answer→END │   │    │  │
│  │  │ (Tavily)    │  │ (Text-to-SQL)│  │ (skip Critique)   │   │    │  │
│  │  └──────┬──────┘  └───────┬──────┘  └───────────────────┘   │    │  │
│  │         └──────────────────┼─────────────────────────────┘   │    │  │
│  │                            │                                 │    │  │
│  │                     ┌──────▼───────┐                         │    │  │
│  │                     │  Synthesize  │ ← Multi-Worker Merge    │    │  │
│  │                     └──────┬───────┘                         │    │  │
│  │                            │                                 │    │  │
│  │                     ┌──────▼───────┐                         │    │  │
│  │                     │   Critique   │ ← Fact-checking (v8)   │    │  │
│  │                     └──────┬───────┘                         │    │  │
│  │                   valid │   │ invalid, retry<2               │    │  │
│  │                    ┌────┘   └────┐                           │    │  │
│  │                    ▼             ▼                           │    │  │
│  │                   END       ┌─────────┐                     │    │  │
│  │                             │ Replan  │ → Supervisor (v8)   │    │  │
│  │                             └─────────┘                     │    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                             │                                           │
│  ┌──────────────────────────▼───────────────────────────────────────┐  │
│  │                     RAG Pipeline (LangGraph)                      │  │
│  │  retrieve → grade → [rewrite → retrieve_expanded → grade_v2]     │  │
│  │                        ↑ force_interrupt on 2nd failure           │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                           │
│  ┌──────────────────────────▼───────────────────────────────────────┐  │
│  │                     Retrieval Engine                               │  │
│  │  Hybrid Vector Search · Reranking · Auto-Merging · Query Rewrite  │  │
│  │  Graph Local Search (vector → 1-hop expansion)                    │  │
│  │  Graph Global Search (community summary matching)                 │  │
│  │  Three-Channel RRF Fusion (Dense + Sparse + Graph)                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┬──────────────┐
        │                      │                      │              │
┌───────▼───────┐  ┌───────────▼──────────┐  ┌───────▼───────┐  ┌──▼─────┐
│    Milvus     │  │       MySQL          │  │     Redis     │  │ Neo4j  │
│  Vector DB    │  │  Sessions · Messages │  │   Hot Cache   │  │ Graph  │
│  HNSW + Sparse│  │  Chunks · Summaries  │  │  HITL Lock    │  │ Store  │
│  + Summaries  │  │  Graph Checkpoints   │  │               │  │        │
└───────────────┘  └──────────────────────┘  └───────────────┘  └────────┘
```

### Agent Routing Flow (v8)

```
                         ┌──────────────┐
                         │  User Query  │
                         └──────┬───────┘
                                │
                     ┌──────────▼──────────┐
                     │    Supervisor       │
                     │   (Intent Router)   │
                     └──────────┬──────────┘
                                │
          ┌─────────────────────┼─────────────────────────────┐
          │                     │                             │
          ▼                     ▼                             ▼
 ┌─────────────────┐  ┌────────▼────────┐  ┌─────────────────▼──┐
 │   Planner (v8)  │  │ RAG Specialist  │  │ Local/Global Graph │
 │ Complex query   │  │ (Doc retrieval) │  │ Search             │
 │ decomposition   │  │                 │  │                    │
 └────────┬────────┘  └────────┬────────┘  └────────┬───────────┘
          │                    │                     │
          └────────────────────┼─────────────────────┘
                               │
          ┌────────────────────┼─────────────────────┐
          │                    │                     │
 ┌────────▼────────┐  ┌────────▼────────┐  ┌─────────▼─────────┐
 │  Web Searcher   │  │  Data Analyst   │  │  Direct Answer    │
 │  (Tavily API)   │  │  (Text-to-SQL)  │  │  → END (skip      │
 │                 │  │                 │  │    Critique)       │
 └────────┬────────┘  └────────┬────────┘  └───────────────────┘
          │                    │
          └────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Synthesize       │ ← Multi-Worker
                    │  (Merge Answers)    │    Aggregation
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │     Critique (v8)   │ ← Fact-checking
                    │  (Cross-verify with │
                    │   retrieved context)│
                    └──────────┬──────────┘
                               │
                    ┌──────────┼──────────┐
                    │                     │
                 valid                invalid, retry<2
                    │                     │
                    ▼                     ▼
               ┌─────────┐         ┌───────────┐
               │  Answer  │         │  Replan   │ → Supervisor
               └─────────┘         └───────────┘   (self-correction)
```

### Document Ingestion Flow

```
Document Upload
    │
    ├── 1. L1/L2/L3 Hierarchical Chunking
    ├── 2. L3 → Milvus Vector Indexing
    ├── 3. L1/L2 → MySQL Parent Chunk Store
    │
    └── 4. L2 Text → LLM Entity/Relation Extraction → Neo4j
              │
              ├── MERGE Entity nodes (name, type, description)
              ├── MERGE RELATES_TO edges (predicate, weight)
              └── Bind source_chunks (L3 chunk IDs) to edges
```

### GraphRAG Offline Pipeline

```
Neo4j Full Graph
    │
    ├── 1. Pull all Entity + RELATES_TO → NetworkX DiGraph
    ├── 2. Leiden (Louvain) Community Detection
    ├── 3. Write community_id back to Neo4j entities
    ├── 4. Per-community: collect entities + relations
    ├── 5. LLM generates community summary (200-400 words)
    └── 6. Summaries → Vectorized → Milvus + MySQL
```

---

## Key Features

### Multi-Agent System

| Feature | Description |
|---------|-------------|
| **Supervisor Router** | LLM-powered intent analysis for intelligent agent selection (supports single + parallel dispatch via LangGraph `Send`) |
| **Planner (v8)** | Complex query decomposition: breaks multi-hop questions into step-by-step execution plans targeting different agents |
| **RAG Specialist** | Full RAG pipeline: hybrid retrieval → rerank → auto-merge → grading → rewrite → expanded retrieval |
| **Local Graph Search** | Vector search → Neo4j entity lookup → 1-hop graph expansion → merged context for multi-hop reasoning |
| **Global Graph Search** | Direct community summary matching in Milvus for panoramic/overview questions |
| **Web Searcher** | Tavily API integration for real-time web search with automatic fallback to RAG on API failure |
| **Data Analyst** | Text-to-SQL worker: discovers schema → generates read-only SQL → executes → presents insights |
| **Direct Answer** | Handles greetings, chitchat, and general knowledge queries without retrieval overhead |
| **Critique (v8)** | Post-generation fact-checking: cross-verifies draft answer against retrieved contexts; triggers self-correction loop on hallucination |
| **Replan (v8)** | Self-correction: injects missing information as supplement queries and re-routes to Supervisor (max 2 retries) |
| **Parallel Dispatch** | Supervisor can route to multiple workers simultaneously; synthesize node aggregates results |

### GraphRAG Engine

| Feature | Description |
|---------|-------------|
| **Entity/Relation Extraction** | LLM structured output extracts (Subject, Predicate, Object) triples from L2 chunks during document ingestion |
| **Entity Deduplication** | Neo4j MERGE with unique constraint on entity name prevents duplicate nodes across documents |
| **Source Provenance** | Every graph edge stores `source_chunks` — the list of L3 chunk IDs it was extracted from, enabling full traceability |
| **Leiden Clustering** | Community detection groups related entities into thematic clusters; community IDs written back to Neo4j |
| **Community Summarization** | LLM generates a 200-400 word overview per community; summaries are vectorized and indexed in Milvus |
| **Graph Local Search** | For multi-hop questions: Milvus retrieval → extract linked entities from Neo4j → 1-hop neighbor expansion |
| **Graph Global Search** | For overview questions: match user query against community summaries in Milvus |
| **Three-Channel RRF** | `RRF_Score = w1/(k+rank_dense) + w2/(k+rank_sparse) + w3/(k+rank_graph)` — weighted fusion of all retrieval channels |

### Retrieval Engine

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | Dense embeddings (Qwen text-embedding-v1) + BM25 sparse vectors, fused via RRF in Milvus |
| **Reranking** | Post-retrieval relevance scoring via DashScope qwen3-rerank API with graceful degradation |
| **Auto-Merging** | L3→L2→L1 hierarchical merging — when multiple sibling leaf chunks are retrieved, they collapse into the parent chunk for coherent context |
| **Three-Level Chunking** | L1 (1200 chars) → L2 (600 chars) → L3 (300 chars) with parent-child relationship tracking |
| **Leaf-Only Indexing** | Only leaf chunks (L3) are vectorized in Milvus; parent chunks stored in MySQL to reduce index size |

### Query Intelligence

| Feature | Description |
|---------|-------------|
| **Step-Back Prompting** | For specific/detail questions — generates a higher-level question to broaden retrieval scope |
| **HyDE** | For vague/conceptual questions — generates a hypothetical document for semantic retrieval |
| **Complex Expansion** | Multi-step questions — combines both strategies with deduplication |
| **Relevance Grading** | LLM-based structured output grades retrieved documents; triggers rewrite if irrelevant; triggers HITL interrupt on second consecutive failure |

### Human-in-the-Loop (HITL)

| Feature | Description |
|---------|-------------|
| **Low-Confidence Defense** | RAG document grading fails twice → LangGraph `interrupt()` → human reviews and modifies query or injects context |
| **SQL Safety Review** | Data Analyst generates non-SELECT SQL → graph pauses → human approves or rejects before execution |
| **Session Lock** | Redis distributed lock prevents concurrent messages during HITL pending state (returns HTTP 423) |
| **State Persistence** | MySQL-based LangGraph checkpointer saves full graph state for multi-hour interrupt/resume cycles |

### Application Layer

| Feature | Description |
|---------|-------------|
| **Streaming Responses** | SSE-based token streaming with real-time agent status events |
| **Trace Canvas** | Right-side panel showing live agent state nodes and execution timeline |
| **Session Management** | Multi-turn conversations persisted in MySQL with Redis caching |
| **Conversation Summarization** | Auto-summarizes history beyond 50 turns to manage token budgets |
| **RAG Trace** | Every response includes full retrieval audit: strategy, scores, merge decisions, source chunks |
| **Answer Abort** | Frontend AbortController + backend StreamingResponse for mid-generation cancellation |
| **Dual Theme** | Gemini-inspired Light/Dark theme with CSS variables; persisted to localStorage |
| **Dead-Loop Detection** | LangGraph `recursion_limit=15` prevents infinite agent loops |

### Knowledge Governance (v4.0)

| Feature | Description |
|---------|-------------|
| **Cascading Soft-Delete** | Cross-database cascade: MySQL `is_deleted` → Milvus batch delete → Neo4j edge strip + orphan cleanup |
| **Document Index** | `document_index` table tracks filename-level version, hash, and deletion state |
| **Entity Resolution** | Two-stage dedup: intra-community edit-distance recall → LLM confirmation → Cypher node merge with edge inheritance |
| **Temporal GraphRAG** | `valid_from`/`valid_to` on entities and relations; Supervisor auto-routes time-sensitive queries with year filter |

### Evaluation & CI/CD (v4.0)

| Feature | Description |
|---------|-------------|
| **Golden Dataset** | 80 hand-crafted QA pairs across 7 query types (conceptual, detail, cross_doc, global_summary, realtime, chat, data_query) with `expected_agent` for routing accuracy |
| **Ragas Metrics** | 4 metrics (ragas 0.2.15): context_precision, context_recall, faithfulness, answer_relevancy; composite score for optimization. Note: `answer_relevancy` and `context_recall` may return NaN due to DashScope API prompt format incompatibility |
| **3 Evaluation Modes** | `retrieval` (initial retrieval only), `pipeline` (full RAG pipeline), `e2e` (LLM generates real answer + routing accuracy + latency stats) |
| **Routing Accuracy** | Supervisor LLM routing vs `expected_agent` comparison, per-query-type breakdown |
| **RRF Grid Search** | `scripts/grid_search_rrf.py` — composite score optimization (0.4×precision + 0.3×faithfulness + 0.3×relevancy), graph channel support |
| **A/B Comparison** | `--compare` flag generates diff report with metric deltas between two evaluation runs |
| **HTML Report** | `scripts/generate_report.py` — radar chart, bar chart, routing matrix, latency distribution |
| **CI Threshold Check** | `scripts/ci_evaluation.sh` — context_precision ≥ 0.6, faithfulness ≥ 0.7, answer_relevancy ≥ 0.6 |
| **Unit Tests** | `tests/test_evaluation.py` — golden dataset validation, RRF fusion, metrics signatures |
| **CI/CD Pipeline** | GitHub Actions: Docker services → DB init → pytest → import verification on every push |

### Observability & High Availability (v5.0)

| Feature | Description |
|---------|-------------|
| **Distributed Tracing** | OpenTelemetry SDK with manual spans on all LangGraph Agent nodes, Milvus queries, and Neo4j Cypher calls |
| **Prometheus Metrics** | `/metrics` endpoint exposes 6 custom metrics: LLM token usage, Agent routing count, vector/graph/LLM latency histograms, circuit breaker state |
| **Structured JSON Logging** | structlog replaces default logging — every log line is JSON with timestamp, level, and event fields ready for ELK/Grafana Loki ingestion |
| **Circuit Breaker** | Protects LLM and Tavily API calls: 3 failures in 60s → circuit opens → returns fallback response → auto-recovers after cooldown |
| **Graceful Degradation** | Neo4j query timeout (1.5s) → automatic fallback to pure Dense+Sparse vector retrieval with warning log |
| **Exponential Backoff Retry** | tenacity-based retry on network jitter: 3 attempts with 1s→2s→4s wait for LLM generation and Milvus writes |
| **Monitoring Stack** | Docker Compose includes Jaeger (:16686), Prometheus (:9090), and Grafana (:3000) — one command to start the full observability suite |

### Cost Optimization (v6.0)

| Feature | Description |
|---------|-------------|
| **Semantic Cache** | Milvus semantic_cache_collection + MySQL query_cache_store; cosine ≥ 0.95 → skip RAG+LLM, return cached response in ~200ms with 0 Token cost |
| **Dynamic Model Routing** | qwen-turbo for lightweight tasks (Supervisor, Direct Answer); qwen-plus/max for heavy reasoning (Data Analyst, Graph Search) |
| **Cache Singleflight** | Redis-based deduplication lock: 10 concurrent identical queries → only 1 penetrates to LLM, remaining 9 share cached result |
| **Cache Invalidation** | Document soft-delete triggers automatic cache eviction; TTL-based expiration for stale entries |
| **Benchmark Script** | `scripts/run_benchmark.py` — concurrent stress test comparing cache hit/miss latency and Token savings |

### Multimodal (v7.0)

| Feature | Description |
|---------|-------------|
| **Layout Analysis** | PyMuPDF-based PDF layout detection — separates text paragraphs from tables/images before chunking |
| **Media Extraction** | Image/table capture from PDF pages → MinIO object storage; chunks linked via `associated_media_urls` |
| **VLM Description** | Qwen-VL generates Chinese markdown descriptions for charts and tables |
| **Visual Retrieval** | 4th RRF channel: text-to-image-description semantic search via Milvus |
| **Multimodal Agent** | `multimodal_specialist` — triggered by keywords (图表/曲线/图片), retrieves visuals + generates cited answers |

### Adaptive Reasoning & Self-Correction (v8.0)

| Feature | Description |
|---------|-------------|
| **Planner Node** | Complex query decomposition into multi-step execution plans; simple queries bypass planner |
| **Critique Node** | Post-generation fact-checking: cross-verifies draft answer against retrieved contexts via LLM |
| **Self-Correction Loop** | Critique → replan → supervisor (max 2 retries); injects missing information as supplement queries |
| **direct_answer Bypass** | Chat/chitchat queries skip Critique (no retrieved context to validate against) |
| **data_analyst Bypass** | SQL query results skip Critique (structured data, not RAG-retrieved context) |

### MCP Integration (v9.0)

| Feature | Description |
|---------|-------------|
| **MCP Connection Manager** | `MCPConnectionManager` manages connections to multiple MCP Servers (SSE/stdio transport) |
| **Dynamic Tool Registration** | MCP `tools/list` Schema → LangChain `StructuredTool` auto-conversion |
| **Tool Semantic Retriever** | Milvus-based top-k tool recall prevents context window explosion (100+ tools) |
| **Data Analyst Multi-Source** | Automatically discovers MCP database tools and queries them alongside local MySQL |
| **Echarts Chart Generation** | LLM-based chart type detection + Echarts JSON config; frontend renders `echarts` code blocks |
| **MCP API Endpoints** | `POST /mcp/connect`, `GET /mcp/servers`, `POST /mcp/disconnect/{name}` |

### Ontology-Controlled Graph Extraction (v10.0)

| Feature | Description |
|---------|-------------|
| **Domain Ontology Schema** | `backend/ontology/schema.py` — 11 entity types, 12 relation predicates, 70+ valid (subject_type, predicate, object_type) rules with wildcard support |
| **Constrained Extraction Prompt** | Explicitly lists all allowed types and predicates; forbids LLM from inventing new categories |
| **Pydantic Field Validators** | `EntityInfo.type` and `RelationInfo.predicate` auto-normalized via lookup tables (handles Chinese/English synonyms, LLM hallucinated types) |
| **Post-Extraction Interceptor** | `_validate_extraction()` filters: invalid entity types, missing subject/object, rule-violating relation directions |
| **Type-Filtered Entity Resolution** | Cypher `WHERE a.type = b.type` — deduplication only within same entity type, prevents cross-type false merges |
| **Graph Topology Stats** | `scripts/graph_topology_stats.py` — node/edge/orphan counts, type/predicate distributions, degree percentiles for A/B comparison |
| **5 Evaluation Modes** | `retrieval`, `pipeline`, `e2e`, `graph` (topology snapshot), `graph_compare` (before/after diff) |

### Incremental Pipeline & DevOps (v11.0)

| Feature | Description |
|---------|-------------|
| **Document Fingerprinting** | `backend/documents/fingerprint.py` — SHA-256 file hash computed at upload time; unchanged files skip the entire pipeline |
| **DocumentIndex Activation** | `document_index` table tracks `file_hash`, `chunk_count`, `version` per document; `upsert_document_index()` handles create/skip/update lifecycle |
| **Incremental Graph Cleanup** | `cleanup_by_filename()` cascades: strip chunk IDs from edges → remove empty edges → remove orphan entities before re-insertion |
| **Async Task Queue** | `arq` (Redis-backed) dispatches ingestion to `backend/pipeline/ingestion_worker.py`; upload returns HTTP 202 immediately |
| **Sync Fallback** | If Redis is unavailable, upload falls back to synchronous processing — no availability impact |
| **Docker Compose Full Stack** | 10 services: etcd + MinIO + Milvus + Attu + Neo4j + MySQL + Redis + Jaeger + Prometheus + Grafana + API + Worker |
| **Resource Limits** | API container: 2G memory limit; Worker container: 4G memory limit (heavy LLM extraction) |

### Adaptive Retrieval & Load-Aware Degradation (v12.0)

| Feature | Description |
|---------|-------------|
| **Query Profiler** | `backend/agent/query_profiler.py` — lightweight intent classifier using keyword matching (60%) + Embedding cosine similarity (40%); classifies queries into L1 (factual), L2 (reasoning), L3 (macro summary) before Supervisor LLM |
| **Dynamic RRF Weights** | `backend/rag/dynamic_rrf.py` + `config/weight_matrix.yaml` — intent-driven weight matrix replaces static env vars; L1: Dense 70%, L2: Graph 65%, L3: balanced 35-35 |
| **Global Load Monitor** | `backend/ha/load_monitor.py` — Redis sliding-window QPS counter with 3-state machine: NORMAL (full pipeline), WARNING (skip Critique/Replan), CRITICAL (circuit-break Neo4j + Tavily) |
| **Adaptive Degradation** | `route_after_critique` checks system state before retry; `local_graph_search_node` and `web_searcher_node` degrade under CRITICAL load |
| **SSE Profiler Events** | New `query_profiler` and `system_state` events pushed to frontend for real-time intent visualization |
| **Prometheus Load Metrics** | `system_load_state` (Gauge), `query_qps` (Gauge), `query_profiler_distribution` (Counter by intent level) |
| **A/B Evaluation Script** | `scripts/run_ab_evaluation.py` — static (v11) vs dynamic (v12) comparison with per-intent RAGAS metrics and latency stats |
| **Locust Load Test** | `scripts/run_load_test.py` — concurrent load testing with weighted task distribution across L1/L2/L3 queries |

### Streaming Incremental Graph Engine (v13.0)

| Feature | Description |
|---------|-------------|
| **Incremental Graph Clustering** | `backend/graph/incremental_clustering.py` — local patching (60% neighbor consensus) + subgraph re-clustering (Louvain on affected communities only), replacing full-graph recomputation |
| **Dirty-Flag Summary Updates** | `CommunitySummary.is_dirty` boolean drives targeted regeneration — only dirty communities get LLM summaries, 80-100% Token savings |
| **Redis Streams Pipeline** | `backend/pipeline/stream_queue.py` — three-stage message bus (doc_ingest → graph_extract → vector_sync) with consumer groups, dead letter handling, and pipeline chaining |
| **Three-Stage Consumer** | `backend/pipeline/stream_consumer.py` — stateless stage handlers for parsing, LLM extraction, and Neo4j/Milvus sync |
| **Benchmark Script** | `scripts/benchmark_incremental.py` — full Louvain vs incremental comparison across graph scales (1K/5K/20K nodes) |

### Multi-Tenant RBAC & Data Isolation (v14.0)

| Feature | Description |
|---------|-------------|
| **JWT Authentication** | OAuth2 Bearer token auth via PyJWT + passlib bcrypt; `/auth/register` and `/auth/token` endpoints; `get_current_user` FastAPI dependency injects `UserContext` (tenant_id, role, access_level) into every request |
| **Tenant/User/Role Models** | `backend/auth/models.py` — `Tenant`, `User`, `Role` SQLAlchemy tables with FK relationships; auto-created on startup via `init_db()` |
| **MySQL Tenant Isolation** | `tenant_id` FK on `DocumentIndex`, `ChatSession`, `ParentChunk`, `QueryCacheStore`; `server_default="1"` for backward compatibility with existing data |
| **Milvus Pre-filtering** | `tenant_id` field added to collection schema; `retrieve_documents()` dynamically appends `expr = "tenant_id == X"` to filter before ANN search — database-level enforcement, not application-level |
| **Neo4j Subgraph Constraint** | Entity MERGE key extended to `{name, tenant_id}`; Cypher queries add `AND a.tenant_id = $tenant_id AND b.tenant_id = $tenant_id` for subgraph-scoped traversal |
| **Ingestion Pipeline Propagation** | `tenant_id` and `access_level` flow through upload endpoint → arq/Redis Streams → `ingestion_worker` → Milvus writer, Neo4j ingestion, and DocumentIndex upsert |
| **LangGraph State Extension** | `SupervisorState.user_context` dict carries tenant/role info through the entire agent graph; all worker nodes (RAG, Graph, Data Analyst) extract `tenant_id` for retrieval filtering |
| **Data Analyst SQL Isolation** | LLM prompt injects `WHERE tenant_id = X` constraint; `execute_sql` defense-in-depth blocks queries on tenant-scoped tables without `tenant_id` filter |
| **Session Scoping** | `list_session_infos()` filters by `tenant_id`; cache keys are tenant-specific to prevent cross-tenant session leakage |
| **Privilege Escalation Tests** | 4 red-team test cases (SEC001-SEC004) in golden dataset; `evaluate_security()` function checks low-privilege users cannot access high-privilege content |

### SaaS Metering, Rate Limiting & Audit Trail (v15.0)

| Feature | Description |
|---------|-------------|
| **Token Usage Tracking** | `backend/billing/token_tracker.py` — per-request `prompt_tokens`/`completion_tokens` recording to `token_usage_logs` table; `get_usage_summary()` aggregates by tenant over configurable period |
| **Per-Tenant Rate Limiting** | `backend/billing/rate_limiter.py` — `TenantRateLimiter` uses Redis sliding-window counters per `tenant_id`; rules stored in `rate_limit_rules` table with tier-based QPS/token limits |
| **Rate-Limit Middleware** | FastAPI HTTP middleware extracts tenant from JWT, checks QPS limit before request processing; returns 429 with `Retry-After` header when exceeded |
| **SLA-Aware Degradation** | `_get_tenant_degradation(state)` in orchestrator resolves tenant SLA tier at each decision point — enterprise: full pipeline even under CRITICAL; premium: skips Critique at CRITICAL; free: skips Critique at WARNING, cache-only at CRITICAL |
| **Audit Trail** | `backend/billing/audit.py` — immutable `audit_logs` table records every MCP tool call, SQL execution, and HITL event with `risk_level` classification |
| **Audit Context Manager** | `AuditContext` wraps operations with before/after semantics; automatically logs exceptions as `risk_level="high"` |
| **Billing API** | `GET /billing/usage` returns token consumption summary; `GET /billing/audit` returns paginated audit logs with action filter — both tenant-scoped |
| **HITL Webhook** | `HITL_WEBHOOK_URL` env var triggers POST notification to tenant admin on interrupt events; non-blocking `asyncio.create_task`, 5s timeout |
| **Frontend Auth** | Login/register UI with JWT token persistence in `localStorage`; `_authFetch()` wrapper injects `Authorization: Bearer` on all API calls; auto-logout on 401 |
| **Config Validation** | `backend/config.py` — Pydantic `BaseSettings` validates all env vars at startup; no hardcoded fallback secrets (missing JWT_SECRET or DATABASE_URL → fatal error) |
| **SQL Execution Safety** | Four-layer defense: SELECT-only guard, multi-statement `;` rejection, `SET TRANSACTION READ ONLY`, tenant-scoped table `tenant_id` filter enforcement |
| **Upload Validation** | File size capped at configurable `UPLOAD_MAX_SIZE_MB` (default 50MB); oversized uploads rejected with 400 before disk I/O |
| **Database Migrations** | Alembic initialized at `alembic/` — schema changes via `alembic revision --autogenerate` + `alembic upgrade head` |
| **OTel OTLP Support** | Tracing auto-detects `OTEL_EXPORTER_OTLP_ENDPOINT` — uses gRPC `OTLPSpanExporter` in production, `ConsoleSpanExporter` in development |
| **Session List Optimization** | Batch queries (`GROUP BY` + `func.count`/`func.min`) instead of N+1 per-session fetches |
| **Test Coverage** | 65 tests covering auth, billing, rate limiting, audit, load monitor, data analyst SQL safety, SLA degradation |

### Agent Workflow Platform (v16.0)

| Feature | Description |
|---------|-------------|
| **Workflow Planner** | `backend/workflow/planner.py` — LLM 将自然语言目标拆解为 DAG 执行计划（WorkflowPlan），自动分析步骤依赖关系 |
| **Workflow Executor** | `backend/workflow/executor.py` — 独立 LangGraph DAG 执行引擎，串行+并行，MySQL Checkpointer 持久化支持断点续跑 |
| **WorkflowTool Abstraction** | `backend/workflow/tool_runtime.py` — 统一工具抽象层，6 个 Agent 注册到 ToolRegistry，轻量 LLM 调用替代完整 agent node |
| **Artifact System** | `backend/workflow/artifact.py` — Report(Markdown LLM)、Excel(openpyxl)、Chart(Echarts)、CSV 交付物，持久化到 `workflow_artifacts` |
| **Workflow API** | `POST /workflows/plan`, `POST /workflows/execute`, `GET /workflows/{id}/status`, `GET /workflows/{id}/artifacts` |
| **Frontend Panel** | Vue 3 任务工作流面板：目标输入→Plan DAG 可视化→执行进度条→产物查看→历史记录回溯 |
| **Model** | 3 张新表：`workflow_definitions`, `workflow_executions`, `workflow_artifacts`；Alembic 管理迁移 |
| **Tests** | 19 个 workflow 单元测试（tool runtime + planner + executor），23 total 全绿 |

### Adaptive GraphRAG (v17.0)

| Feature | Description |
|---------|-------------|
| **6-Type Query Classification** | `backend/agent/query_profiler.py` — factoid/entity_relation/multi_hop/global_summary/temporal/comparison 六种类型，关键词+Embedding 混合分类 |
| **RetrievalPlanner** | `backend/rag/retrieval_planner.py` — 查询类型→RetrievalPlan（通道选择+图深度+融合策略），factoid 跳 Neo4j，multi_hop 3-hop |
| **Adaptive RRF Weights** | `config/weight_matrix.yaml` — 6 种类型独立 RRF 权重（factoid: Dense=0.8/Graph=0，multi_hop: Graph=0.85） |
| **GraphUtilityEstimator** | `backend/rag/graph_utility_estimator.py` — 5 维启发式特征预测图检索价值，score<0.35 跳过 Neo4j（零 LLM 调用） |
| **Orchestrator Integration** | `local_graph_search_node` + `global_graph_search_node` 动态读取 intent，条件跳过图检索/社区摘要 |
| **Evaluation** | 23 条 benchmark，3 项评测：分类 73.9%, Plan 决策 91.3%, Overall 78.3%；50 测试全绿 |

### Graph Reasoning Engine (v18.0)

| Feature | Description |
|---------|-------------|
| **ReasoningPlanner** | `backend/rag/graph_reasoning/planning.py` — NL→结构化 ReasoningPlan（起始实体+目标关系+最大跳数） |
| **SubgraphRetriever** | `backend/rag/graph_reasoning/subgraph.py` — 多跳 Cypher 抽取 Neo4j 子图为 NetworkX DiGraph |
| **PathExplorer** | `backend/rag/graph_reasoning/path_explorer.py` — BFS + Beam Search 候选推理路径发现 |
| **PathRanker** | `backend/rag/graph_reasoning/path_ranker.py` — 4 维加权排序（语义+置信度+时序+长度） |
| **ReasoningVerifier** | `backend/rag/graph_reasoning/verifier.py` — LLM 答案-路径交叉验证（SUPPORTED/PARTIAL/UNSUPPORTED） |
| **Multi-Hop Fix** | `graph_retriever.py` 真正 n-hop 循环扩展，不再仅 1-hop |
| **Tests** | 47 测试全绿（15 reasoning + 32 regression） |

### Memory Graph System (v19.0)

| Feature | Description |
|---------|-------------|
| **Memory Schemas** | `backend/memory/schemas.py` — 4 种记忆类型：Fact/Preference/Task/Relation |
| **MemoryGraphStore** | `backend/memory/store.py` — Neo4j `:Memory` 节点 MERGE + `:MENTIONS` 关系链接知识图谱 Entity |
| **MemoryExtractor** | `backend/memory/extractor.py` — LLM 从对话末尾 10 条消息提取结构化记忆（JSON 输出） |
| **MemoryImportance** | `backend/memory/importance.py` — 时间衰减（30 天半衰期）+ 访问频次三维评分 |
| **MemoryRetriever** | `backend/memory/retriever.py` — 查询时检索用户记忆并格式化为 `## 用户记忆` 注入 LLM 上下文 |
| **Brain Hook** | `chat_with_agent` / `chat_with_agent_stream` 保存后异步提取，非阻塞 |
| **Config Toggle** | `memory_enabled: bool = False` 配置开关，默认关闭 |
| **Tests** | 57 测试全绿（10 memory + 47 regression） |

### Deep Research Engine (v20.0)

| Feature | Description |
|---------|-------------|
| **Research Planner** | `backend/research/planner.py` — LLM 将研究目标拆解为 DAG 执行计划，自动分析任务依赖关系 |
| **Research Executor** | `backend/research/executor.py` — DAG 执行引擎，串行+并行调度 Research Agent，断点恢复 + 实时进度持久化 |
| **Research Agents** | `backend/research/research_agents.py` — Web/Graph/Data/Internal KB 四大代理，统一输出结构化 Evidence |
| **Evidence Store** | `backend/research/evidence_store.py` — 证据持久化 + 多维度查询 + 覆盖率统计 |
| **Research Reviewer** | `backend/research/reviewer.py` — 4 维加权评分（覆盖率/多样性/引用/置信度），阈值 0.70 |
| **Gap Analyzer** | `backend/research/gap_analyzer.py` — LLM 分析证据缺口 → 自动生成补充检索查询 |
| **Report Generator** | `backend/research/report_generator.py` — 证据驱动中文报告（Markdown），每条结论绑定 Evidence ID |
| **Artifact Extension** | `backend/workflow/artifact.py` — 新增 PDF (reportlab) + PPTX (python-pptx) 生成 |
| **Research API** | `backend/research/routes.py` — POST /create, GET /{id}/status/evidence/report, POST /cancel, GET /list |
| **Frontend Panel** | 研究工作区标签页：目标输入 → 实时进度条 → 证据卡片查看 → 报告阅读 → 历史回溯 |
| **Config** | `research_enabled: bool = True`, `research_max_review_rounds: int = 3` |
| **Tests** | 16 测试全绿（schemas + reviewer + gap_analyzer + evidence_store + planner + executor） |

---

## Tech Stack

<table>
<tr>
<td><strong>Backend</strong></td>
<td>FastAPI · Uvicorn · LangChain · LangGraph · Pydantic · SQLAlchemy</td>
</tr>
<tr>
<td><strong>Frontend</strong></td>
<td>Vue 3 (CDN) · marked.js · highlight.js · Font Awesome</td>
</tr>
<tr>
<td><strong>Vector Store</strong></td>
<td>Milvus 2.5 (HNSW + SPARSE_INVERTED_INDEX)</td>
</tr>
<tr>
<td><strong>Graph Store</strong></td>
<td>Neo4j 5.26 (Community Edition)</td>
</tr>
<tr>
<td><strong>Embedding</strong></td>
<td>Qwen text-embedding-v1 (1536-dim) · BM25 (custom impl.)</td>
</tr>
<tr>
<td><strong>LLM</strong></td>
<td>Qwen-Plus / Qwen3.6-Plus via DashScope (OpenAI-compatible API)</td>
</tr>
<tr>
<td><strong>Database</strong></td>
<td>MySQL 8.0 (sessions, messages, parent chunks, community summaries, graph checkpoints) · Redis 7.0 (hot cache, HITL locks)</td>
</tr>
<tr>
<td><strong>Graph Algorithms</strong></td>
<td>NetworkX · python-louvain (Leiden/Louvain community detection)</td>
</tr>
<tr>
<td><strong>Search APIs</strong></td>
<td>Tavily (web search) · Gaode/Amap (weather)</td>
</tr>
<tr>
<td><strong>Evaluation</strong></td>
<td>Ragas 0.2.15 · matplotlib · pytest · 3 eval modes · HTML reports (v4.0)</td>
</tr>
<tr>
<td><strong>Observability</strong></td>
<td>OpenTelemetry · Prometheus · Grafana · Jaeger · structlog (v5.0)</td>
</tr>
<tr>
<td><strong>High Availability</strong></td>
<td>pybreaker · tenacity · Neo4j query timeout (v5.0)</td>
</tr>
<tr>
<td><strong>Cost Optimization</strong></td>
<td>Semantic Cache (Milvus ANN) · Dynamic Model Routing (v6.0)</td>
</tr>
<tr>
<td><strong>Infrastructure</strong></td>
<td>Docker Compose (Milvus + etcd + MinIO + Attu + Neo4j + MySQL + Redis + Jaeger + Prometheus + Grafana + API + Worker) · GitHub Actions CI · Dockerfile</td>
</tr>
<tr>
<td><strong>Auth & Multi-Tenancy</strong></td>
<td>PyJWT · passlib[bcrypt] · FastAPI OAuth2 Depends · Tenant/User/Role models · Milvus pre-filtering · Neo4j subgraph constraint (v14.0)</td>
</tr>
<tr>
<td><strong>Async Pipeline</strong></td>
<td>arq (Redis-backed task queue) · Async ingestion worker · Sync fallback (v11.0)</td>
</tr>
<tr>
<td><strong>MCP Integration</strong></td>
<td>MCP Python SDK · SSE/stdio transport · Dynamic tool registration · Tool semantic retriever (v9.0)</td>
</tr>
<tr>
<td><strong>Visualization</strong></td>
<td>Echarts · Chart type auto-detection · Markdown echarts code block rendering (v9.0)</td>
</tr>
</table>

---

## Project Structure

```
Ragent-AI/
├── backend/
│   ├── api/
│   │   ├── app.py              # FastAPI application factory, CORS, middleware
│   │   └── routes.py           # REST API routes (chat, sessions, documents, HITL)
│   ├── auth/                   # Multi-tenant RBAC (v14.0)
│   │   ├── __init__.py
│   │   ├── models.py           # Tenant, User, Role SQLAlchemy models
│   │   ├── jwt_handler.py      # JWT encode/decode, password hashing
│   │   ├── dependencies.py     # UserContext dataclass, get_current_user dependency
│   │   └── routes.py           # /auth/register, /auth/token endpoints
│   ├── agent/
│   │   ├── brain.py            # Conversation storage, SSE streaming, HITL resume
│   │   ├── orchestrator.py     # LangGraph graph: 6 agents + synthesize + planner + critique + replan (v8)
│   │   ├── tools.py            # Agent tools (weather, knowledge base, web search, graph steps)
│   │   ├── model_router.py     # Dynamic LLM routing: turbo/plus/max by task
│   │   ├── web_searcher.py     # Tavily web search integration + RAG fallback
│   │   ├── data_analyst.py     # Text-to-SQL + MCP multi-data-source query
│   │   ├── multimodal_specialist.py  # Visual retrieval: image/table description + Milvus search
│   │   ├── mcp_client.py       # MCP connection manager (SSE/stdio transport)
│   │   ├── chart_generator.py  # Echarts chart generation (type detection + config)
│   │   └── tool_retriever.py   # MCP tool semantic retriever (Milvus top-k recall)
│   ├── rag/
│   │   ├── pipeline.py         # LangGraph RAG workflow (retrieve → grade → rewrite → expanded)
│   │   ├── utils.py            # Hybrid retrieval, reranking, auto-merging, query expansion, 4-ch RRF
│   │   ├── graph_retriever.py  # Graph-enhanced retrieval (local search + global search)
│   │   └── visual_retriever.py # Visual retrieval: text-to-image-description semantic search
│   ├── documents/
│   │   ├── loader.py           # Three-level hierarchical document chunking (PDF/Word/Excel/MD)
│   │   ├── graph_extractor.py  # LLM entity/relation extraction from L2 chunks (ontology-controlled v10)
│   │   └── fingerprint.py      # SHA-256 file/chunk content fingerprinting (v11.0)
│   ├── ontology/               # Domain ontology constraint layer (v10.0)
│   ├── pipeline/               # Async ingestion pipeline (v11.0)
│   │   ├── __init__.py
│   │   ├── task_queue.py       # arq Redis task queue configuration
│   │   └── ingestion_worker.py # Async document ingestion worker
│   │   ├── __init__.py
│   │   └── schema.py           # Entity types, relation predicates, triple rules, validation functions
│   │   ├── layout_analyzer.py  # PDF layout analysis (text/image/table separation)
│   │   ├── media_extractor.py  # Image/table extraction + MinIO upload
│   │   └── vlm_descriptor.py   # Qwen-VL chart/table description generation
│   ├── embedding/
│   │   └── service.py          # Dense (Qwen API) + Sparse (BM25) embedding service
│   ├── milvus/
│   │   ├── client.py           # Milvus vector DB client (hybrid search, RRF, CRUD)
│   │   └── writer.py           # Batch vectorization & Milvus ingestion
│   ├── storage/
│   │   ├── database.py         # MySQL connection, session factory, table init
│   │   ├── models.py           # SQLAlchemy ORM (sessions, messages, chunks, summaries, checkpoints)
│   │   ├── cache.py            # Redis cache utility + HITL distributed lock
│   │   ├── checkpointer.py     # MySQLSaver — LangGraph state persistence
│   │   ├── parent_chunk_store.py  # Parent chunk storage (MySQL + Redis cache)
│   │   ├── doc_lifecycle.py    # Document lifecycle: soft-delete, chunk ID query (v4.0)
│   │   ├── graph_client.py     # Neo4j driver wrapper (run_cypher / write_cypher)
│   │   ├── graph_schema.py     # Neo4j constraints & indexes initialization
│   │   ├── graph_ingestion.py  # Batch MERGE entities + relations into Neo4j
│   │   └── graph_cleanup.py    # Neo4j cascade cleanup (orphan edges/nodes) (v4.0)
│   ├── graph/
│   │   ├── community.py        # Leiden clustering, community summarization, Milvus indexing
│   │   └── entity_resolution.py # Two-stage entity dedup (edit-distance + LLM + merge) (v4.0)
│   ├── evaluation/             # Automated RAG evaluation (v4.0)
│   │   ├── __init__.py
│   │   ├── dataset.py          # Golden dataset loader (80 QA pairs, expected_agent)
│   │   └── metrics.py          # Ragas metrics + generate_answer + routing accuracy
│   ├── observability/          # OpenTelemetry + Prometheus + structlog (v5.0)
│   │   ├── __init__.py
│   │   ├── tracing.py          # OTel init + manual span utilities
│   │   ├── metrics.py          # Prometheus metrics + /metrics endpoint
│   │   └── logging.py          # Structlog JSON logging configuration
│   ├── ha/                     # High Availability modules (v5.0)
│   │   ├── __init__.py
│   │   ├── circuit_breaker.py  # Circuit breaker state machine
│   │   ├── retry.py            # Exponential backoff retry decorator
│   │   └── degradation.py      # Neo4j timeout → Dense+Sparse fallback
│   ├── cache/                  # Semantic cache layer (v6.0)
│   │   ├── __init__.py
│   │   ├── semantic_cache.py   # Milvus ANN + cosine + MySQL store
│   │   ├── singleflight.py     # Redis Singleflight anti-stampede
│   │   └── invalidation.py     # Document delete → cache eviction
│   ├── memory/                 # Memory Graph System (v19.0)
│   │   ├── __init__.py
│   │   ├── schemas.py          # MemoryNode, MemoryType, MemoryExtraction
│   │   ├── extractor.py        # LLM 从对话提取结构化记忆
│   │   ├── store.py            # Neo4j :Memory 节点 CRUD
│   │   ├── retriever.py        # 查询用户记忆注入 LLM 上下文
│   │   └── importance.py       # 时间衰减 + 频次评分
│   ├── research/               # Deep Research Engine (v20.0)
│   │   ├── __init__.py
│   │   ├── schemas.py          # ResearchPlan, Evidence, ReviewResult, GapAnalysis
│   │   ├── models.py           # ORM: ResearchExecution, ResearchEvidence, ResearchReportRecord
│   │   ├── planner.py          # Goal → DAG 执行计划
│   │   ├── executor.py         # DAG 执行 + 审核循环 + 实时进度
│   │   ├── evidence_store.py   # 证据持久化 + 多维查询
│   │   ├── research_agents.py  # Web/Graph/Data/Internal KB 研究代理
│   │   ├── reviewer.py         # 4 维证据评分
│   │   ├── gap_analyzer.py     # 证据缺口分析 + 补充检索
│   │   ├── report_generator.py # 证据驱动中文报告
│   │   └── routes.py           # /research/* API 端点
│   └── schemas.py              # Pydantic: Chat*, Document*, HITL*, GraphEntity, QueryPlan, CritiqueResult
│
├── scripts/
│   ├── run_community_clustering.py  # Offline: build graph → cluster → summarize → index
│   ├── run_entity_resolution.py    # Offline: entity dedup pipeline (v4.0)
│   ├── run_evaluation.py           # RAG eval: 5 modes + latency + A/B compare (v4.0/v10.0)
│   ├── graph_topology_stats.py     # Graph topology metrics for A/B comparison (v10.0)
│   ├── grid_search_rrf.py          # RRF weight optimization (composite score, graph channel)
│   ├── generate_report.py          # HTML evaluation report generator
│   ├── ci_evaluation.sh            # CI threshold check script
│   └── run_benchmark.py            # Concurrent cache benchmark (v6.0)
│
├── frontend/
│   ├── index.html              # Vue 3 SPA (chat, trace canvas, HITL modal, settings)
│   ├── script.js               # Application logic, SSE handler, API integration
│   ├── style.css               # Gemini-inspired dual-theme (Light/Dark)
│   └── logo.svg                # Application logo
│
├── tests/
│   ├── test_doc_lifecycle.py   # Soft-delete unit tests (v4.0)
│   ├── test_evaluation.py      # Evaluation pipeline unit tests
│   ├── test_v10_ontology.py    # v10 ontology schema + extraction validation tests (53 tests)
│   ├── test_fingerprint.py     # Document fingerprint SHA-256 unit tests
│   └── test_incremental_upload.py  # Incremental upload integration tests
│   └── golden_dataset.json     # 80-item evaluation dataset (v4.0)
│
├── data/
│   └── documents/              # Uploaded document storage
│
├── docs/
│   ├── planning/                # Feature specification documents
│   │   ├── 5.23todov2.md        # v2.0 — Multi-Agent + HITL specification
│   │   ├── 5.24todov3.md        # v3.0 — GraphRAG requirements specification
│   │   ├── 5.25todolistv4.md    # v4.0 — UI optimization & planning
│   │   ├── 5.25todov5.md        # v5.0 — Observability & HA specification
│   │   ├── 5.25todov6.md        # v6.0 — Cost optimization specification
│   │   ├── 5.25todov7.md        # v7.0 — Multimodal upgrade specification
│   │   ├── 5.30todov8.md        # v8.0 — Adaptive reasoning & self-correction loop
│   │   ├── 5.30todov9.md        # v9.0 — MCP integration & data federation
│   │   └── 5.31tdov10.md        # v10.0 — Ontology-controlled graph extraction
│   │   └── GraphRAG-v3.0-升级计划.md  # v3.0 — Implementation plan (5 phases)
│   ├── superpowers/plans/       # Detailed implementation plans
│   └── img.png                  # Application screenshot
│
├── docker-compose.yml          # Full stack (Milvus + Neo4j + Jaeger + Prometheus + Grafana)
├── docker-compose.ci.yml       # CI environment services (v4.0)
├── Dockerfile                  # Application container image (v4.0)
├── prometheus.yml              # Prometheus scrape configuration (v5.0)
├── .github/workflows/ci.yml    # GitHub Actions CI pipeline (v4.0)
├── pyproject.toml              # Python dependencies & project metadata
├── start.py                    # UTF-8 startup script (uvicorn wrapper)
├── start_worker.py             # arq async ingestion worker entrypoint (v11.0)
├── .env.example                # Environment configuration template
└── .env                        # Local environment configuration (gitignored)
```

---

## Getting Started

### Prerequisites

- **Python** 3.12+
- **Docker** & Docker Compose
- **MySQL** 8.0+
- **Redis** 7.0+
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip

### 1. Clone & Install

```bash
git clone https://github.com/your-username/Ragent-AI.git
cd Ragent-AI

# Option A: uv (recommended)
uv sync

# Option B: pip
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your keys:

```env
# ===== LLM (DashScope / Qwen) =====
ARK_API_KEY=your_dashscope_api_key
MODEL=qwen-plus
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDER=text-embedding-v1
GRADE_MODEL=qwen-plus
SUPERVISOR_MODEL=qwen-plus
MAX_TOKENS=8192

# ===== Database =====
DATABASE_URL=mysql+pymysql://root:password@localhost:3306/agent_chat
REDIS_URL=redis://localhost:6379/0

# ===== Milvus =====
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_VECTOR_DIM=1536
MILVUS_SEARCH_TOP_K=20

# ===== Neo4j (GraphRAG) =====
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# ===== Rerank (optional, degrades gracefully) =====
RERANK_MODEL=qwen3-rerank
RERANK_BINDING_HOST=https://dashscope.aliyuncs.com/compatible-mode/v1
RERANK_API_KEY=your_dashscope_api_key
RERANK_TOP_K=10

# ===== Web Search (optional) =====
TAVILY_API_KEY=your_tavily_api_key

# ===== Tools (optional) =====
AMAP_WEATHER_API=https://restapi.amap.com/v3/weather/weatherInfo
AMAP_API_KEY=your_amap_key
```

### 3. Start Infrastructure

```bash
# Start full stack (Milvus + Neo4j)
docker compose up -d

# Verify health
docker compose ps
```

| Service | Port | Description |
|---------|------|-------------|
| Milvus | 19530 | Vector database (gRPC) |
| Milvus Health | 9091 | Health check endpoint |
| MinIO | 9000/9001 | Object storage / Console |
| Attu | 8080 | Milvus web management UI |
| Neo4j | 7474 | Neo4j Browser (HTTP) |
| Neo4j Bolt | 7687 | Neo4j driver protocol |

### 4. Create Database

```sql
CREATE DATABASE IF NOT EXISTS agent_chat CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Tables (`chat_sessions`, `chat_messages`, `parent_chunks`, `community_summaries`, `graph_checkpoints`, `graph_checkpoint_writes`) are auto-created on first launch via SQLAlchemy's `Base.metadata.create_all()`.

### 5. Launch Application

```bash
# Option A: uv
uv run python start.py

# Option B: python
python start.py
```

Open in browser:
- **Frontend**: http://127.0.0.1:8000/
- **API Docs**: http://127.0.0.1:8000/docs
- **Neo4j Browser**: http://localhost:7474

### 6. Run GraphRAG Offline Pipeline

After uploading documents, run the community clustering script:

```bash
uv run python scripts/run_community_clustering.py
```

This will:
1. Pull the full knowledge graph from Neo4j
2. Run Leiden community detection
3. Generate LLM-powered community summaries
4. Index summaries into Milvus for global search

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARK_API_KEY` | — | DashScope API key (required) |
| `MODEL` | `qwen-plus` | Chat model for Worker agents |
| `SUPERVISOR_MODEL` | `qwen-plus` | Model for Supervisor routing (falls back to MODEL) |
| `BASE_URL` | — | LLM API endpoint (OpenAI-compatible) |
| `EMBEDDER` | `text-embedding-v1` | Embedding model name |
| `GRADE_MODEL` | `qwen-plus` | Document grading model |
| `MAX_TOKENS` | `8192` | Max output tokens |
| `DATABASE_URL` | `mysql+pymysql://...` | MySQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `MILVUS_HOST` | `127.0.0.1` | Milvus server host |
| `MILVUS_PORT` | `19530` | Milvus server port |
| `MILVUS_COLLECTION` | `embeddings_collection` | Milvus collection name |
| `MILVUS_VECTOR_DIM` | `1536` | Embedding dimension |
| `MILVUS_SEARCH_TOP_K` | `20` | Initial retrieval candidate count |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt protocol URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `RERANK_MODEL` | `qwen3-rerank` | Rerank model name (qwen3-rerank via native API, gte-rerank via compatible API) |
| `RERANK_TOP_K` | `10` | Rerank output candidates |
| `TAVILY_API_KEY` | — | Tavily web search API key (optional) |
| `AMAP_API_KEY` | — | Gaode weather API key (optional) |
| `AUTO_MERGE_ENABLED` | `true` | Enable hierarchical auto-merging |
| `AUTO_MERGE_THRESHOLD` | `2` | Min sibling chunks to trigger merge |
| `LEAF_RETRIEVE_LEVEL` | `3` | Leaf chunk level for retrieval |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Max web search results |
| `RRF_WEIGHT_DENSE` | `0.4` | RRF dense channel weight (v4.0) |
| `RRF_WEIGHT_SPARSE` | `0.3` | RRF sparse channel weight (v4.0) |
| `RRF_WEIGHT_GRAPH` | `0.3` | RRF graph channel weight (v4.0) |
| `ENTITY_SIM_THRESHOLD` | `0.75` | Entity dedup edit-distance threshold (v4.0) |
| `OTEL_ENABLED` | `false` | Enable OpenTelemetry tracing (v5.0) |
| `METRICS_ENABLED` | `true` | Enable Prometheus /metrics endpoint (v5.0) |
| `LOG_LEVEL` | `INFO` | Log level: DEBUG / INFO / WARNING (v5.0) |
| `LOG_FORMAT` | `json` | Log format: json / console (v5.0) |
| `NEO4J_QUERY_TIMEOUT` | `1.5` | Neo4j Cypher query timeout in seconds (v5.0) |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Semantic cache cosine similarity threshold (v6.0) |
| `CACHE_TTL_SECONDS` | `86400` | Cache entry TTL in seconds (v6.0) |
| `MODEL_TURBO` | `qwen-turbo` | Lightweight task model (v6.0) |
| `MODEL_MAX` | `qwen-max` | Heavy reasoning model (v6.0) |

### Chunking Parameters

| Level | Chunk Size | Overlap | Purpose |
|-------|-----------|---------|---------|
| L1 (Root) | 1200 chars | 240 chars | Topical context unit · Graph extraction source |
| L2 (Mid) | 600 chars | 120 chars | Intermediate grouping · Graph extraction source |
| L3 (Leaf) | 300 chars | 60 chars | Vectorized retrieval unit |

---

## API Reference

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Synchronous chat (returns full response) |
| `POST` | `/chat/stream` | SSE streaming chat with agent status events |

### HITL (Human-in-the-Loop)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat/hitl/resume` | Resume a paused graph after human intervention |

**SSE Event Types** (streamed during `/chat/stream`):

| Event | Description |
|-------|-------------|
| `routing` | Supervisor routing decision (agent + reason) |
| `agent_start` | Agent node began execution |
| `agent_done` | Agent node completed execution |
| `rag_step` | RAG / Graph pipeline step (retrieval, grading, rewriting, graph expansion) |
| `graph_expand` | Local graph search event (entity lookup, hop expansion) |
| `community_match` | Global graph search event (community summary matching) |
| `content` | Answer text chunk |
| `worker_content` | Worker-level answer for trace panel |
| `trace` | Full RAG pipeline trace (audit data) |
| `agent_trace` | Full agent-level trace (routing, fallback, graph mode, triples count, etc.) |
| `hitl_interrupt` | HITL interrupt triggered (graph paused, lock acquired) |
| `error` | Error message |

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/sessions` | List all sessions |
| `GET` | `/sessions/{id}` | Get session messages |
| `DELETE` | `/sessions/{id}` | Delete a session |

### Metrics (v5.0)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/metrics` | Prometheus metrics endpoint (token usage, latency, routing, circuit breaker) |

### Research (v20.0)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/research/create` | Create and start a new research task |
| `GET` | `/research/list` | List user's research executions |
| `GET` | `/research/{id}` | Get research status + progress |
| `GET` | `/research/{id}/evidence` | List collected evidence items |
| `GET` | `/research/{id}/report` | Get generated research report |
| `POST` | `/research/{id}/cancel` | Cancel running research |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/documents` | List all documents with chunk counts |
| `POST` | `/documents/upload` | Upload & vectorize a document (SSE progress); triggers graph extraction |
| `DELETE` | `/documents/{filename}` | Delete document & its vectors |

> Full interactive API documentation is available at `/docs` (Swagger UI) when the application is running.

---

## Roadmap

### v2.0 — Multi-Agent + HITL ✓

- [x] Multi-Agent Supervisor-Workers architecture (4 agents)
- [x] Parallel worker dispatch via LangGraph `Send`
- [x] Data Analyst (Text-to-SQL) agent
- [x] HITL interrupt/resume (RAG low confidence + SQL approval)
- [x] MySQL LangGraph checkpointer (state persistence)
- [x] Redis distributed lock (HITL concurrency guard)
- [x] Web search → RAG automatic fallback
- [x] Dead-loop detection (recursion_limit=15)
- [x] Frontend Trace Canvas (agent status + timeline)
- [x] Frontend HITL approval modal
- [x] Gemini-inspired dual-theme UI

### v3.0 — GraphRAG Semantic Network ✓

- [x] Neo4j graph database deployment (Docker Compose)
- [x] Graph client + schema initialization (constraints & indexes)
- [x] LLM entity/relation extraction during document ingestion
- [x] Neo4j MERGE with entity deduplication + source_chunk binding
- [x] Local Graph Search node (vector → Neo4j entity → 1-hop expansion)
- [x] Global Graph Search node (community summary matching)
- [x] Leiden (Louvain) community detection + hierarchical summarization
- [x] Community summaries indexed to Milvus + MySQL
- [x] Supervisor routing updated with graph search nodes
- [x] Three-channel RRF fusion (dense + sparse + graph)
- [x] Frontend agent labels for graph search workers
- [x] SSE graph_expand / community_match events (brain.py)
- [x] Frontend trace panel graph event handling (script.js)

### v4.0 — Knowledge Governance & Evaluation Pipeline ✓

- [x] Cross-database cascading soft-delete (MySQL → Milvus → Neo4j)
- [x] Document lifecycle state machine (DocumentIndex table + versioning)
- [x] Neo4j orphan node/edge garbage collection
- [x] Two-stage entity resolution (edit-distance + LLM confirmation + Cypher merge)
- [x] Temporal knowledge graph (valid_from / valid_to on entities and relations)
- [x] Temporal sensitivity routing in Supervisor
- [x] Golden evaluation dataset (80 QA pairs across 7 query types with expected_agent)
- [x] Ragas automated evaluation pipeline (4 metrics: precision, recall, faithfulness, relevancy; ragas 0.2.15, DashScope API partial compatibility)
- [x] 3 evaluation modes: retrieval, pipeline, e2e (with LLM answer generation)
- [x] Routing accuracy evaluation (Supervisor vs expected_agent)
- [x] RRF weight grid search with composite score optimization + graph channel
- [x] A/B comparison report with metric diffs
- [x] HTML evaluation report (radar chart + bar chart + routing matrix + latency)
- [x] CI threshold check script (ci_evaluation.sh)
- [x] Evaluation unit tests (test_evaluation.py)
- [x] RRF weights configurable via environment variables
- [x] GitHub Actions CI/CD pipeline
- [x] Dockerfile for application containerization
- [x] Entity resolution CLI script (`scripts/run_entity_resolution.py`)

### v5.0 — Observability & High Availability ✓

- [x] OpenTelemetry distributed tracing (manual spans on Agent nodes + Milvus + Neo4j)
- [x] Prometheus `/metrics` endpoint (6 custom metrics)
- [x] structlog structured JSON logging
- [x] Circuit breaker for LLM and Tavily API (3 failures → OPEN → fallback)
- [x] Neo4j query timeout with graceful degradation to Dense+Sparse
- [x] Exponential backoff retry for LLM generation and DB writes
- [x] Docker Compose monitoring stack (Jaeger + Prometheus + Grafana)
- [x] Grafana pre-configured with Prometheus data source

### v6.0 — Cost & Latency Optimization ✓

- [x] Semantic cache layer (Milvus ANN + MySQL store, cosine ≥ 0.95 threshold)
- [x] Dynamic model routing (qwen-turbo for lightweight, qwen-plus/max for heavy)
- [x] Redis Singleflight cache stampede protection
- [x] Event-driven cache invalidation on document soft-delete
- [x] TTL-based cache expiration
- [x] Concurrent benchmark script (cache hit/miss latency + Token comparison)

### v7.0 — Multimodal Upgrade ✓

- [x] PyMuPDF layout analysis for PDF (text/image/table separation)
- [x] Image/table extraction + MinIO upload + associated_media_urls
- [x] Qwen-VL chart/table description generation
- [x] 4-channel RRF fusion (Dense + Sparse + Graph + Visual)
- [x] Neo4j ImageNode/TableNode constraints
- [x] Multimodal Specialist Agent (keyword-triggered visual retrieval)
- [x] Supervisor routing updated with multimodal route

### v8.0 — Adaptive Reasoning & Self-Correction Loop ✓

- [x] GraphState extension: `query_plan`, `critique_result`, `retry_count`, `draft_answer`
- [x] Planner node: complex query decomposition into multi-step execution plans
- [x] Critique node: LLM-driven cross-verification of draft answers against retrieved contexts
- [x] Replan node: inject missing information as supplement queries for re-retrieval
- [x] Self-correction loop: Critique → replan → supervisor (max 2 retries)
- [x] New SSE events: `plan_generated`, `critique_feedback`, `self_correction`
- [x] Frontend Trace Canvas: planner/critique agent nodes with distinct styling
- [x] Fix: multimodal_specialist missing edge to synthesize
- [x] direct_answer bypasses Critique (闲聊无检索上下文，跳过事实核查避免无效重试)

### v9.0 — MCP Integration & Data Federation ✓

- [x] MCP connection manager: SSE/stdio transport, tools/list, tools/call
- [x] Dynamic tool registration: MCP tools → LangChain StructuredTool auto-conversion
- [x] Data Analyst multi-source: local MySQL + MCP external databases (PostgreSQL, Salesforce, etc.)
- [x] Tool semantic retriever: Milvus-based top-k tool recall (prevents context window explosion)
- [x] Echarts chart generation: LLM-based chart type detection + config generation
- [x] Frontend Echarts rendering: markdown `echarts` code block → live chart
- [x] MCP SSE events: `mcp_tool_call`, `mcp_tool_result` in Trace Canvas
- [x] Planner DAG support: dependencies + input_mapping for multi-step workflows
- [x] SupervisorState `tool_outputs` for cross-step data passing

### v10.0 — Ontology-Controlled Graph Extraction ✓

- [x] Domain ontology schema: 11 entity types, 12 relation predicates, 70+ triple rules
- [x] Pydantic field validators: auto-normalize LLM hallucinated types/predicates
- [x] Constrained extraction prompt: explicit type/predicate whitelist
- [x] Post-extraction interceptor: `_validate_extraction()` filters violations
- [x] Manual JSON parsing: DashScope `source`/`target` → `subject`/`object` field mapping
- [x] Type-filtered entity resolution: `a.type = b.type` Cypher constraint
- [x] Graph topology stats script: node/edge/orphan/type/predicate/degree metrics
- [x] Extended evaluation: `graph` and `graph_compare` modes in `run_evaluation.py`
- [x] Topology charts in HTML report: type distribution, before/after comparison
- [x] 53 unit tests + full integration test (extraction → ingestion → topology stats)

### v11.0 — Incremental Pipeline & DevOps ✓

- [x] Document fingerprinting: SHA-256 file hash, skip unchanged uploads
- [x] DocumentIndex activation: file_hash, chunk_count, version tracking
- [x] Incremental graph cleanup: `cleanup_by_filename()` cascade (strip → prune → orphans)
- [x] Fix Milvus `is_deleted` phantom field: set `False` on insert
- [x] Async task queue: `arq` (Redis-backed) + `ingestion_worker.py`
- [x] Sync fallback: graceful degradation if Redis unavailable
- [x] Docker Compose full stack: MySQL, Redis, API, Worker services with resource limits
- [x] 65 unit tests (v10 + v11 fingerprint + incremental upload)

### v12.0 — Adaptive Retrieval & Load-Aware Degradation ✓

- [x] Query Profiler: lightweight intent classifier (keyword + embedding similarity, L1/L2/L3)
- [x] Dynamic RRF weights: intent-driven weight matrix via YAML config (replaces static env vars)
- [x] Global load monitor: Redis sliding-window QPS counter with NORMAL/WARNING/CRITICAL states
- [x] Adaptive degradation: WARNING skips Critique/Replan, CRITICAL circuit-breaks Neo4j + Tavily
- [x] SSE events: `query_profiler` and `system_state` pushed to frontend
- [x] Prometheus metrics: `system_load_state`, `query_qps`, `query_profiler_distribution`
- [x] Locust load testing script with L1/L2/L3 query coverage
- [x] A/B evaluation script: static vs dynamic chain comparison
- [x] 37 new unit/integration tests, 118 total passing with real databases

### v13.0 — Streaming Incremental Graph Engine ✓

- [x] Incremental clustering: local patching (neighbor consensus) + subgraph re-clustering (Louvain on affected communities)
- [x] Dirty-flag summary regeneration: `is_dirty` on CommunitySummary, targeted LLM calls
- [x] Redis Streams message queue: three-stage pipeline with consumer groups
- [x] Three-stage consumer: doc_ingest → graph_extract → vector_sync
- [x] Benchmark script: full vs incremental time comparison + Token cost analysis
- [x] 10 new unit tests for incremental clustering

### v14.0 — Multi-Tenant RBAC & Data Isolation ✓

- [x] Auth package: Tenant/User/Role SQLAlchemy models, JWT handler (PyJWT + passlib bcrypt), UserContext dependency
- [x] Auth endpoints: `/auth/register` (create user + tenant), `/auth/token` (OAuth2 password grant)
- [x] MySQL tenant_id: FK on DocumentIndex, ChatSession, ParentChunk, QueryCacheStore with server_default
- [x] Milvus tenant_id: field added to collection schema; pre-filtering via `expr` in hybrid_retrieve
- [x] Neo4j tenant_id: entity MERGE key extended to `{name, tenant_id}`; Cypher subgraph constraint
- [x] Ingestion propagation: tenant_id flows through upload → arq/Redis Streams → worker → all stores
- [x] SupervisorState.user_context: tenant/role info propagated through entire agent graph
- [x] Data Analyst SQL isolation: LLM prompt constraint + execute_sql defense-in-depth check
- [x] Session scoping: list_session_infos filters by tenant_id; tenant-specific cache keys
- [x] Privilege escalation evaluation: 4 red-team test cases + evaluate_security function
- [x] 47 tests passing (12 integration, 9 auth, 9 isolation, 13 evaluation, 4 fingerprint)

### v15.0 — SaaS Metering, Rate Limiting & Audit Trail ✓

- [x] Token usage tracking: `token_usage_logs` table with per-request prompt/completion token recording
- [x] Per-tenant rate limiting: Redis sliding-window QPS limiter with `rate_limit_rules` table
- [x] Rate-limit middleware: FastAPI HTTP middleware returns 429 with Retry-After header
- [x] SLA-aware degradation: enterprise/premium/free tiers with different degradation levels under load
- [x] Audit trail: `audit_logs` table with immutable logging for MCP tool calls, SQL execution, HITL events
- [x] Audit context manager: `AuditContext` with automatic risk level classification on success/failure
- [x] Billing API: `GET /billing/usage` (token summary) + `GET /billing/audit` (paginated audit logs)
- [x] HITL webhook: POST to `HITL_WEBHOOK_URL` on interrupt events for admin notification
- [x] 40 tests passing (v14+v15 combined: 28 billing + 12 privilege escalation)

### v16.0 — Agent Workflow Platform ✓

- [x] Workflow Planner: LLM 目标拆解为 DAG 执行计划（`POST /workflows/plan`）
- [x] Workflow Executor: LangGraph DAG 引擎，串行+并行执行（`POST /workflows/execute`）
- [x] WorkflowTool 抽象: 6 Agent 统一注册为 WorkflowTool，轻量 LLM 调用
- [x] Artifact System: Report/Excel/Chart/CSV 交付物生成 + 持久化
- [x] Workflow API: plan/execute/status/artifacts/list 全链路
- [x] Frontend Panel: 任务工作流标签页，DAG 可视化，进度条，产物查看，历史记录
- [x] 23 tests passing (19 workflow + 4 audit)

### v17.0 — Adaptive GraphRAG ✓

- [x] 6-Type Query Classification: factoid/entity_relation/multi_hop/global_summary/temporal/comparison
- [x] RetrievalPlanner: query-type→RetrievalPlan 通道选择 + 图深度决策
- [x] Adaptive RRF: 6 种类型独立 weight_matrix，query_type 优先查找
- [x] GraphUtilityEstimator: 5 维启发式预测图检索价值，低分跳过 Neo4j
- [x] Orchestrator Integration: graph nodes 动态读取 intent 条件跳过检索
- [x] Evaluation: 23 条 benchmark, Overall 78.3%
- [x] 50 tests passing (8 planner + 5 utility + 13 profiler + 10 rrf + 14 other)

### v18.0 — Graph Reasoning Engine ✓

- [x] ReasoningPlanner: NL→结构化 ReasoningPlan（起始实体+最大跳数+推理策略）
- [x] SubgraphRetriever: 多跳 Cypher → NetworkX DiGraph 子图抽取
- [x] PathExplorer: BFS + Beam Search 候选推理路径发现
- [x] PathRanker: 4 维加权路径排序（语义+置信度+时序+长度）
- [x] ReasoningVerifier: LLM 答案-路径交叉验证
- [x] Multi-hop fix: graph_retriever 真正 n-hop 循环扩展
- [x] 47 tests passing (15 reasoning + 32 regression)

### v19.0 — Memory Graph System ✓

- [x] Memory Schemas: Fact/Preference/Task/Relation 四种记忆类型
- [x] MemoryGraphStore: Neo4j `:Memory` 节点 + `:MENTIONS` 关系链接 Entity
- [x] MemoryExtractor: LLM 提取对话结构化记忆（JSON）
- [x] MemoryImportance: 时间衰减 + 访问频次评分
- [x] MemoryRetriever: 用户记忆上下文注入 LLM prompt
- [x] Brain Hook: 对话保存后异步提取，Config toggle 控制
- [x] 57 tests passing (10 memory + 47 regression)

### v20.0 — Deep Research Engine ✓

- [x] Research Planner: LLM 将研究目标拆解为 DAG 执行计划（3~6 子任务，依赖关系自动分析）
- [x] Research Executor: DAG 执行引擎，串行+并行调度 4 个 Research Agent，支持断点恢复
- [x] Research Agents: Web/Graph/Data/Internal KB 四大研究代理，统一输出结构化证据
- [x] Evidence Store: 证据持久化 + 多维度查询 + 统计（来源/置信度/覆盖率）
- [x] Research Reviewer: 4 维证据评分（覆盖率 35% + 多样性 20% + 引用 25% + 置信度 20%）
- [x] Gap Analyzer: LLM 缺失分析 → 自动补充检索，Collect→Review→Gap→Collect 循环（max 3 rounds）
- [x] Report Generator: 证据驱动中文研究报告（Markdown/PDF/PPTX），每条结论绑定 Evidence ID
- [x] Research API: /research/create/status/evidence/report/cancel/list 全链路
- [x] Frontend Research Workspace: 进度实时监控 + 证据卡片查看 + 报告阅读 + 历史回溯
- [x] 性能优化: qwen-turbo + max_tokens=1024 + 精简中文提示词，Planner 74s→6.6s（11x）
- [x] UI 升级: GPT 风格侧边栏收起、统一三面板大厂审美、类型筛选历史记录、弹出模态框查看结果
- [x] 34 tests passing (16 research + 18 evidence_graph)

### v21.0 — Dynamic Research Agent ✓

- [x] Hypothesis Generator: LLM 从研究目标生成 2~4 个竞争性假设（H1/H2/H3），每个假设独立验证
- [x] Evidence Graph: Neo4j 证据图谱（`:EvidenceNode` + `:SUPPORTS`/`:REFUTES` 关系），替代 v20 平面列表
- [x] Conflict Detector: LLM 逐对比较跨假设证据，检测 factual/inferential/contextual 矛盾
- [x] Question Expander: 从证据冲突和未验证假设自动生成追问，触发 Hypothesis→Evidence→Conflict→Question 动态循环
- [x] Confidence Estimator: 多维度置信度评分（来源权威度 20% + 交叉验证 40% + 反驳惩罚 30% + 引用质量 10%）
- [x] Executor 重构: 假设驱动动态循环——假设生成→证据收集→冲突检测→追问展开→再次收集
- [x] 前端证据图谱可视化: Echarts 力导向图（绿色=高置信度，红线=证据矛盾）+ 假设卡片 + 矛盾告警
- [x] 34 tests passing (16 v20 + 18 v21)
- [x] **Workflow 合并入 Research**: 任务工作流页面移除，功能统一到研究工具（rag_specialist 真正调用 Milvus 检索）

### v7.x — Planned

- [ ] Editable session names in sidebar
- [ ] HITL state recovery on page refresh (polling endpoint)
- [ ] Conversation export (Markdown / PDF)
- [ ] Graph visualization panel in frontend (D3.js force graph)
- [ ] Entity-level citation links in answers

---

<div align="center">

**Built with LangGraph · Milvus · Neo4j · FastAPI · Vue 3**

</div>
