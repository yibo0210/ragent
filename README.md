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
│  │            LangGraph Supervisor-Workers Orchestrator              │  │
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
│  │  │ Web Searcher│  │ Data Analyst │  │ Direct Answer     │   │    │  │
│  │  │ (Tavily)    │  │ (Text-to-SQL)│  │                   │   │    │  │
│  │  └──────┬──────┘  └───────┬──────┘  └─────────┬─────────┘   │    │  │
│  │         └──────────────────┼───────────────────┘             │    │  │
│  │                            │                                 │    │  │
│  │                     ┌──────▼───────┐                         │    │  │
│  │                     │  Synthesize  │ ← Multi-Worker Merge    │    │  │
│  │                     └──────────────┘                         │    │  │
│  └──────────────────────────┬──────────────────────────────────────┘  │
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

### Agent Routing Flow

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
 ┌────────▼────────┐  ┌────────▼────────┐  ┌─────────────────▼──┐
 │ RAG Specialist  │  │ Local Graph     │  │ Global Graph       │
 │ (Doc retrieval) │  │ Search          │  │ Search             │
 │                 │  │ (Vector→Neo4j   │  │ (Community Summary │
 │                 │  │  entity expand) │  │  matching)         │
 └────────┬────────┘  └────────┬────────┘  └────────┬───────────┘
          │                    │                     │
          └────────────────────┼─────────────────────┘
                               │
          ┌────────────────────┼─────────────────────┐
          │                    │                     │
 ┌────────▼────────┐  ┌────────▼────────┐  ┌─────────▼─────────┐
 │  Web Searcher   │  │  Data Analyst   │  │  Direct Answer    │
 │  (Tavily API)   │  │  (Text-to-SQL)  │  │  (Chat/General)   │
 └────────┬────────┘  └────────┬────────┘  └─────────┬─────────┘
          │                    │                      │
          └────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Synthesize       │ ← Multi-Worker
                    │  (Merge Answers)    │    Aggregation
                    └──────────┬──────────┘
                               │
                         ┌─────▼─────┐
                         │  Answer   │
                         └───────────┘
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
| **RAG Specialist** | Full RAG pipeline: hybrid retrieval → rerank → auto-merge → grading → rewrite → expanded retrieval |
| **Local Graph Search** | Vector search → Neo4j entity lookup → 1-hop graph expansion → merged context for multi-hop reasoning |
| **Global Graph Search** | Direct community summary matching in Milvus for panoramic/overview questions |
| **Web Searcher** | Tavily API integration for real-time web search with automatic fallback to RAG on API failure |
| **Data Analyst** | Text-to-SQL worker: discovers schema → generates read-only SQL → executes → presents insights |
| **Direct Answer** | Handles greetings, chitchat, and general knowledge queries without retrieval overhead |
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
| **Golden Dataset** | 50 hand-crafted QA pairs covering conceptual and detail queries across all system features |
| **Ragas Metrics** | Automated faithfulness, answer relevancy, and context precision scoring |
| **RRF Grid Search** | `scripts/grid_search_rrf.py` — 0.1-step traversal of normalized weight combinations, auto-selects best |
| **CI/CD Pipeline** | GitHub Actions: Docker services → DB init → pytest → import verification on every push |
| **Visualization** | `scripts/run_evaluation.py` generates radar chart + per-query-type bar chart via matplotlib |

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
<td>Ragas 0.4 · matplotlib · pytest (v4.0)</td>
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
<td><strong>Infrastructure</strong></td>
<td>Docker Compose (Milvus + etcd + MinIO + Attu + Neo4j + Jaeger + Prometheus + Grafana) · GitHub Actions CI · Dockerfile</td>
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
│   ├── agent/
│   │   ├── brain.py            # Conversation storage, SSE streaming, HITL resume
│   │   ├── orchestrator.py     # LangGraph Supervisor-Workers graph (6 agents + synthesize)
│   │   ├── tools.py            # Agent tools (weather, knowledge base, web search, graph steps)
│   │   ├── web_searcher.py     # Tavily web search integration
│   │   └── data_analyst.py     # Text-to-SQL worker (schema discovery → SQL gen → execution)
│   ├── rag/
│   │   ├── pipeline.py         # LangGraph RAG workflow (retrieve → grade → rewrite → expanded)
│   │   ├── utils.py            # Hybrid retrieval, reranking, auto-merging, query expansion, 3-ch RRF
│   │   └── graph_retriever.py  # Graph-enhanced retrieval (local search + global search)
│   ├── documents/
│   │   ├── loader.py           # Three-level hierarchical document chunking (PDF/Word/Excel/MD)
│   │   └── graph_extractor.py  # LLM entity/relation extraction from L2 chunks
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
│   │   ├── dataset.py          # Golden dataset loader
│   │   └── metrics.py          # Ragas metrics (faithfulness, relevancy, precision)
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
│   └── schemas.py              # Pydantic request/response models + GraphEntity/GraphRelation
│
├── scripts/
│   ├── run_community_clustering.py  # Offline: build graph → cluster → summarize → index
│   ├── run_entity_resolution.py    # Offline: entity dedup pipeline (v4.0)
│   ├── run_evaluation.py           # Automated RAG eval + charts (v4.0)
│   └── grid_search_rrf.py          # RRF weight optimization (v4.0)
│
├── frontend/
│   ├── index.html              # Vue 3 SPA (chat, trace canvas, HITL modal, settings)
│   ├── script.js               # Application logic, SSE handler, API integration
│   └── style.css               # Gemini-inspired dual-theme (Light/Dark)
│
├── tests/
│   ├── test_doc_lifecycle.py   # Soft-delete unit tests (v4.0)
│   └── golden_dataset.json     # 50-item evaluation dataset (v4.0)
│
├── data/
│   └── documents/              # Uploaded document storage
│
├── docs/
│   ├── planning/                # Feature specification documents
│   │   ├── 5.23todov2.md        # v2.0 — Multi-Agent + HITL specification
│   │   ├── 5.24todov3.md        # v3.0 — GraphRAG requirements specification
│   │   ├── 5.25todolistv4.md    # v4.0 — UI optimization & planning
│   │   └── GraphRAG-v3.0-升级计划.md  # v3.0 — Implementation plan (5 phases)
│   └── img.png                  # Application screenshot
│
├── docker-compose.yml          # Full stack (Milvus + Neo4j + Jaeger + Prometheus + Grafana)
├── docker-compose.ci.yml       # CI environment services (v4.0)
├── Dockerfile                  # Application container image (v4.0)
├── prometheus.yml              # Prometheus scrape configuration (v5.0)
├── .github/workflows/ci.yml    # GitHub Actions CI pipeline (v4.0)
├── pyproject.toml              # Python dependencies & project metadata
├── start.py                    # UTF-8 startup script (uvicorn wrapper)
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
- [x] Golden evaluation dataset (50 QA pairs across 2 query types)
- [x] Ragas automated evaluation pipeline (faithfulness, answer relevancy, context precision)
- [x] RRF three-channel weight grid search (0.1 step, auto-optimize)
- [x] RRF weights configurable via environment variables
- [x] GitHub Actions CI/CD pipeline
- [x] Dockerfile for application containerization
- [x] Evaluation visualization (radar chart + per-type bar chart)
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

### v5.x — Planned

- [ ] Editable session names in sidebar
- [ ] HITL state recovery on page refresh (polling endpoint)
- [ ] Conversation export (Markdown / PDF)
- [ ] Graph visualization panel in frontend (D3.js force graph)
- [ ] Entity-level citation links in answers
- [ ] Scheduled auto-run of community clustering after batch uploads
- [ ] Graph extraction progress callback during upload

---

<div align="center">

**Built with LangGraph · Milvus · Neo4j · FastAPI · Vue 3**

</div>
