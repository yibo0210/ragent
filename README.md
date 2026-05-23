<div align="center">

# Ragent AI

**Enterprise-Grade RAG Knowledge Base Assistant**

A full-stack Retrieval-Augmented Generation platform built on LangChain + LangGraph, featuring hybrid vector retrieval, multi-stage query expansion, and real-time streaming responses.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)
![Vue.js](https://img.shields.io/badge/Vue.js-3-4FC08D?style=flat&logo=vuedotjs&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Milvus](https://img.shields.io/badge/Milvus-2.5-00A1E0?style=flat&logo=milvus&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

<br/>

<img src="docs/screenshot.png" width="90%" alt="Ragent AI Interface" />

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

Ragent AI is a production-ready RAG (Retrieval-Augmented Generation) platform that enables intelligent question answering over private document collections. It combines a sophisticated multi-stage retrieval pipeline with a clean, responsive chat interface — delivering accurate, source-attributed answers in real time.

**Core Capabilities:**

- **Multi-Stage RAG Pipeline** — LangGraph-orchestrated workflow: initial retrieval → relevance grading → intelligent query rewriting → expanded retrieval → answer generation
- **Hybrid Vector Retrieval** — Dense (Qwen text-embedding-v1, 1536-dim) + Sparse (BM25) dual-channel search, fused via Reciprocal Rank Fusion (RRF)
- **Three-Level Hierarchical Chunking** — L1/L2/L3 sliding-window chunking with auto-merging retriever for coherent context assembly
- **Real-Time Streaming** — SSE-based token streaming with interleaved RAG step visualization (Searching → Grading → Rewriting)
- **Transparent RAG Trace** — Every response includes a full audit trail: retrieval strategy, rerank scores, merge decisions, and source chunks

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (Vue 3 SPA)                        │
│   Chat UI · Session Management · Knowledge Base · RAG Trace Viewer │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ SSE / HTTP
┌──────────────────────────────▼──────────────────────────────────────┐
│                      FastAPI Application Layer                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐ │
│  │  api.py      │  │  agent.py    │  │  schemas.py                │ │
│  │  REST Routes │  │  LangChain   │  │  Pydantic Models           │ │
│  └──────┬──────┘  │  Agent Core  │  └────────────────────────────┘ │
│         │         └──────┬───────┘                                  │
│         │                │                                          │
│  ┌──────▼────────────────▼──────────────────────────────────────┐  │
│  │                    RAG Pipeline (LangGraph)                   │  │
│  │  retrieve_initial → grade_documents → rewrite → retrieve_exp │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐  │
│  │                    RAG Utilities Layer                        │  │
│  │  Hybrid Retrieval · Reranking · Auto-Merging · Query Rewrite │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼───────┐  ┌───────────▼──────────┐  ┌───────▼───────┐
│    Milvus     │  │       MySQL          │  │     Redis     │
│  Vector DB    │  │  Sessions & Chunks   │  │   Hot Cache   │
│  HNSW + Sparse│  │  Persistent Storage  │  │  TTL-based    │
└───────────────┘  └──────────────────────┘  └───────────────┘
```

### RAG Pipeline Flow

```
                        ┌──────────────┐
                        │  User Query  │
                        └──────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │  Initial Retrieval  │  Hybrid (Dense + Sparse)
                    │  + Rerank + Merge   │  → RRF Fusion → Rerank → Auto-Merge
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
              ┌─────│  Grade Documents    │─────┐
              │     │  (Relevance Check)  │     │
              │     └─────────────────────┘     │
              │ Relevant                   Not Relevant
              │                                 │
     ┌────────▼────────┐            ┌───────────▼───────────┐
     │ Generate Answer │            │  Rewrite Question     │
     │  (with context) │            │  ┌─────────────────┐  │
     └─────────────────┘            │  │ Router:          │  │
                                    │  │ • step_back      │  │
                                    │  │ • hyde           │  │
                                    │  │ • complex        │  │
                                    │  └─────────────────┘  │
                                    └───────────┬───────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │  Expanded Retrieval   │
                                    │  + Rerank + Merge     │
                                    └───────────────────────┘
```

---

## Key Features

### Retrieval Engine

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | Dense embeddings (Qwen text-embedding-v1) + BM25 sparse vectors, fused via RRF in Milvus |
| **Reranking** | Post-retrieval relevance scoring via DashScope gte-rerank API with graceful degradation |
| **Auto-Merging** | L3→L2→L1 hierarchical merging — when multiple sibling leaf chunks are retrieved, they collapse into the parent chunk for coherent context |
| **Three-Level Chunking** | L1 (1200 chars) → L2 (600 chars) → L3 (300 chars) with parent-child relationship tracking |
| **Leaf-Only Indexing** | Only leaf chunks (L3) are vectorized in Milvus; parent chunks stored in MySQL to reduce index size |

### Query Intelligence

| Feature | Description |
|---------|-------------|
| **Step-Back Prompting** | For specific/detail questions — generates a higher-level question to broaden retrieval scope |
| **HyDE** | For vague/conceptual questions — generates a hypothetical document for semantic retrieval |
| **Complex Expansion** | Multi-step questions — combines both strategies with deduplication |
| **Relevance Grading** | LLM-based structured output grades retrieved documents; triggers rewrite if irrelevant |

### Application Layer

| Feature | Description |
|---------|-------------|
| **Streaming Responses** | SSE-based token streaming with real-time RAG step visualization |
| **Session Management** | Multi-turn conversations persisted in MySQL with Redis caching |
| **Conversation Summarization** | Auto-summarizes history beyond 50 turns to manage token budgets |
| **RAG Trace** | Every response includes full retrieval audit: strategy, scores, merge decisions, source chunks |
| **Answer Abort** | Frontend AbortController + backend StreamingResponse for mid-generation cancellation |
| **Weather Tool** | Amap (Gaode) API integration for real-time weather queries |

---

## Tech Stack

<table>
<tr>
<td><strong>Backend</strong></td>
<td>FastAPI · Uvicorn · LangChain · LangGraph · Pydantic</td>
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
<td><strong>Embedding</strong></td>
<td>Qwen text-embedding-v1 (1536-dim) · BM25 (custom impl.)</td>
</tr>
<tr>
<td><strong>LLM</strong></td>
<td>Qwen-Plus via DashScope (OpenAI-compatible API)</td>
</tr>
<tr>
<td><strong>Database</strong></td>
<td>MySQL (persistent storage) · Redis (hot cache layer)</td>
</tr>
<tr>
<td><strong>Infrastructure</strong></td>
<td>Docker Compose (Milvus + etcd + MinIO + Attu)</td>
</tr>
</table>

---

## Project Structure

```
Ragent-AI/
├── backend/
│   ├── app.py                 # FastAPI application factory & middleware
│   ├── api.py                 # REST API routes (chat, sessions, documents)
│   ├── agent.py               # LangChain agent core & conversation storage
│   ├── rag_pipeline.py        # LangGraph multi-stage RAG workflow
│   ├── rag_utils.py           # Retrieval utilities (hybrid, rerank, merge, rewrite)
│   ├── document_loader.py     # Three-level hierarchical document chunking
│   ├── embedding.py           # Dense (Qwen API) + Sparse (BM25) embedding service
│   ├── milvus_client.py       # Milvus vector DB client (hybrid search, RRF)
│   ├── milvus_writer.py       # Batch vectorization & Milvus ingestion
│   ├── parent_chunk_store.py  # Parent chunk storage (MySQL + Redis)
│   ├── tools.py               # Agent tools (weather, knowledge base search)
│   ├── models.py              # SQLAlchemy ORM models
│   ├── schemas.py             # Pydantic request/response schemas
│   ├── database.py            # MySQL connection & session factory
│   └── cache.py               # Redis cache utility (TTL-based)
│
├── frontend/
│   ├── index.html             # Vue 3 SPA (single-file component)
│   ├── script.js              # Application logic & API integration
│   ├── style.css              # Doubao-inspired responsive theme
│   └── logo.svg               # Brand logo
│
├── data/
│   └── documents/             # Uploaded document storage
│
├── testapiall/                # Integration test scripts
│   ├── testapi.py             # Chat API tests
│   ├── testrag.py             # RAG pipeline tests
│   ├── testmilvus.py          # Milvus CRUD tests
│   ├── testamap.py            # Weather API tests
│   └── testrerankapi.py       # Rerank API tests
│
├── docker-compose.yml         # Milvus stack (etcd + MinIO + Milvus + Attu)
├── pyproject.toml             # Python dependencies & project metadata
├── start.py                   # UTF-8 startup script (uvicorn wrapper)
└── .env                       # Environment configuration
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

Create `.env` in the project root:

```env
# ===== LLM =====
ARK_API_KEY=your_dashscope_api_key
MODEL=qwen-plus
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDER=text-embedding-v1
GRADE_MODEL=qwen-plus
MAX_TOKENS=8192

# ===== Database =====
DATABASE_URL=mysql+pymysql://root:password@localhost:3306/agent_chat
REDIS_URL=redis://localhost:6379/0

# ===== Milvus =====
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_VECTOR_DIM=1536
MILVUS_SEARCH_TOP_K=20

# ===== Rerank (optional) =====
RERANK_MODEL=gte-rerank
RERANK_BINDING_HOST=https://dashscope.aliyuncs.com/compatible-mode/v1
RERANK_API_KEY=your_dashscope_api_key
RERANK_TOP_K=10

# ===== Tools (optional) =====
AMAP_WEATHER_API=https://restapi.amap.com/v3/weather/weatherInfo
AMAP_API_KEY=your_amap_key
```

### 3. Start Infrastructure

```bash
# Start Milvus stack (etcd + MinIO + Milvus + Attu)
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

### 4. Create Database

```sql
CREATE DATABASE IF NOT EXISTS agent_chat CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 5. Launch Application

```bash
# Option A: uv
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Option B: python
python start.py
```

Open in browser:
- **Frontend**: http://127.0.0.1:8000/
- **API Docs**: http://127.0.0.1:8000/docs

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARK_API_KEY` | — | DashScope API key |
| `MODEL` | `qwen-plus` | Chat model name |
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
| `RERANK_MODEL` | `gte-rerank` | Rerank model name |
| `RERANK_TOP_K` | `10` | Rerank output candidates |

### Chunking Parameters

| Level | Chunk Size | Overlap | Purpose |
|-------|-----------|---------|---------|
| L1 (Root) | 1200 chars | 240 chars | Topical context unit |
| L2 (Mid) | 600 chars | 120 chars | Intermediate grouping |
| L3 (Leaf) | 300 chars | 60 chars | Vectorized retrieval unit |

---

## API Reference

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Synchronous chat (returns full response) |
| `POST` | `/chat/stream` | SSE streaming chat (token-by-token) |

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/sessions` | List all sessions |
| `GET` | `/sessions/{id}` | Get session messages |
| `DELETE` | `/sessions/{id}` | Delete a session |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/documents` | List all documents with chunk counts |
| `POST` | `/documents/upload` | Upload & vectorize a document |
| `DELETE` | `/documents/{filename}` | Delete document & its vectors |

> Full interactive API documentation is available at `/docs` (Swagger UI) when the application is running.

---

## Roadmap

### RAG Pipeline

- [ ] Document structure-aware chunking (heading-based pre-split + semantic refinement)
- [ ] Special handling for code blocks, tables, and images
- [ ] BM25 parameter tuning (k1/b grid search)
- [ ] RRF weight optimization via A/B testing
- [ ] Sub-question decomposition (CoT-based multi-hop reasoning)
- [ ] Multi-document Refine & conflict detection
- [ ] RAG evaluation framework (faithfulness, relevance, recall)

### Platform Capabilities

- [ ] SQL Assistant skill
- [ ] Web search integration
- [ ] Multi-agent architecture (tool delegation to specialized agents)
- [ ] Editable session names
- [ ] Dead-loop detection & recovery
- [ ] Memory optimization (MemO / LangMem integration)
- [ ] Multimodal embedding support

---

<div align="center">

**Built with LangChain · LangGraph · Milvus · FastAPI · Vue 3**

</div>
