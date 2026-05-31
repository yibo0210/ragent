# v11 Incremental Pipeline & DevOps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the document ingestion pipeline from "nuke-and-pave" to incremental delta updates, add async task processing via Redis queue, and containerize the full stack with Docker Compose.

**Architecture:** Document fingerprinting (SHA-256) detects changes at upload time. Changed documents trigger a 3-phase cascade: (1) old data cleanup by `source_filename` across Milvus/MySQL/Neo4j, (2) re-extraction and re-insertion of changed chunks only, (3) orphan graph node garbage collection. A Redis-backed async worker decouples ingestion from the HTTP request path. Docker Compose orchestrates all services including the new worker process.

**Tech Stack:** Python 3.12, FastAPI, Redis (arq task queue), Milvus 2.5, Neo4j 5.26, MySQL 8.0, Docker Compose

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/documents/fingerprint.py` | Create | SHA-256 hashing, chunk-level content hashing, delta detection |
| `backend/storage/doc_lifecycle.py` | Modify | Add `upsert_document_index`, wire `file_hash` and `chunk_count` |
| `backend/api/routes.py` | Modify | Hash check before ingestion, async task dispatch, SSE progress |
| `backend/pipeline/ingestion_worker.py` | Create | Async worker: full ingestion pipeline as a background task |
| `backend/pipeline/__init__.py` | Create | Package init |
| `backend/pipeline/task_queue.py` | Create | arq Redis queue configuration and task definitions |
| `start_worker.py` | Create | Worker process entrypoint |
| `docker-compose.yml` | Modify | Add Redis, api, worker services with depends_on and resource limits |
| `Dockerfile` | Modify | Multi-stage build, support for both api and worker entrypoints |
| `backend/storage/graph_cleanup.py` | Modify | Add `cleanup_by_filename` for incremental graph rebuild |
| `backend/milvus/writer.py` | Modify | Set `is_deleted=False` on insert for consistency |
| `tests/test_fingerprint.py` | Create | Unit tests for hashing and delta detection |
| `tests/test_incremental_upload.py` | Create | Integration tests for re-upload behavior |

---

### Task 1: Document Fingerprinting — SHA-256 Hash

**Files:**
- Create: `backend/documents/fingerprint.py`
- Create: `tests/test_fingerprint.py`

- [ ] **Step 1: Write failing tests for fingerprint module**

```python
# tests/test_fingerprint.py
import os, tempfile
from backend.documents.fingerprint import compute_file_hash, compute_chunk_hash, compute_chunks_hash


def test_compute_file_hash_deterministic():
    """Same file content produces same hash."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"hello world")
        path = f.name
    try:
        h1 = compute_file_hash(path)
        h2 = compute_file_hash(path)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest
    finally:
        os.unlink(path)


def test_compute_file_hash_different_content():
    """Different file content produces different hash."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"hello world")
        path1 = f.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"goodbye world")
        path2 = f.name
    try:
        assert compute_file_hash(path1) != compute_file_hash(path2)
    finally:
        os.unlink(path1)
        os.unlink(path2)


def test_compute_chunk_hash():
    """Chunk hash is SHA-256 of text content."""
    h = compute_chunk_hash("This is a test chunk.")
    assert len(h) == 64
    assert h == compute_chunk_hash("This is a test chunk.")
    assert h != compute_chunk_hash("Different text")


def test_compute_chunks_hash_batch():
    """Batch hash computation returns dict mapping chunk_id to hash."""
    chunks = [
        {"chunk_id": "doc::p1::l3::0", "text": "chunk A"},
        {"chunk_id": "doc::p1::l3::1", "text": "chunk B"},
    ]
    result = compute_chunks_hash(chunks)
    assert len(result) == 2
    assert "doc::p1::l3::0" in result
    assert len(result["doc::p1::l3::0"]) == 64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fingerprint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.documents.fingerprint'`

- [ ] **Step 3: Implement fingerprint module**

```python
# backend/documents/fingerprint.py
"""Document and chunk content fingerprinting for incremental updates."""
import hashlib


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_chunk_hash(text: str) -> str:
    """Compute SHA-256 hash of a single chunk's text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_chunks_hash(chunks: list[dict]) -> dict[str, str]:
    """Compute content hash for a batch of chunks.

    Args:
        chunks: list of dicts with 'chunk_id' and 'text' keys.

    Returns:
        dict mapping chunk_id -> SHA-256 hex digest of text.
    """
    return {c["chunk_id"]: compute_chunk_hash(c["text"]) for c in chunks}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fingerprint.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add backend/documents/fingerprint.py tests/test_fingerprint.py
git commit -m "feat(pipeline): add document and chunk content fingerprinting (SHA-256)"
```

---

### Task 2: Activate DocumentIndex — Wire Hash & Chunk Count

**Files:**
- Modify: `backend/storage/doc_lifecycle.py`
- Modify: `backend/api/routes.py` (upload endpoint)

- [ ] **Step 1: Add `upsert_document_index` to doc_lifecycle.py**

```python
# Add to backend/storage/doc_lifecycle.py

from datetime import datetime


def upsert_document_index(filename: str, file_hash: str, chunk_count: int) -> dict:
    """Create or update DocumentIndex entry with fingerprint and chunk count.

    Returns:
        dict with 'action' (created/updated/skipped), 'old_hash', 'new_hash'.
    """
    from backend.storage.database import SessionLocal
    from backend.storage.models import DocumentIndex

    session = SessionLocal()
    try:
        doc = session.query(DocumentIndex).filter_by(filename=filename).first()
        now = datetime.utcnow()

        if doc is None:
            doc = DocumentIndex(
                filename=filename,
                file_hash=file_hash,
                chunk_count=chunk_count,
                is_deleted=False,
                version=1,
                created_at=now,
                updated_at=now,
            )
            session.add(doc)
            session.commit()
            return {"action": "created", "old_hash": "", "new_hash": file_hash}

        old_hash = doc.file_hash or ""
        if old_hash == file_hash:
            return {"action": "skipped", "old_hash": old_hash, "new_hash": file_hash}

        doc.file_hash = file_hash
        doc.chunk_count = chunk_count
        doc.is_deleted = False
        doc.version = (doc.version or 1) + 1
        doc.updated_at = now
        session.commit()
        return {"action": "updated", "old_hash": old_hash, "new_hash": file_hash}
    finally:
        session.close()
```

- [ ] **Step 2: Integrate hash check into upload endpoint**

In `backend/api/routes.py`, after saving the file to disk and before chunking, add hash computation and skip logic:

```python
# In upload_document(), after file save, before chunking:
from backend.documents.fingerprint import compute_file_hash
from backend.storage.doc_lifecycle import upsert_document_index

file_hash = compute_file_hash(file_path)
index_result = upsert_document_index(filename, file_hash, 0)  # chunk_count updated later

if index_result["action"] == "skipped":
    yield _sse("complete", {"filename": filename, "status": "unchanged", "hash": file_hash})
    return

# ... existing chunking + ingestion pipeline ...
# After ingestion completes, update chunk_count:
upsert_document_index(filename, file_hash, total_chunk_count)
```

- [ ] **Step 3: Manual verification**

Upload the same document twice. Second upload should emit `complete` with `status: "unchanged"` and skip all processing.

- [ ] **Step 4: Commit**

```bash
git add backend/storage/doc_lifecycle.py backend/api/routes.py
git commit -m "feat(pipeline): activate DocumentIndex with file_hash fingerprinting and skip-on-unchanged"
```

---

### Task 3: Incremental Cleanup — Graph Cascade by Filename

**Files:**
- Modify: `backend/storage/graph_cleanup.py`

- [ ] **Step 1: Add `cleanup_by_filename` function**

```python
# Add to backend/storage/graph_cleanup.py

def cleanup_by_filename(filename: str) -> dict:
    """Remove all graph data originating from a specific document.

    1. Find all chunk IDs belonging to this filename (from MySQL).
    2. Strip those chunk IDs from edge source_chunks arrays.
    3. Remove edges that became empty.
    4. Remove orphan entities.

    Args:
        filename: the document filename to clean up.

    Returns:
        dict with 'stripped_edges', 'removed_edges', 'removed_entities'.
    """
    from backend.storage.doc_lifecycle import get_chunk_ids_by_filename

    chunk_ids = get_chunk_ids_by_filename(filename, include_deleted=True)
    if not chunk_ids:
        return {"stripped_edges": 0, "removed_edges": 0, "removed_entities": 0}

    return full_cascade_cleanup(chunk_ids)
```

- [ ] **Step 2: Wire into upload endpoint**

In `backend/api/routes.py`, before re-inserting graph data, clean up old graph data:

```python
# Before graph extraction:
from backend.storage.graph_cleanup import cleanup_by_filename

cleanup_stats = cleanup_by_filename(filename)
yield _sse("rag_step", {"step": "graph_cleanup", "detail": cleanup_stats})
```

- [ ] **Step 3: Commit**

```bash
git add backend/storage/graph_cleanup.py backend/api/routes.py
git commit -m "feat(pipeline): add cleanup_by_filename for incremental graph rebuild"
```

---

### Task 4: Fix Milvus is_deleted Phantom Field

**Files:**
- Modify: `backend/milvus/writer.py`

- [ ] **Step 1: Add `is_deleted=False` to Milvus insert payload**

In `backend/milvus/writer.py`, in the `write_documents` method, add `is_deleted` to each insert entity:

```python
# In the batch insert loop, add to each entity dict:
entity["is_deleted"] = False
```

- [ ] **Step 2: Verify retrieval filter works correctly**

The existing filter `is_deleted != true` in `rag/utils.py` will now correctly match inserted documents (which have `is_deleted=False`) and correctly exclude any future soft-deleted documents (which would have `is_deleted=True`).

- [ ] **Step 3: Commit**

```bash
git add backend/milvus/writer.py
git commit -m "fix(pipeline): set is_deleted=False on Milvus insert for consistent soft-delete filtering"
```

---

### Task 5: Async Task Queue — arq Worker

**Files:**
- Create: `backend/pipeline/__init__.py`
- Create: `backend/pipeline/task_queue.py`
- Create: `backend/pipeline/ingestion_worker.py`
- Create: `start_worker.py`
- Modify: `pyproject.toml`
- Modify: `backend/api/routes.py`

- [ ] **Step 1: Add arq dependency**

In `pyproject.toml`, add to `dependencies`:
```
"arq>=0.26.0",
```

Run: `uv sync`

- [ ] **Step 2: Create task queue configuration**

```python
# backend/pipeline/__init__.py
# (empty)
```

```python
# backend/pipeline/task_queue.py
"""arq task queue configuration for async document ingestion."""
import asyncio
from arq import create_pool
from arq.connections import RedisSettings

REDIS_URL = "redis://localhost:6379/1"

async def get_redis_pool():
    """Get or create the arq Redis connection pool."""
    return await create_pool(RedisSettings.from_dsn(REDIS_URL))
```

- [ ] **Step 3: Create ingestion worker**

```python
# backend/pipeline/ingestion_worker.py
"""Async document ingestion worker — runs the full pipeline outside the HTTP request."""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()


async def run_ingestion_task(ctx, filename: str, file_path: str, file_hash: str):
    """Execute the full document ingestion pipeline as a background task.

    This is the async worker function that arq dispatches. It mirrors the logic
    in the upload endpoint but runs outside the HTTP request lifecycle.
    """
    from backend.storage.database import init_db
    from backend.storage.graph_schema import init_graph_schema
    from backend.documents.loader import DocumentLoader
    from backend.documents.fingerprint import compute_chunks_hash
    from backend.documents.graph_extractor import extract_from_l2_chunks
    from backend.storage.graph_ingestion import ingest_extraction_result
    from backend.storage.graph_cleanup import cleanup_by_filename
    from backend.storage.doc_lifecycle import upsert_document_index, get_chunk_ids_by_filename
    from backend.storage.parent_chunk_store import ParentChunkStore
    from backend.milvus.writer import MilvusWriter
    from backend.storage.cache import RedisCache

    # Initialize services
    init_db()
    init_graph_schema()
    cache = RedisCache()
    parent_store = ParentChunkStore(cache)
    writer = MilvusWriter()

    # Step 1: Clean up old data
    old_chunk_ids = get_chunk_ids_by_filename(filename, include_deleted=True)
    if old_chunk_ids:
        from backend.milvus.client import MilvusClient
        milvus = MilvusClient()
        milvus.init_collection()
        milvus.delete(f'filename == "{filename}"')
        parent_store.delete_by_filename(filename)
        cleanup_by_filename(filename)

    # Step 2: Parse and chunk
    loader = DocumentLoader()
    all_chunks = loader.load_document(file_path, filename)
    parent_docs = [c for c in all_chunks if c.get("chunk_level") in (1, 2)]
    leaf_docs = [c for c in all_chunks if c.get("chunk_level") == 3]

    # Step 3: Store parent chunks
    parent_store.upsert_documents(parent_docs)

    # Step 4: Vectorize and write to Milvus
    writer.write_documents(leaf_docs)

    # Step 5: Graph extraction and ingestion
    l1_chunks = [c for c in all_chunks if c.get("chunk_level") == 1]
    if not l1_chunks:
        l1_chunks = parent_docs[:10]

    l3_ids = [c["chunk_id"] for c in leaf_docs]
    result = await extract_from_l2_chunks(l1_chunks, filename)
    ingest_extraction_result(result.entities, result.relations, l3_ids)

    # Step 6: Update document index with final chunk count
    upsert_document_index(filename, file_hash, len(leaf_docs))

    return {
        "filename": filename,
        "total_chunks": len(all_chunks),
        "leaf_chunks": len(leaf_docs),
        "entities": len(result.entities),
        "relations": len(result.relations),
    }


class WorkerSettings:
    """arq worker configuration."""
    functions = [run_ingestion_task]
    redis_settings = None  # Uses default localhost:6379

    @staticmethod
    async def startup(ctx):
        """Worker startup hook."""
        from backend.storage.database import init_db
        init_db()

    @staticmethod
    async def shutdown(ctx):
        """Worker shutdown hook."""
        pass
```

- [ ] **Step 4: Create worker entrypoint**

```python
# start_worker.py
"""Entrypoint for the async document ingestion worker."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def main():
    import uvloop
    from arq import run_worker
    from backend.pipeline.ingestion_worker import WorkerSettings

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    run_worker(WorkerSettings)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Modify upload endpoint for async dispatch**

In `backend/api/routes.py`, replace the synchronous ingestion pipeline with async task dispatch:

```python
# At the top of upload_document(), after hash check:
import arq

async def _dispatch_ingestion(filename: str, file_path: str, file_hash: str):
    """Dispatch ingestion to async worker, return task ID."""
    from backend.pipeline.task_queue import get_redis_pool
    pool = await get_redis_pool()
    job = await pool.enqueue_job("run_ingestion_task", filename, file_path, file_hash)
    return job.job_id if job else None

# In the upload handler, after hash check passes:
yield _sse("rag_step", {"step": "task_dispatched", "detail": f"Ingestion queued for {filename}"})
job_id = await _dispatch_ingestion(filename, file_path, file_hash)
yield _sse("complete", {"filename": filename, "status": "queued", "job_id": job_id})
```

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/ start_worker.py pyproject.toml backend/api/routes.py
git commit -m "feat(pipeline): add arq async task queue for document ingestion"
```

---

### Task 6: Docker Compose — Full Stack Orchestration

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`

- [ ] **Step 1: Add Redis service to docker-compose.yml**

```yaml
# Add to docker-compose.yml services:
  redis:
    container_name: ragent-redis
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - default
```

Add volume:
```yaml
redis_data:
```

- [ ] **Step 2: Add api and worker services**

```yaml
# Add to docker-compose.yml services:
  api:
    container_name: ragent-api
    build:
      context: .
      dockerfile: Dockerfile
    command: ["uv", "run", "python", "start.py"]
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      milvus-standalone:
        condition: service_healthy
      neo4j:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./data:/app/data
    deploy:
      resources:
        limits:
          memory: 2G
    networks:
      - default

  worker:
    container_name: ragent-worker
    build:
      context: .
      dockerfile: Dockerfile
    command: ["uv", "run", "python", "start_worker.py"]
    env_file:
      - .env
    depends_on:
      milvus-standalone:
        condition: service_healthy
      neo4j:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./data:/app/data
    deploy:
      resources:
        limits:
          memory: 4G
    networks:
      - default
```

- [ ] **Step 3: Update Dockerfile for multi-entrypoint support**

```dockerfile
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

COPY . .

# Default entrypoint (overridden by docker-compose command)
CMD ["uv", "run", "python", "start.py"]
```

- [ ] **Step 4: Verify docker-compose up**

```bash
docker compose up -d
docker compose ps
# All services should show "Up" status
```

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml Dockerfile
git commit -m "feat(devops): add Redis, api, and worker services to Docker Compose with resource limits"
```

---

### Task 7: Integration Test — Re-upload Incremental Behavior

**Files:**
- Create: `tests/test_incremental_upload.py`

- [ ] **Step 1: Write integration test for re-upload**

```python
# tests/test_incremental_upload.py
"""Integration tests for incremental document upload behavior.

These tests require running Neo4j and MySQL services.
"""
import pytest
import hashlib


def test_fingerprint_skip_unchanged(tmp_path):
    """Re-uploading the same file should be detected as unchanged."""
    from backend.documents.fingerprint import compute_file_hash

    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    h1 = compute_file_hash(str(test_file))
    h2 = compute_file_hash(str(test_file))
    assert h1 == h2
    assert len(h1) == 64


def test_fingerprint_detects_change(tmp_path):
    """Modifying a file should produce a different hash."""
    from backend.documents.fingerprint import compute_file_hash

    test_file = tmp_path / "test.txt"
    test_file.write_text("version 1")
    h1 = compute_file_hash(str(test_file))

    test_file.write_text("version 2")
    h2 = compute_file_hash(str(test_file))

    assert h1 != h2


def test_document_index_upsert():
    """upsert_document_index should create, skip, or update correctly."""
    from backend.storage.doc_lifecycle import upsert_document_index

    # This test requires a running MySQL database
    # First call: create
    result1 = upsert_document_index("test_hash.txt", "abc123", 10)
    assert result1["action"] in ("created", "updated")

    # Second call same hash: skip
    result2 = upsert_document_index("test_hash.txt", "abc123", 10)
    assert result2["action"] == "skipped"

    # Third call different hash: update
    result3 = upsert_document_index("test_hash.txt", "def456", 20)
    assert result3["action"] == "updated"
    assert result3["old_hash"] == "abc123"
    assert result3["new_hash"] == "def456"
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_incremental_upload.py -v`
Expected: Tests pass (fingerprint tests are standalone; doc_index test needs MySQL)

- [ ] **Step 3: Commit**

```bash
git add tests/test_incremental_upload.py
git commit -m "test(pipeline): add incremental upload integration tests"
```

---

### Task 8: End-to-End Verification

- [ ] **Step 1: Start all services**

```bash
docker compose up -d
docker compose ps  # Verify all healthy
```

- [ ] **Step 2: Upload a test document**

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@data/documents/test_graph.md" \
  --no-buffer
# Should see SSE events: rag_step (chunking), rag_step (graph), complete
```

- [ ] **Step 3: Re-upload same document (should skip)**

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@data/documents/test_graph.md" \
  --no-buffer
# Should see: complete with status=unchanged
```

- [ ] **Step 4: Modify and re-upload (should process delta)**

```bash
echo "# New content added" >> data/documents/test_graph.md
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@data/documents/test_graph.md" \
  --no-buffer
# Should see: full pipeline execution (graph_cleanup + re-extraction)
```

- [ ] **Step 5: Verify graph cleanup**

```bash
python -c "
from scripts.graph_topology_stats import collect_topology_stats
stats = collect_topology_stats()
print(f'Nodes: {stats[\"total_nodes\"]}, Orphans: {stats[\"orphan_nodes\"]}')
"
# Orphan count should be 0
```

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 7: Final commit**

```bash
git add -A
git commit -m "feat(v11): incremental pipeline + async worker + Docker Compose full stack"
```

---

## Dependency Graph

```
Task 1 (fingerprint)
  │
  ├──> Task 2 (DocumentIndex + hash check)
  │       │
  │       ├──> Task 5 (async worker + queue)
  │       │
  │       └──> Task 7 (integration tests)
  │
  ├──> Task 3 (graph cleanup by filename)
  │
  └──> Task 4 (Milvus is_deleted fix)

Task 6 (Docker Compose) ── depends on Task 5

Task 8 (E2E verification) ── depends on all above
```

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Chunk IDs change on re-upload (content shift) | Can't match old vs new chunks | Use content hash per chunk, not ID matching |
| arq worker crash mid-pipeline | Inconsistent state across stores | DocumentIndex tracks pipeline state; re-upload retries from scratch |
| Redis not available | Queue fails | Fallback to synchronous ingestion in the API handler |
| Large documents block worker | Memory pressure | Worker memory limit (4G) in Docker Compose; batch processing |
| Neo4j orphan cleanup slow on large graphs | Timeout | Run cleanup async, don't block upload response |
