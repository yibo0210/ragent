"""Integration tests for incremental document upload behavior.

Fingerprint and hash-skip logic tests are standalone.
DocumentIndex tests require a running MySQL database.
"""
import os
import tempfile

import pytest

from backend.documents.fingerprint import compute_file_hash, compute_chunk_hash


class TestFingerprint:
    """Standalone fingerprint tests (no external services)."""

    def test_same_file_same_hash(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world")
            path = f.name
        try:
            assert compute_file_hash(path) == compute_file_hash(path)
        finally:
            os.unlink(path)

    def test_different_file_different_hash(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"version 1")
            p1 = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"version 2")
            p2 = f.name
        try:
            assert compute_file_hash(p1) != compute_file_hash(p2)
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_hash_is_sha256(self):
        h = compute_chunk_hash("test")
        assert len(h) == 64
        assert h == compute_chunk_hash("test")

    def test_hash_deterministic(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as f:
            f.write(b"# Title\nSome content here.")
            path = f.name
        try:
            h1 = compute_file_hash(path)
            h2 = compute_file_hash(path)
            assert h1 == h2
            assert len(h1) == 64
        finally:
            os.unlink(path)


class TestDocumentIndex:
    """Tests for DocumentIndex upsert (requires MySQL)."""

    def test_upsert_creates_new(self):
        import uuid
        from backend.storage.doc_lifecycle import upsert_document_index
        unique_name = f"test_new_{uuid.uuid4().hex[:8]}.txt"
        result = upsert_document_index(unique_name, "abc123hash", 10)
        assert result["action"] in ("created", "updated")
        assert result["new_hash"] == "abc123hash"

    def test_upsert_skips_unchanged(self):
        import uuid
        from backend.storage.doc_lifecycle import upsert_document_index
        unique_name = f"test_skip_{uuid.uuid4().hex[:8]}.txt"
        # First call to create
        upsert_document_index(unique_name, "hash_unchanged", 5)
        # Second call same hash: should skip
        result = upsert_document_index(unique_name, "hash_unchanged", 5)
        assert result["action"] == "skipped"

    def test_upsert_updates_changed(self):
        import uuid
        from backend.storage.doc_lifecycle import upsert_document_index
        unique_name = f"test_update_{uuid.uuid4().hex[:8]}.txt"
        # Create
        upsert_document_index(unique_name, "old_hash", 5)
        # Update with new hash
        result = upsert_document_index(unique_name, "new_hash", 10)
        assert result["action"] == "updated"
        assert result["old_hash"] == "old_hash"
        assert result["new_hash"] == "new_hash"


class TestGraphCleanupByFilename:
    """Tests for cleanup_by_filename (requires Neo4j)."""

    def test_cleanup_no_data(self):
        from backend.storage.graph_cleanup import cleanup_by_filename
        result = cleanup_by_filename("nonexistent_file_xyz.txt")
        assert result["edges_updated"] == 0
        assert result["empty_edges_deleted"] == 0
        assert result["orphan_nodes_deleted"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
