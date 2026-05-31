"""Tests for document and chunk content fingerprinting."""
import os
import tempfile

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


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
