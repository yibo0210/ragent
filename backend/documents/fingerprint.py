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
