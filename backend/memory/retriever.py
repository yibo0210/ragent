"""MemoryRetriever: queries user memory graph and formats as context."""

from __future__ import annotations

from backend.memory.store import get_memory_store
from backend.memory.schemas import MemoryType


class MemoryRetriever:
    """Retrieves relevant user memories for injection into LLM context."""

    def retrieve(self, user_id: int, tenant_id: int, query: str = "", limit: int = 10) -> str:
        store = get_memory_store()
        memories = store.get_by_user(user_id, tenant_id, limit=limit)
        if not memories:
            return ""
        lines = ["## 用户记忆"]
        for m in memories:
            lines.append(f"- [{m.memory_type.value}] {m.content} (importance: {m.importance:.2f})")
        return "\n".join(lines)


_retriever: MemoryRetriever | None = None


def get_memory_retriever() -> MemoryRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MemoryRetriever()
    return _retriever
