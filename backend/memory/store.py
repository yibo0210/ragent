"""MemoryGraphStore: Neo4j CRUD for Memory nodes."""

from __future__ import annotations

import uuid

from backend.storage.graph_client import write_cypher, run_cypher
from backend.memory.schemas import MemoryNode, MemoryType


class MemoryGraphStore:
    """Persists Memory nodes to Neo4j."""

    def save(self, memory: MemoryNode) -> bool:
        if not memory.memory_id:
            memory.memory_id = f"mem_{uuid.uuid4().hex[:12]}"

        cypher = """
            MERGE (m:Memory {memory_id: $memory_id})
            ON CREATE SET
                m.memory_type = $memory_type,
                m.content = $content,
                m.subject = $subject,
                m.object_entity = $object_entity,
                m.predicate = $predicate,
                m.importance = $importance,
                m.session_id = $session_id,
                m.tenant_id = $tenant_id,
                m.user_id = $user_id,
                m.created_at = $created_at
            ON MATCH SET
                m.importance = CASE
                    WHEN $importance > m.importance THEN $importance
                    ELSE m.importance
                END,
                m.content = CASE
                    WHEN $content <> '' THEN $content
                    ELSE m.content
                END
        """
        write_cypher(cypher, {
            "memory_id": memory.memory_id, "memory_type": memory.memory_type.value,
            "content": memory.content, "subject": memory.subject,
            "object_entity": memory.object_entity, "predicate": memory.predicate,
            "importance": memory.importance, "session_id": memory.session_id,
            "tenant_id": memory.tenant_id, "user_id": memory.user_id,
            "created_at": memory.created_at,
        })

        if memory.object_entity:
            try:
                write_cypher("""
                    MATCH (m:Memory {memory_id: $memory_id})
                    MATCH (e:Entity {name: $entity_name})
                    MERGE (m)-[r:MENTIONS]->(e)
                    ON CREATE SET r.predicate = $predicate
                """, {"memory_id": memory.memory_id, "entity_name": memory.object_entity,
                       "predicate": memory.predicate})
            except Exception:
                pass

        return True

    def get_by_user(self, user_id: int, tenant_id: int, limit: int = 50) -> list[MemoryNode]:
        rows = run_cypher("""
            MATCH (m:Memory)
            WHERE m.tenant_id = $tenant_id AND m.user_id = $user_id
            RETURN m ORDER BY m.importance DESC LIMIT $limit
        """, {"user_id": user_id, "tenant_id": tenant_id, "limit": limit})
        return [self._row_to_node(r["m"]) for r in rows]

    def get_by_type(self, user_id: int, tenant_id: int, memory_type: MemoryType, limit: int = 20) -> list[MemoryNode]:
        rows = run_cypher("""
            MATCH (m:Memory)
            WHERE m.tenant_id = $tenant_id AND m.user_id = $user_id
              AND m.memory_type = $memory_type
            RETURN m ORDER BY m.importance DESC LIMIT $limit
        """, {"user_id": user_id, "tenant_id": tenant_id,
              "memory_type": memory_type.value, "limit": limit})
        return [self._row_to_node(r["m"]) for r in rows]

    def _row_to_node(self, m: dict) -> MemoryNode:
        return MemoryNode(
            memory_id=m.get("memory_id", ""),
            memory_type=MemoryType(m.get("memory_type", "fact")),
            content=m.get("content", ""),
            subject=m.get("subject", ""),
            object_entity=m.get("object_entity", ""),
            predicate=m.get("predicate", ""),
            importance=float(m.get("importance", 0.5)),
            session_id=m.get("session_id", ""),
            tenant_id=int(m.get("tenant_id", 0)),
            user_id=int(m.get("user_id", 0)),
            created_at=m.get("created_at", ""),
        )


_store: MemoryGraphStore | None = None


def get_memory_store() -> MemoryGraphStore:
    global _store
    if _store is None:
        _store = MemoryGraphStore()
    return _store
