import pytest
from backend.memory.schemas import MemoryNode, MemoryType, MemoryExtraction
from backend.memory.importance import MemoryImportance
from backend.memory.store import MemoryGraphStore, get_memory_store


class TestMemorySchemas:
    def test_fact_memory(self):
        m = MemoryNode(memory_type=MemoryType.FACT, content="User works at Google",
                       subject="user", predicate="WORKS_AT", object_entity="Google")
        assert m.memory_type == MemoryType.FACT
        assert m.importance == 0.5

    def test_preference_memory(self):
        m = MemoryNode(memory_type=MemoryType.PREFERENCE, content="Prefers Python",
                       subject="user", predicate="PREFERS", object_entity="Python",
                       importance=0.9)
        assert m.importance == 0.9
        assert m.predicate == "PREFERS"

    def test_memory_id_auto_generated(self):
        m = MemoryNode(content="test")
        assert m.memory_id == ""

    def test_task_memory(self):
        m = MemoryNode(memory_type=MemoryType.TASK, content="Analyzed Q2 sales",
                       subject="user", predicate="WORKED_ON")
        assert m.memory_type == MemoryType.TASK

    def test_relation_memory(self):
        m = MemoryNode(memory_type=MemoryType.RELATION, content="Knows Sam Altman",
                       subject="user", predicate="KNOWS", object_entity="Sam Altman")
        assert m.object_entity == "Sam Altman"

    def test_extraction_empty(self):
        e = MemoryExtraction()
        assert e.memories == []


class TestMemoryImportance:
    def test_recent_high_score(self):
        imp = MemoryImportance(decay_days=30)
        score = imp.compute(0.9, "2026-06-04T00:00:00", access_count=1)
        assert score > 0.5

    def test_old_low_score(self):
        imp = MemoryImportance(decay_days=30)
        score = imp.compute(0.9, "2020-01-01T00:00:00", access_count=1)
        assert score < 0.5

    def test_frequent_boosts_score(self):
        imp = MemoryImportance(decay_days=30)
        s1 = imp.compute(0.9, "2026-06-01T00:00:00", access_count=1)
        s2 = imp.compute(0.9, "2026-06-01T00:00:00", access_count=10)
        assert s2 > s1


class TestMemoryStore:
    def test_save_and_retrieve(self):
        store = MemoryGraphStore()
        m = MemoryNode(memory_type=MemoryType.FACT, content="test memory",
                       subject="test", tenant_id=1, user_id=99, session_id="test")
        ok = store.save(m)
        assert ok
        assert m.memory_id != ""
        from backend.storage.graph_client import write_cypher
        write_cypher("MATCH (m:Memory {memory_id: $mid}) DETACH DELETE m",
                     {"mid": m.memory_id})
