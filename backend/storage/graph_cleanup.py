"""Neo4j 图清理：移除失效 chunk 引用、回收孤立节点。"""
from backend.storage.graph_client import run_cypher, write_cypher


def strip_chunk_from_edges(chunk_ids: list[str]) -> dict:
    """从 RELATES_TO 边的 source_chunks 中移除指定 chunk ID。"""
    cypher = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE any(cid IN r.source_chunks WHERE cid IN $chunk_ids)
    SET r.source_chunks = [cid IN r.source_chunks WHERE NOT cid IN $chunk_ids]
    RETURN count(r) AS updated_edges
    """
    records = run_cypher(cypher, {"chunk_ids": chunk_ids})
    return {"updated_edges": records[0]["updated_edges"] if records else 0}


def remove_empty_edges() -> int:
    """删除 source_chunks 为空的关系边。"""
    cypher = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE size(r.source_chunks) = 0
    DELETE r
    RETURN count(r) AS deleted
    """
    records = write_cypher(cypher)
    return records.get("relationships_created", 0)


def remove_orphan_entities() -> int:
    """删除没有任何关系的孤立实体节点。"""
    cypher = """
    MATCH (e:Entity)
    WHERE NOT (e)--()
    DELETE e
    RETURN count(e) AS deleted
    """
    records = write_cypher(cypher)
    return records.get("nodes_created", 0)


def full_cascade_cleanup(chunk_ids: list[str]) -> dict:
    """完整级联清理流程。"""
    strip_result = strip_chunk_from_edges(chunk_ids)
    empty_deleted = remove_empty_edges()
    orphan_deleted = remove_orphan_entities()
    return {
        "edges_updated": strip_result["updated_edges"],
        "empty_edges_deleted": empty_deleted,
        "orphan_nodes_deleted": orphan_deleted,
    }
