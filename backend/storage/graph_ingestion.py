"""将抽取结果写入 Neo4j。"""
from .graph_client import write_cypher


def ingest_extraction_result(
    entities: list, relations: list, l3_chunk_ids: list[str]
) -> dict:
    """批量写入实体和关系到 Neo4j。"""
    stats = {"entities": 0, "relations": 0}

    for entity in entities:
        try:
            vf = getattr(entity, "valid_from", "") or ""
            vt = getattr(entity, "valid_to", "") or ""
            write_cypher(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = $type, e.description = $desc,
                    e.valid_from = $valid_from, e.valid_to = $valid_to
                ON MATCH SET e.type = $type,
                    e.description = CASE WHEN $desc <> '' THEN $desc ELSE e.description END,
                    e.valid_from = CASE WHEN $valid_from <> '' THEN $valid_from ELSE e.valid_from END,
                    e.valid_to = CASE WHEN $valid_to <> '' THEN $valid_to ELSE e.valid_to END
                """,
                {"name": entity.name, "type": entity.type, "desc": entity.description,
                 "valid_from": vf, "valid_to": vt},
            )
            stats["entities"] += 1
        except Exception as e:
            print(f"[INGEST] Entity error: {e}")

    for rel in relations:
        try:
            # 先确保目标实体存在（如果不存在则创建占位）
            write_cypher(
                "MERGE (e:Entity {name: $name}) "
                "ON CREATE SET e.type = 'Concept', e.description = '' ",
                {"name": rel.object},
            )
            rvf = getattr(rel, "valid_from", "") or ""
            rvt = getattr(rel, "valid_to", "") or ""
            write_cypher(
                """
                MATCH (a:Entity {name: $subject})
                MATCH (b:Entity {name: $object})
                MERGE (a)-[r:RELATES_TO {predicate: $predicate}]->(b)
                ON CREATE SET r.description = $desc, r.weight = $weight,
                    r.source_chunks = $chunks,
                    r.valid_from = $valid_from, r.valid_to = $valid_to
                ON MATCH SET r.weight = CASE WHEN $weight > r.weight THEN $weight ELSE r.weight END,
                    r.valid_from = CASE WHEN $valid_from <> '' THEN $valid_from ELSE r.valid_from END,
                    r.valid_to = CASE WHEN $valid_to <> '' THEN $valid_to ELSE r.valid_to END,
                    r.source_chunks = CASE
                        WHEN $chunk_id IN r.source_chunks THEN r.source_chunks
                        ELSE r.source_chunks + $chunk_id
                    END
                """,
                {
                    "subject": rel.subject,
                    "object": rel.object,
                    "predicate": rel.predicate,
                    "desc": rel.description,
                    "weight": rel.weight,
                    "chunks": l3_chunk_ids,
                    "valid_from": rvf,
                    "valid_to": rvt,
                    "chunk_id": l3_chunk_ids[0] if l3_chunk_ids else "",
                },
            )
            stats["relations"] += 1
        except Exception as e:
            print(f"[INGEST] Relation error: {e}")

    print(f"[INGEST] {stats}")
    return stats
