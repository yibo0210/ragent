"""社区聚类与摘要生成。"""
import networkx as nx
import community as community_louvain
from backend.storage.graph_client import run_cypher, write_cypher
from backend.milvus.client import MilvusManager
from backend.embedding.service import EmbeddingService
from backend.observability import get_logger

logger = get_logger("graph.community")


def build_graph_from_neo4j() -> nx.Graph:
    """从 Neo4j 拉取全量图。"""
    rows = run_cypher(
        "MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) "
        "RETURN a.name AS source, b.name AS target, r.weight AS weight"
    )
    G = nx.Graph()
    for row in rows:
        weight = row.get("weight", 0.5) if row.get("weight") else 0.5
        G.add_edge(row["source"], row["target"], weight=float(weight))

    isolated = run_cypher(
        "MATCH (e:Entity) WHERE NOT (e)-[:RELATES_TO]-() RETURN e.name AS name"
    )
    for row in isolated:
        G.add_node(row["name"])
    return G


def run_leiden_clustering(G: nx.Graph, resolution: float = 1.0) -> dict[str, int]:
    """运行 Louvain 聚类，返回 {node_name: community_id}。"""
    partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)
    return partition


def write_community_ids(partition: dict[str, int]):
    """将社区 ID 写回 Neo4j 实体节点。"""
    batch = [{"name": name, "cid": cid} for name, cid in partition.items()]

    for i in range(0, len(batch), 100):
        chunk = batch[i:i + 100]
        write_cypher(
            "UNWIND $batch AS item "
            "MATCH (e:Entity {name: item.name}) "
            "SET e.community_id = toString(item.cid)",
            {"batch": chunk},
        )
    print(f"[CLUSTER] Written community IDs for {len(batch)} entities")


def generate_community_summary(community_id: str) -> str:
    """为一个社区生成综述摘要。"""
    from backend.agent.orchestrator import _get_worker_model
    from langchain_core.messages import HumanMessage

    rows = run_cypher(
        "MATCH (e:Entity {community_id: $cid}) "
        "OPTIONAL MATCH (e)-[r:RELATES_TO]-(other:Entity) "
        "RETURN e.name AS entity, e.type AS type, e.description AS desc",
        {"cid": community_id},
    )

    context_parts = []
    for row in rows:
        context_parts.append(
            f"- [{row.get('type', 'Concept')}] {row['entity']}: {row.get('desc', '')}"
        )
    context = "\n".join(context_parts)

    if not context.strip():
        return "该社区暂无实体数据。"

    prompt = (
        "你是知识分析师。以下是知识图谱中一个社区的实体列表。"
        "请写一份简短的综述报告（200-400 字），总结该社区涵盖的主要主题、"
        "关键实体及其关联。\n\n"
        f"社区实体:\n{context}\n\n综述报告:"
    )

    model = _get_worker_model()
    response = model.invoke([HumanMessage(content=prompt)])
    summary_text = response.content

    # Persist to MySQL
    from backend.storage.database import SessionLocal
    from backend.storage.models import CommunitySummary

    entity_count = len(rows)
    db = SessionLocal()
    try:
        existing = db.query(CommunitySummary).filter_by(community_id=community_id).first()
        if existing:
            existing.summary_text = summary_text
            existing.entity_count = entity_count
            existing.is_dirty = False
        else:
            db.add(CommunitySummary(
                community_id=community_id,
                summary_text=summary_text,
                entity_count=entity_count,
                is_dirty=False,
            ))
        db.commit()
    finally:
        db.close()

    return summary_text


def generate_all_summaries(partition: dict[str, int]) -> list[dict]:
    """为所有社区生成摘要，返回 [{community_id, summary_text}]。"""
    unique_cids = sorted(set(partition.values()))
    summaries = []
    for i, cid in enumerate(unique_cids):
        print(f"[SUMMARY] Generating for community {cid} ({i+1}/{len(unique_cids)})")
        text = generate_community_summary(str(cid))
        summaries.append({"community_id": str(cid), "summary_text": text})
    return summaries


def index_summaries_to_milvus(summaries: list[dict]):
    """将社区摘要向量化并写入 Milvus。"""
    embed_service = EmbeddingService()
    milvus = MilvusManager()
    milvus.init_collection()

    texts = [s["summary_text"] for s in summaries]
    embeddings = embed_service.get_embeddings(texts)

    insert_data = []
    for s, emb in zip(summaries, embeddings):
        insert_data.append({
            "dense_embedding": emb,
            "sparse_embedding": {},
            "text": s["summary_text"],
            "filename": f"community_{s['community_id']}",
            "file_type": "CommunitySummary",
            "file_path": "",
            "page_number": 0,
            "chunk_idx": 0,
            "chunk_id": f"community_{s['community_id']}",
            "parent_chunk_id": "",
            "root_chunk_id": "",
            "chunk_level": 0,
        })

    milvus.insert(insert_data)
    print(f"[SUMMARY] Indexed {len(summaries)} summaries to Milvus")


def mark_communities_dirty(community_ids: list[str]) -> int:
    """Mark communities as needing summary regeneration. Returns count updated."""
    if not community_ids:
        return 0
    from backend.storage.database import SessionLocal
    from backend.storage.models import CommunitySummary

    db = SessionLocal()
    try:
        count = 0
        for cid in community_ids:
            existing = db.query(CommunitySummary).filter_by(community_id=cid).first()
            if existing:
                existing.is_dirty = True
                count += 1
            else:
                db.add(CommunitySummary(
                    community_id=cid,
                    summary_text="",
                    entity_count=0,
                    is_dirty=True,
                ))
                count += 1
        db.commit()
        return count
    finally:
        db.close()


def update_dirty_summaries() -> int:
    """Regenerate summaries only for dirty communities. Returns count updated."""
    from backend.storage.database import SessionLocal
    from backend.storage.models import CommunitySummary

    db = SessionLocal()
    try:
        dirty = db.query(CommunitySummary).filter_by(is_dirty=True).all()
        if not dirty:
            return 0

        updated = 0
        for cs in dirty:
            try:
                generate_community_summary(cs.community_id)
                updated += 1
            except Exception as e:
                logger.warning("summary_update_failed", community_id=cs.community_id, error=str(e))

        return updated
    finally:
        db.close()


def get_community_count() -> dict:
    """Get counts of total, dirty, and clean communities."""
    from backend.storage.database import SessionLocal
    from backend.storage.models import CommunitySummary

    db = SessionLocal()
    try:
        total = db.query(CommunitySummary).count()
        dirty = db.query(CommunitySummary).filter_by(is_dirty=True).count()
        return {"total": total, "dirty": dirty, "clean": total - dirty}
    finally:
        db.close()
