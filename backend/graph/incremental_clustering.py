"""增量图聚类引擎

替代全量 Louvain 聚类，仅对受影响的局部子图进行重新计算。
策略 A: 局部补丁 — 新节点直接归入邻居多数社区（零算法开销）
策略 B: 子图重构 — 桥接多社区时仅对局部子图运行 Louvain
"""
from collections import Counter
from typing import Optional

import networkx as nx
import community as community_louvain

from backend.storage.graph_client import run_cypher, write_cypher
from backend.observability import get_logger

log = get_logger("graph.incremental")

# 阈值：邻居中某社区占比超过此值时直接归入
PATCH_THRESHOLD = 0.6


def get_neighbor_communities(node_name: str) -> dict:
    """获取节点的 1-hop 邻居及其 community_id。

    Returns:
        {neighbor_name: community_id} — community_id 可能为 None
    """
    cypher = """
    MATCH (e:Entity)-[:RELATES_TO]-(neighbor:Entity)
    WHERE e.name = $name
    RETURN neighbor.name AS name, neighbor.community_id AS cid
    """
    rows = run_cypher(cypher, {"name": node_name})
    return {r["name"]: r.get("cid") for r in rows}


def patch_new_node(node_name: str) -> dict:
    """尝试将新节点局部补丁到邻居多数社区。

    Returns:
        {"action": "patched", "community_id": str} — 成功归入
        {"action": "no_neighbors"} — 孤立节点，无法归入
        {"action": "recluster", "affected_communities": list[str]} — 需要子图重构
    """
    neighbors = get_neighbor_communities(node_name)

    if not neighbors:
        return {"action": "no_neighbors"}

    # 统计邻居社区分布（忽略 None）
    cid_counts = Counter(cid for cid in neighbors.values() if cid is not None)

    if not cid_counts:
        # 所有邻居都没有 community_id
        return {"action": "recluster", "affected_communities": []}

    total = sum(cid_counts.values())
    top_cid, top_count = cid_counts.most_common(1)[0]
    ratio = top_count / total

    if ratio >= PATCH_THRESHOLD:
        # 策略 A: 直接归入
        write_cypher(
            "MATCH (e:Entity {name: $name}) SET e.community_id = $cid",
            {"name": node_name, "cid": top_cid},
        )
        log.info("node_patched", node=node_name, community=top_cid, ratio=round(ratio, 2))
        return {"action": "patched", "community_id": top_cid}
    else:
        # 策略 B: 需要子图重构
        affected = list(cid_counts.keys())
        log.info("node_needs_recluster", node=node_name, affected_communities=affected)
        return {"action": "recluster", "affected_communities": affected}


def get_community_subgraph(community_ids: list[str]) -> tuple[nx.Graph, dict]:
    """从 Neo4j 提取指定社区的局部子图。

    Returns:
        (G, node_community_map) — NetworkX 无向图 + 节点→社区映射
    """
    if not community_ids:
        return nx.Graph(), {}

    cypher = """
    MATCH (e:Entity)-[r:RELATES_TO]-(other:Entity)
    WHERE e.community_id IN $cids AND other.community_id IN $cids
    RETURN e.name AS src, e.community_id AS src_cid,
           other.name AS dst, other.community_id AS dst_cid,
           r.weight AS weight
    """
    rows = run_cypher(cypher, {"cids": community_ids})

    G = nx.Graph()
    node_cid = {}
    for r in rows:
        src, dst = r["src"], r["dst"]
        weight = r.get("weight", 0.5)
        G.add_edge(src, dst, weight=weight)
        node_cid[src] = r.get("src_cid")
        node_cid[dst] = r.get("dst_cid")

    # 添加孤立节点（属于这些社区但没有边）
    iso_cypher = """
    MATCH (e:Entity)
    WHERE e.community_id IN $cids AND NOT (e)-[:RELATES_TO]-()
    RETURN e.name AS name, e.community_id AS cid
    """
    iso_rows = run_cypher(iso_cypher, {"cids": community_ids})
    for r in iso_rows:
        G.add_node(r["name"])
        node_cid[r["name"]] = r.get("cid")

    return G, node_cid


def recluster_subgraph(affected_communities: list[str]) -> dict:
    """对受影响社区的局部子图重新运行 Louvain。

    Returns:
        {"changed_nodes": int, "old_communities": list, "new_communities": list}
    """
    if not affected_communities:
        return {"changed_nodes": 0, "old_communities": [], "new_communities": []}

    G, old_cid_map = get_community_subgraph(affected_communities)

    if G.number_of_nodes() < 2:
        return {"changed_nodes": 0, "old_communities": affected_communities, "new_communities": []}

    # 运行 Louvain
    partition = community_louvain.best_partition(G, random_state=42)

    # 计算变化：旧 community_id vs 新 partition
    # 新 community_id = "sub_{prefix}_{louvain_id}" 避免与全局 ID 冲突
    prefix = "_".join(sorted(affected_communities[:3]))
    changed = 0
    new_cids = set()

    # 批量收集需要更新的节点
    updates = []
    for node, new_local_id in partition.items():
        new_cid = f"sub_{prefix}_{new_local_id}"
        new_cids.add(new_cid)
        old_cid = old_cid_map.get(node)

        if old_cid != new_cid:
            updates.append({"name": node, "cid": new_cid})
            changed += 1

    # 批量写回 Neo4j（每 100 条一批）
    for i in range(0, len(updates), 100):
        batch = updates[i : i + 100]
        write_cypher(
            "UNWIND $batch AS item "
            "MATCH (e:Entity {name: item.name}) "
            "SET e.community_id = item.cid",
            {"batch": batch},
        )

    log.info(
        "subgraph_reclustered",
        affected=affected_communities,
        nodes=G.number_of_nodes(),
        changed=changed,
        new_communities=list(new_cids),
    )

    return {
        "changed_nodes": changed,
        "old_communities": affected_communities,
        "new_communities": list(new_cids),
    }


def incremental_cluster_after_ingest(filename: str) -> dict:
    """文档摄入后对新实体执行增量聚类。

    流程：
    1. 查询该文件新增的实体
    2. 对每个新实体尝试局部补丁
    3. 收集需要重构的社区
    4. 批量子图重构
    5. 标记受影响社区为 dirty

    Returns:
        {"patched": int, "reclustered": int, "affected_communities": list}
    """
    from backend.graph.community import mark_communities_dirty

    # 查询该文件产生的新实体（通过 source_chunks 关联）
    cypher = """
    MATCH (e:Entity)-[r:RELATES_TO]-()
    WHERE any(cid IN r.source_chunks WHERE cid STARTS WITH $prefix)
    RETURN DISTINCT e.name AS name, e.community_id AS cid
    """
    rows = run_cypher(cypher, {"prefix": filename.replace(".", "_")})

    if not rows:
        # 也检查直接通过文件名关联的实体
        cypher2 = """
        MATCH (e:Entity)
        WHERE e.community_id IS NULL OR e.community_id = ''
        RETURN e.name AS name, e.community_id AS cid
        LIMIT 50
        """
        rows = run_cypher(cypher2, {})

    if not rows:
        return {"patched": 0, "reclustered": 0, "affected_communities": []}

    patched = 0
    recluster_communities = set()

    for row in rows:
        node_name = row["name"]
        if row.get("cid"):
            continue  # 已有 community_id，跳过

        result = patch_new_node(node_name)

        if result["action"] == "patched":
            patched += 1
        elif result["action"] == "recluster":
            recluster_communities.update(result.get("affected_communities", []))

    # 批量重构
    reclustered = 0
    all_new_cids = []
    if recluster_communities:
        recluster_result = recluster_subgraph(list(recluster_communities))
        reclustered = recluster_result["changed_nodes"]
        all_new_cids = recluster_result["new_communities"]

    # 标记受影响社区为 dirty
    all_affected = list(recluster_communities) + all_new_cids
    if all_affected:
        mark_communities_dirty(all_affected)

    log.info(
        "incremental_cluster_complete",
        filename=filename,
        patched=patched,
        reclustered=reclustered,
        affected=all_affected,
    )

    return {
        "patched": patched,
        "reclustered": reclustered,
        "affected_communities": all_affected,
    }
