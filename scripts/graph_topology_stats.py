#!/usr/bin/env python3
"""图谱拓扑统计脚本。

查询 Neo4j 收集图谱结构指标：节点数、边数、孤岛率、类型分布、谓词分布等。
用于 v10 A/B 对比实验，验证受控抽取对图谱质量的改善效果。

用法:
    python scripts/graph_topology_stats.py
    python scripts/graph_topology_stats.py --output before.json
"""
import argparse
import json
import sys

sys.path.insert(0, ".")

from backend.storage.graph_client import run_cypher


def collect_topology_stats() -> dict:
    """收集图谱拓扑指标。"""
    stats: dict = {}

    # 总节点数
    rows = run_cypher("MATCH (e:Entity) RETURN count(e) AS total")
    total_nodes = rows[0]["total"] if rows else 0
    stats["total_nodes"] = total_nodes

    # 总边数
    rows = run_cypher("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS total")
    total_edges = rows[0]["total"] if rows else 0
    stats["total_edges"] = total_edges

    # 孤岛节点（无任何边）
    rows = run_cypher("MATCH (e:Entity) WHERE NOT (e)--() RETURN count(e) AS orphans")
    orphan_nodes = rows[0]["orphans"] if rows else 0
    stats["orphan_nodes"] = orphan_nodes

    # 边密度（有向图：实际边数 / 理论最大边数）
    if total_nodes > 1:
        stats["edge_density"] = round(total_edges / (total_nodes * (total_nodes - 1)), 6)
    else:
        stats["edge_density"] = 0.0

    # 平均度
    if total_nodes > 0:
        stats["avg_degree"] = round(2 * total_edges / total_nodes, 2)
    else:
        stats["avg_degree"] = 0.0

    # 孤岛率
    if total_nodes > 0:
        stats["orphan_rate"] = round(orphan_nodes / total_nodes, 4)
    else:
        stats["orphan_rate"] = 0.0

    # 类型分布
    rows = run_cypher(
        "MATCH (e:Entity) RETURN e.type AS type, count(e) AS cnt ORDER BY cnt DESC"
    )
    stats["type_distribution"] = {r["type"]: r["cnt"] for r in rows if r["type"]}

    # 谓词分布
    rows = run_cypher(
        "MATCH ()-[r:RELATES_TO]->() RETURN r.predicate AS predicate, count(r) AS cnt ORDER BY cnt DESC"
    )
    stats["predicate_distribution"] = {r["predicate"]: r["cnt"] for r in rows if r["predicate"]}

    # 度分布统计
    rows = run_cypher(
        "MATCH (e:Entity) "
        "OPTIONAL MATCH (e)-[r:RELATES_TO]-() "
        "WITH e, count(r) AS deg "
        "RETURN deg ORDER BY deg DESC LIMIT 1000"
    )
    degrees = [r["deg"] for r in rows] if rows else []
    if degrees:
        degrees.sort()
        n = len(degrees)
        stats["degree_p50"] = degrees[n // 2]
        stats["degree_p90"] = degrees[int(n * 0.9)]
        stats["degree_p99"] = degrees[int(n * 0.99)]
        stats["degree_max"] = degrees[-1]
    else:
        stats["degree_p50"] = 0
        stats["degree_p90"] = 0
        stats["degree_p99"] = 0
        stats["degree_max"] = 0

    return stats


def main():
    parser = argparse.ArgumentParser(description="图谱拓扑统计")
    parser.add_argument("--output", default="topology_stats.json", help="输出 JSON 路径")
    args = parser.parse_args()

    print("[TOPOLOGY] Collecting graph stats from Neo4j...")
    stats = collect_topology_stats()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[TOPOLOGY] Results saved to {args.output}")
    print(f"  Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}, "
          f"Orphans: {stats['orphan_nodes']} ({stats['orphan_rate']:.1%})")
    print(f"  Avg degree: {stats['avg_degree']}, Edge density: {stats['edge_density']}")
    print(f"  Types: {stats['type_distribution']}")


if __name__ == "__main__":
    main()
