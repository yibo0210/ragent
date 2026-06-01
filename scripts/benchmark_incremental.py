#!/usr/bin/env python3
"""增量聚类 Benchmark — 对比全量 Louvain vs 增量局部补丁 + 子图重构。

用法:
    python scripts/benchmark_incremental.py
    python scripts/benchmark_incremental.py --output benchmark_result.csv
"""
import argparse
import csv
import json
import sys
import time
import random

sys.path.insert(0, ".")

import networkx as nx
import community as community_louvain


def build_test_graph(n_nodes: int, n_communities: int, edge_density: float = 0.02) -> nx.Graph:
    """构建测试图：n_nodes 个节点，n_communities 个社区。"""
    G = nx.planted_partition_graph(n_communities, n_nodes // n_communities, edge_density, 0.001, seed=42)
    return G


def benchmark_full_louvain(G: nx.Graph) -> dict:
    """全量 Louvain 聚类。"""
    t0 = time.time()
    partition = community_louvain.best_partition(G, random_state=42)
    elapsed = time.time() - t0
    n_communities = len(set(partition.values()))
    return {"method": "full_louvain", "time_s": round(elapsed, 3), "communities": n_communities, "nodes": G.number_of_nodes()}


def benchmark_incremental_patch(G: nx.Graph, n_new_nodes: int = 10) -> dict:
    """增量局部补丁：模拟添加 n_new_nodes 个新节点。"""
    existing_partition = community_louvain.best_partition(G, random_state=42)

    # 获取图中已有的节点列表
    existing_nodes = list(G.nodes())

    t0 = time.time()
    patched = 0
    reclustering_needed = 0
    patch_threshold = 0.6

    for i in range(n_new_nodes):
        # 模拟新节点连接到 3-5 个已有节点
        neighbors = random.sample(existing_nodes, min(random.randint(3, 5), len(existing_nodes)))

        # 统计邻居社区分布
        from collections import Counter
        cid_counts = Counter(existing_partition.get(n) for n in neighbors if existing_partition.get(n) is not None)

        if cid_counts:
            total = sum(cid_counts.values())
            top_cid, top_count = cid_counts.most_common(1)[0]
            ratio = top_count / total

            if ratio >= patch_threshold:
                patched += 1
            else:
                reclustering_needed += 1

    elapsed = time.time() - t0
    return {
        "method": "incremental_patch",
        "time_s": round(elapsed, 6),
        "patched": patched,
        "recluster_needed": reclustering_needed,
        "nodes": G.number_of_nodes(),
    }


def benchmark_incremental_subgraph(G: nx.Graph, n_affected_communities: int = 3) -> dict:
    """增量子图重构：仅对 n_affected_communities 个社区运行 Louvain。"""
    existing_partition = community_louvain.best_partition(G, random_state=42)
    all_cids = list(set(existing_partition.values()))

    # 选取受影响的社区
    affected = random.sample(all_cids, min(n_affected_communities, len(all_cids)))

    # 提取子图节点
    subgraph_nodes = [n for n, cid in existing_partition.items() if cid in affected]
    subG = G.subgraph(subgraph_nodes).copy()

    t0 = time.time()
    if subG.number_of_nodes() >= 2:
        new_partition = community_louvain.best_partition(subG, random_state=42)
        changed = sum(1 for n in subG.nodes() if new_partition.get(n) != existing_partition.get(n))
    else:
        changed = 0
    elapsed = time.time() - t0

    return {
        "method": "incremental_subgraph",
        "time_s": round(elapsed, 3),
        "subgraph_nodes": subG.number_of_nodes(),
        "subgraph_edges": subG.number_of_edges(),
        "changed_nodes": changed,
        "full_nodes": G.number_of_nodes(),
    }


def benchmark_summary_token_estimate(n_communities: int, dirty_ratio: float) -> dict:
    """估算摘要生成的 Token 消耗对比。"""
    avg_tokens_per_summary = 800  # 输入 + 输出
    cost_per_1k_tokens = 0.002  # qwen-plus 价格（元）

    full_tokens = n_communities * avg_tokens_per_summary
    full_cost = full_tokens / 1000 * cost_per_1k_tokens

    dirty_count = int(n_communities * dirty_ratio)
    incremental_tokens = dirty_count * avg_tokens_per_summary
    incremental_cost = incremental_tokens / 1000 * cost_per_1k_tokens

    savings_pct = (1 - incremental_tokens / full_tokens) * 100 if full_tokens > 0 else 0

    return {
        "n_communities": n_communities,
        "dirty_ratio": dirty_ratio,
        "dirty_count": dirty_count,
        "full_tokens": full_tokens,
        "full_cost_yuan": round(full_cost, 4),
        "incremental_tokens": incremental_tokens,
        "incremental_cost_yuan": round(incremental_cost, 4),
        "savings_pct": round(savings_pct, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="v13 增量聚类 Benchmark")
    parser.add_argument("--output", default="benchmark_result.csv", help="输出 CSV 文件")
    args = parser.parse_args()

    results = []

    print("=" * 60)
    print("v13 增量聚类 Benchmark")
    print("=" * 60)

    # 测试不同规模
    for n_nodes in [1000, 5000, 20000]:
        n_communities = max(5, n_nodes // 100)
        print(f"\n--- 图规模: {n_nodes} 节点, {n_communities} 社区 ---")

        G = build_test_graph(n_nodes, n_communities)

        # 全量 Louvain
        r1 = benchmark_full_louvain(G)
        print(f"  全量 Louvain: {r1['time_s']}s, {r1['communities']} communities")
        results.append(r1)

        # 增量局部补丁
        r2 = benchmark_incremental_patch(G, n_new_nodes=10)
        print(f"  增量补丁 (10新节点): {r2['time_s']*1000:.1f}ms, patched={r2['patched']}, recluster={r2['recluster_needed']}")
        results.append(r2)

        # 增量子图重构
        r3 = benchmark_incremental_subgraph(G, n_affected_communities=3)
        speedup = r1['time_s'] / r3['time_s'] if r3['time_s'] > 0 else float('inf')
        print(f"  增量子图重构 (3社区): {r3['time_s']}s, {r3['subgraph_nodes']} nodes, 加速 {speedup:.1f}x")
        results.append(r3)

    # Token 成本对比
    print(f"\n--- Token 成本对比 ---")
    for n_comm in [10, 50, 100]:
        for dirty_ratio in [0.05, 0.10, 0.20]:
            r = benchmark_summary_token_estimate(n_comm, dirty_ratio)
            print(f"  {n_comm} 社区, {dirty_ratio*100:.0f}% dirty: 全量 {r['full_cost_yuan']:.2f}元 vs 增量 {r['incremental_cost_yuan']:.2f}元 (节省 {r['savings_pct']}%)")
            results.append(r)

    # 保存 CSV
    if results:
        keys = set()
        for r in results:
            keys.update(r.keys())
        keys = sorted(keys)

        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
