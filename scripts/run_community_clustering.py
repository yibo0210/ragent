"""独立脚本: 执行社区聚类 + 摘要生成。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv; load_dotenv()

from backend.graph.community import (
    build_graph_from_neo4j,
    run_leiden_clustering,
    write_community_ids,
    generate_all_summaries,
    index_summaries_to_milvus,
)

if __name__ == "__main__":
    print("Step 1/4: Building graph from Neo4j...")
    G = build_graph_from_neo4j()
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    print("Step 2/4: Running Leiden clustering...")
    partition = run_leiden_clustering(G)
    n_communities = len(set(partition.values()))
    print(f"  Communities: {n_communities}")

    print("Step 3/4: Writing community IDs back to Neo4j...")
    write_community_ids(partition)

    print("Step 4/4: Generating and indexing summaries...")
    summaries = generate_all_summaries(partition)
    index_summaries_to_milvus(summaries)
    print("Done!")
