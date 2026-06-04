"""SubgraphRetriever: extracts n-hop subgraph from Neo4j into NetworkX."""

from __future__ import annotations

import networkx as nx

from backend.storage.graph_client import run_cypher


class SubgraphRetriever:
    """Extracts a reasoning subgraph centered on given entities."""

    def retrieve(
        self,
        entity_names: list[str],
        max_hops: int = 3,
        tenant_id: int = None,
        limit: int = 1000,
    ) -> nx.DiGraph:
        if not entity_names:
            return nx.DiGraph()

        params: dict = {"names": entity_names}

        cypher = f"""
            MATCH p = (a:Entity)-[:RELATES_TO*1..{max_hops}]->(b:Entity)
            WHERE a.name IN $names
            WITH p, relationships(p) AS rels, nodes(p) AS nds
            UNWIND range(0, size(rels)-1) AS i
            WITH nds[i] AS src, rels[i] AS r, nds[i+1] AS tgt
            RETURN DISTINCT
                src.name AS subject,
                r.predicate AS predicate,
                tgt.name AS object,
                r.description AS desc,
                r.weight AS weight,
                r.valid_from AS valid_from,
                r.valid_to AS valid_to
            LIMIT {limit}
        """

        rows = run_cypher(cypher, params, timeout=5.0)

        G = nx.DiGraph()
        for row in rows:
            s = row.get("subject", "")
            o = row.get("object", "")
            p = row.get("predicate", "")
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, predicate=p, weight=float(row.get("weight", 1.0)),
                       desc=row.get("desc", ""),
                       valid_from=str(row.get("valid_from", "")),
                       valid_to=str(row.get("valid_to", "")))

        return G

    def has_entity(self, G: nx.DiGraph, name: str) -> bool:
        return name in G.nodes

    def node_count(self, G: nx.DiGraph) -> int:
        return G.number_of_nodes()

    def edge_count(self, G: nx.DiGraph) -> int:
        return G.number_of_edges()


_subgraph_retriever: SubgraphRetriever | None = None


def get_subgraph_retriever() -> SubgraphRetriever:
    global _subgraph_retriever
    if _subgraph_retriever is None:
        _subgraph_retriever = SubgraphRetriever()
    return _subgraph_retriever
