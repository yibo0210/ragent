"""PathExplorer: discovers reasoning paths through a subgraph."""

from __future__ import annotations

import networkx as nx

from backend.rag.graph_reasoning.schemas import ReasoningPlan, ReasoningPath


class PathExplorer:
    """Finds candidate reasoning paths through a NetworkX graph."""

    def __init__(self, beam_width: int = 10):
        self.beam_width = beam_width

    def explore(
        self,
        G: nx.DiGraph,
        plan: ReasoningPlan,
        query: str = "",
    ) -> list[ReasoningPath]:
        if G.number_of_nodes() == 0 or not plan.start_entities:
            return []

        paths: list[ReasoningPath] = []
        start_entities = [e for e in plan.start_entities if e in G.nodes]

        for start in start_entities:
            for target in G.nodes:
                if target == start:
                    continue
                try:
                    raw_paths = list(nx.all_simple_paths(
                        G, start, target, cutoff=plan.max_hops
                    ))
                    for node_list in raw_paths:
                        edges = []
                        for i in range(len(node_list) - 1):
                            edge_data = G.get_edge_data(node_list[i], node_list[i + 1])
                            if edge_data:
                                p = edge_data.get("predicate", "")
                                edges.append(p)

                        paths.append(ReasoningPath(
                            nodes=node_list,
                            edges=edges,
                            confidence=1.0,
                            hop_count=len(node_list) - 1,
                            path_score=1.0,
                        ))
                except nx.NetworkXNoPath:
                    continue

                if len(paths) >= 100:
                    break
            if len(paths) >= 100:
                break

        return paths

    def beam_search(
        self,
        G: nx.DiGraph,
        start_entity: str,
        max_hops: int = 3,
        beam_width: int = 10,
    ) -> list[ReasoningPath]:
        if start_entity not in G.nodes:
            return []

        frontier: list[ReasoningPath] = [
            ReasoningPath(nodes=[start_entity], edges=[], hop_count=0, path_score=1.0)
        ]
        all_paths: list[ReasoningPath] = []

        for _ in range(1, max_hops + 1):
            candidates: list[ReasoningPath] = []
            for path in frontier:
                last_node = path.nodes[-1]
                for neighbor in G.neighbors(last_node):
                    edge_data = G.get_edge_data(last_node, neighbor)
                    if not edge_data:
                        continue
                    weight = float(edge_data.get("weight", 1.0))
                    new_path = ReasoningPath(
                        nodes=path.nodes + [neighbor],
                        edges=path.edges + [edge_data.get("predicate", "")],
                        hop_count=path.hop_count + 1,
                        path_score=path.path_score * weight,
                    )
                    candidates.append(new_path)

            candidates.sort(key=lambda p: p.path_score, reverse=True)
            frontier = candidates[:beam_width]
            all_paths.extend(frontier)

        return all_paths


_explorer: PathExplorer | None = None


def get_path_explorer(beam_width: int = 10) -> PathExplorer:
    global _explorer
    if _explorer is None:
        _explorer = PathExplorer(beam_width=beam_width)
    return _explorer
