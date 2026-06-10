# backend/research/evidence_graph.py
"""EvidenceGraph: stores evidence as Neo4j graph nodes with SUPPORTS/REFUTES relations."""

from __future__ import annotations

import uuid

from backend.storage.graph_client import write_cypher, run_cypher
from backend.research.schemas import EvidenceNode, EvidenceRelationType, Hypothesis


class EvidenceGraph:
    """Persists evidence as a graph in Neo4j for relationship-based analysis."""

    def add_evidence(self, ev: EvidenceNode) -> bool:
        """CREATE or MERGE an :EvidenceNode in Neo4j."""
        if not ev.node_id:
            ev.node_id = f"ev_{uuid.uuid4().hex[:12]}"

        cypher = """
            MERGE (e:EvidenceNode {node_id: $node_id})
            ON CREATE SET
                e.content = $content,
                e.source = $source,
                e.citation = $citation,
                e.confidence = $confidence,
                e.hypothesis_id = $hypothesis_id,
                e.task_id = $task_id,
                e.execution_id = $execution_id,
                e.created_at = $created_at
        """
        write_cypher(cypher, {
            "node_id": ev.node_id, "content": ev.content, "source": ev.source,
            "citation": ev.citation, "confidence": ev.confidence,
            "hypothesis_id": ev.hypothesis_id, "task_id": ev.task_id,
            "execution_id": ev.execution_id, "created_at": ev.created_at,
        })
        return True

    def link_to_hypothesis(self, evidence_id: str, hypothesis_id: str, rel_type: EvidenceRelationType):
        """Link evidence to a Hypothesis node."""
        cypher = f"""
            MATCH (e:EvidenceNode {{node_id: $evidence_id}})
            MATCH (h:Hypothesis {{hypothesis_id: $hypothesis_id}})
            MERGE (e)-[:{rel_type.value}]->(h)
        """
        try:
            write_cypher(cypher, {"evidence_id": evidence_id, "hypothesis_id": hypothesis_id})
        except Exception:
            pass

    def get_by_hypothesis(self, hypothesis_id: str) -> list[EvidenceNode]:
        """Retrieve all evidence for a specific hypothesis."""
        cypher = """
            MATCH (e:EvidenceNode {hypothesis_id: $hid})
            RETURN e ORDER BY e.confidence DESC
        """
        rows = run_cypher(cypher, {"hid": hypothesis_id})
        return [self._to_node(r["e"]) for r in rows]

    def get_by_execution(self, execution_id: str) -> list[EvidenceNode]:
        cypher = """
            MATCH (e:EvidenceNode {execution_id: $eid})
            RETURN e ORDER BY e.confidence DESC
        """
        rows = run_cypher(cypher, {"eid": execution_id})
        return [self._to_node(r["e"]) for r in rows]

    def get_evidence_graph(self, execution_id: str) -> dict:
        """Return full evidence graph as nodes+edges for frontend visualization."""
        cypher = """
            MATCH (e:EvidenceNode {execution_id: $eid})
            OPTIONAL MATCH (e)-[r:SUPPORTS|REFUTES|RELATES_TO]->(h:Hypothesis)
            RETURN e, r, h LIMIT 200
        """
        rows = run_cypher(cypher, {"eid": execution_id})
        nodes = []
        edges = []
        node_ids = set()
        for row in rows:
            e_data = row["e"]
            if e_data.get("node_id") not in node_ids:
                nodes.append({
                    "id": e_data.get("node_id", ""),
                    "label": e_data.get("content", "")[:80],
                    "source": e_data.get("source", ""),
                    "confidence": float(e_data.get("confidence", 0.5)),
                    "hypothesis_id": e_data.get("hypothesis_id", ""),
                })
                node_ids.add(e_data.get("node_id", ""))
            if row.get("r") and row.get("h"):
                edges.append({
                    "from": e_data.get("node_id", ""),
                    "to": row["h"].get("hypothesis_id", ""),
                    "type": type(row["r"]).__name__.replace("SUPPORTS", "SUPPORTS").replace("REFUTES", "REFUTES"),
                })
        return {"nodes": nodes, "edges": edges}

    def add_hypothesis(self, hyp: Hypothesis) -> bool:
        """CREATE a :Hypothesis node in Neo4j."""
        cypher = """
            MERGE (h:Hypothesis {hypothesis_id: $hid})
            ON CREATE SET
                h.statement = $statement,
                h.rationale = $rationale,
                h.status = $status,
                h.confidence = $confidence
        """
        write_cypher(cypher, {
            "hid": hyp.hypothesis_id, "statement": hyp.statement,
            "rationale": hyp.rationale, "status": hyp.status.value,
            "confidence": hyp.confidence,
        })
        return True

    def link_evidence_pair(self, ev_a: str, ev_b: str, rel_type: EvidenceRelationType, conflict_reason: str = ""):
        """Link two evidence nodes to each other."""
        cypher = f"""
            MATCH (a:EvidenceNode {{node_id: $ev_a}})
            MATCH (b:EvidenceNode {{node_id: $ev_b}})
            MERGE (a)-[:{rel_type.value} {{reason: $reason}}]->(b)
        """
        try:
            write_cypher(cypher, {"ev_a": ev_a, "ev_b": ev_b, "reason": conflict_reason})
        except Exception:
            pass

    def _to_node(self, data: dict) -> EvidenceNode:
        return EvidenceNode(
            node_id=data.get("node_id", ""),
            content=data.get("content", ""),
            source=data.get("source", ""),
            citation=data.get("citation", ""),
            confidence=float(data.get("confidence", 0.5)),
            hypothesis_id=data.get("hypothesis_id", ""),
            task_id=data.get("task_id", ""),
            execution_id=data.get("execution_id", ""),
            created_at=data.get("created_at", ""),
        )


_graph: EvidenceGraph | None = None


def get_evidence_graph() -> EvidenceGraph:
    global _graph
    if _graph is None:
        _graph = EvidenceGraph()
    return _graph
