"""实体消歧：两阶段同义词检测 + Cypher 节点融合。"""
import os
from difflib import SequenceMatcher
from backend.storage.graph_client import run_cypher, write_cypher

SIMILARITY_THRESHOLD = float(os.getenv("ENTITY_SIM_THRESHOLD", "0.75"))


def find_candidates_in_community(community_id: str = None) -> list[dict]:
    """阶段一：从社区内通过编辑距离召回可疑同义词对。"""
    cypher = """
    MATCH (a:Entity)
    WHERE a.community_id IS NOT NULL
    MATCH (b:Entity)
    WHERE b.community_id = a.community_id AND id(a) < id(b) AND a.type = b.type
    WITH a, b, a.community_id AS cid
    WHERE a.name <> b.name
    RETURN a.name AS name_a, b.name AS name_b, cid AS community_id
    LIMIT 500
    """
    records = run_cypher(cypher)
    candidates = []
    for r in records:
        ratio = SequenceMatcher(
            None, r["name_a"].lower(), r["name_b"].lower()
        ).ratio()
        if ratio >= SIMILARITY_THRESHOLD:
            candidates.append({
                "name_a": r["name_a"],
                "name_b": r["name_b"],
                "community_id": r["community_id"],
                "edit_ratio": round(ratio, 3),
            })
    return candidates


def resolve_entities_batch(candidates: list[dict]) -> list[dict]:
    """阶段二：LLM 确认同义关系。返回确认要合并的实体对。"""
    if not candidates:
        return []

    from backend.agent.orchestrator import _get_worker_model

    model = _get_worker_model()
    confirmed = []

    for c in candidates[:20]:
        prompt = (
            f"节点A: '{c['name_a']}'，节点B: '{c['name_b']}'。"
            f"编辑距离相似度: {c['edit_ratio']}。"
            f"它们属于同一社区 (community_id={c['community_id']})。"
            f"请判断这两个名称是否指代同一实体。"
            f"只回复 'YES' 或 'NO'。"
        )
        try:
            response = model.invoke(prompt)
            if response and hasattr(response, "content"):
                text = response.content.strip().upper()
                if "YES" in text[:10]:
                    confirmed.append(c)
        except Exception:
            continue

    return confirmed


def merge_entity_pair(name_a: str, name_b: str) -> dict:
    """将 name_b 合并到 name_a，继承所有边。"""
    cypher = """
    MATCH (a:Entity {name: $name_a})
    MATCH (b:Entity {name: $name_b})
    OPTIONAL MATCH (b)-[r_in:RELATES_TO]->(other)
    OPTIONAL MATCH (other2)-[r_out:RELATES_TO]->(b)
    FOREACH (_ IN CASE WHEN r_in IS NOT NULL THEN [1] ELSE [] END |
      MERGE (a)-[new_r:RELATES_TO]->(other)
      SET new_r = properties(r_in)
    )
    FOREACH (_ IN CASE WHEN r_out IS NOT NULL THEN [1] ELSE [] END |
      MERGE (other2)-[new_r:RELATES_TO]->(a)
      SET new_r = properties(r_out)
    )
    DETACH DELETE b
    RETURN a.name AS merged_into
    """
    records = write_cypher(cypher, {"name_a": name_a, "name_b": name_b})
    return {"merged_into": name_a, "removed": name_b, **records}


def run_entity_resolution() -> dict:
    """完整消歧流程。"""
    candidates = find_candidates_in_community()
    confirmed = resolve_entities_batch(candidates)
    results = []
    for pair in confirmed:
        result = merge_entity_pair(pair["name_a"], pair["name_b"])
        results.append(result)
    return {
        "candidates_found": len(candidates),
        "llm_confirmed": len(confirmed),
        "merged_pairs": results,
    }
