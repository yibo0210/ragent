"""图增强检索引擎 — 局部搜索（向量 + 图谱外扩）和全局搜索（社区摘要）。"""
from backend.rag.utils import retrieve_documents
from backend.storage.graph_client import run_cypher


def local_graph_search(
    query: str, top_k: int = 5, graph_hops: int = 1, time_filter: dict = None,
    tenant_id: int = None,
) -> dict:
    """
    局部图搜索:
    1. Milvus 混合检索 → Top-K 文本块
    2. 提取块中关联的 Neo4j 实体 → 向外扩散 1-hop
    3. 合并 CHUNKS + GRAPH TRIPLES → 返回上下文
    """
    # Step 1: Vector search
    retrieved = retrieve_documents(query, top_k=top_k, tenant_id=tenant_id)
    chunks = retrieved.get("docs", [])

    # Step 2: Graph expansion
    chunk_ids = [c.get("chunk_id", "") for c in chunks if c.get("chunk_id")]
    graph_triples = []

    if chunk_ids:
        time_clause = ""
        time_params = {"chunk_ids": chunk_ids}
        if time_filter and time_filter.get("year"):
            year = str(time_filter["year"])
            if time_filter.get("mode") == "at_or_after":
                time_clause = """
        AND (r.valid_from = '' OR r.valid_from <= $filter_year)
        AND (r.valid_to = '' OR r.valid_to >= $filter_year)
        """
            elif time_filter.get("mode") == "before":
                time_clause = """
        AND r.valid_to <> '' AND r.valid_to < $filter_year
        """
            elif time_filter.get("mode") == "exact":
                time_clause = """
        AND (r.valid_from = $filter_year OR r.valid_to = $filter_year)
        """
            time_params["filter_year"] = year

        tenant_clause = ""
        if tenant_id is not None:
            tenant_clause = "AND a.tenant_id = $tenant_id AND b.tenant_id = $tenant_id"
            time_params["tenant_id"] = tenant_id

        cypher = f"""
        MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
        WHERE any(cid IN r.source_chunks WHERE cid IN $chunk_ids)
        {time_clause}
        {tenant_clause}
        RETURN a.name AS subject, r.predicate AS predicate, b.name AS object,
               r.description AS desc, r.weight AS weight
        LIMIT 30
        """
        triples = run_cypher(cypher, time_params)
        graph_triples.extend(triples)

        # v18: True multi-hop expansion loop
        if graph_hops >= 1 and triples:
            expanded_entities: set[str] = set()
            for t in triples:
                expanded_entities.add(t["subject"])
                expanded_entities.add(t["object"])

            neighbor_tenant_clause = ""
            if tenant_id is not None:
                neighbor_tenant_clause = "AND e.tenant_id = $tenant_id AND other.tenant_id = $tenant_id"

            for hop in range(1, graph_hops):
                if len(expanded_entities) > 50:
                    break
                neighbor_params: dict = {"names": list(expanded_entities)}
                if tenant_id is not None:
                    neighbor_params["tenant_id"] = tenant_id

                neighbor_cypher = f"""
                MATCH (e:Entity)-[r:RELATES_TO]-(other:Entity)
                WHERE e.name IN $names
                  AND NOT other.name IN $names
                  {neighbor_tenant_clause}
                RETURN DISTINCT e.name AS source, r.predicate AS predicate,
                       other.name AS target, r.description AS desc,
                       r.weight AS weight
                LIMIT 30
                """
                neighbors = run_cypher(neighbor_cypher, neighbor_params)
                if not neighbors:
                    break
                graph_triples.extend(neighbors)
                for n in neighbors:
                    expanded_entities.add(n.get("target", ""))

    # Step 3: Normalize keys
    normalized_triples = []
    for t in graph_triples:
        s = t.get("subject") or t.get("source", "")
        o = t.get("object") or t.get("target", "")
        p = t.get("predicate", "")
        d = t.get("desc", "")
        normalized_triples.append({"s": s, "p": p, "o": o, "d": d})

    # Step 4: Format context
    chunk_texts = [
        f"[Chunk {i+1}] {c.get('text', '')[:400]}"
        for i, c in enumerate(chunks)
    ]
    triple_texts = [
        f"({t['s']})-[{t['p']}]->({t['o']})"
        + (f": {t['d']}" if t['d'] else "")
        for t in normalized_triples
    ]

    context = "\n\n".join(
        ["## 检索到的文本片段", *chunk_texts]
        + (["## 知识图谱关系", *triple_texts] if triple_texts else [])
    )

    return {
        "chunks": chunks,
        "graph_triples": graph_triples,
        "context": context,
        "mode": "local_graph",
    }


def global_graph_search(query: str, top_k: int = 5, tenant_id: int = None) -> dict:
    """
    全局图搜索: 在社区摘要索引中检索匹配的社区综述。
    依赖 Phase 3 已将摘要索引到 Milvus（file_type = CommunitySummary）。
    """
    from backend.milvus.client import MilvusManager

    milvus = MilvusManager()
    milvus.init_collection()

    summaries = []
    try:
        filter_expr = 'file_type == "CommunitySummary"'
        if tenant_id is not None:
            filter_expr += f' && (tenant_id == {tenant_id})'
        raw = milvus.query(
            filter_expr=filter_expr,
            output_fields=["text", "filename", "chunk_id"],
            limit=100,
        )
        if raw:
            # Simple keyword matching as vector search proxy
            query_lower = query.lower()
            scored = []
            for item in raw:
                text = item.get("text", "")
                score = sum(1 for word in query_lower.split() if word in text.lower())
                if score > 0:
                    scored.append((score, item))
            scored.sort(key=lambda x: x[0], reverse=True)
            summaries = [item for _, item in scored[:top_k]]
    except Exception:
        pass

    context_parts = [
        f"## 社区摘要 {i+1}\n{s.get('text', '')}"
        for i, s in enumerate(summaries)
    ]

    return {
        "summaries": summaries,
        "context": "\n\n".join(context_parts) if context_parts else "暂无相关社区摘要。",
        "mode": "global_graph",
    }
