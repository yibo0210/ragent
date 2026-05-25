"""Neo4j 图数据库客户端，封装 driver 生命周期和常用操作。"""
import os
from neo4j import GraphDatabase, Driver

_neo4j_driver: Driver | None = None


def _get_driver() -> Driver:
    global _neo4j_driver
    if _neo4j_driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        _neo4j_driver = GraphDatabase.driver(
            uri, auth=(user, password),
            max_connection_lifetime=30,
            connection_acquisition_timeout=3,
        )
    return _neo4j_driver


def run_cypher(query: str, params: dict = None, timeout: float = None) -> list[dict]:
    """执行只读 Cypher 查询，返回记录列表，支持超时。"""
    import time
    from backend.observability import get_tracer, Metrics

    if timeout is None:
        timeout = float(os.getenv("NEO4J_QUERY_TIMEOUT", "1.5"))
    tracer = get_tracer("ragent.neo4j")
    with tracer.start_as_current_span("neo4j.run_cypher") as span:
        span.set_attribute("cypher.query", query[:200])
        t0 = time.time()
        with _get_driver().session() as session:
            result = session.run(query, params or {}, timeout=timeout)
            records = [dict(record) for record in result]
        dt = time.time() - t0
        span.set_attribute("duration_ms", dt * 1000)
        span.set_attribute("result_count", len(records))
        Metrics.record_graph_query(dt)
        return records


def write_cypher(query: str, params: dict = None) -> dict:
    """执行写入 Cypher，返回汇总信息。"""
    with _get_driver().session() as session:
        result = session.run(query, params or {})
        summary = result.consume()
        return {
            "nodes_created": summary.counters.nodes_created,
            "relationships_created": summary.counters.relationships_created,
            "properties_set": summary.counters.properties_set,
        }


def close_driver():
    global _neo4j_driver
    if _neo4j_driver:
        _neo4j_driver.close()
        _neo4j_driver = None
