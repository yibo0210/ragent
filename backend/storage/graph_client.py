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
        _neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    return _neo4j_driver


def run_cypher(query: str, params: dict = None) -> list[dict]:
    """执行只读 Cypher 查询，返回记录列表。"""
    with _get_driver().session() as session:
        result = session.run(query, params or {})
        return [dict(record) for record in result]


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
