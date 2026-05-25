"""Neo4j 约束与索引初始化，应用启动时幂等执行。"""
from .graph_client import write_cypher

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_community IF NOT EXISTS FOR (e:Entity) ON (e.community_id)",
]


def init_graph_schema():
    """应用启动时调用，创建约束和索引。"""
    for cypher in CONSTRAINTS + INDEXES:
        try:
            write_cypher(cypher)
        except Exception as e:
            print(f"[GRAPH] Schema init warning: {e}")
    print("[GRAPH] Schema initialized")
