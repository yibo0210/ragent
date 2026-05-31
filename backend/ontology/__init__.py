"""领域本体约束层：定义合法的实体类型、关系谓词和三元组规则。"""
from .schema import (
    ENTITY_TYPES,
    ENTITY_TYPE_SET,
    RELATION_PREDICATES,
    RELATION_PREDICATE_SET,
    RELATION_RULES,
    is_valid_relation,
    get_allowed_predicates,
)

__all__ = [
    "ENTITY_TYPES",
    "ENTITY_TYPE_SET",
    "RELATION_PREDICATES",
    "RELATION_PREDICATE_SET",
    "RELATION_RULES",
    "is_valid_relation",
    "get_allowed_predicates",
]
