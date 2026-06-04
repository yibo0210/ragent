"""意图驱动的动态 RRF 权重分配。

根据 Query Profiler 输出的意图标签，从配置文件加载对应的权重向量，
替代原有的静态环境变量权重。
"""
import os
from pathlib import Path
from typing import Tuple

import yaml

from backend.observability import get_logger

log = get_logger("ragent.dynamic_rrf")

# 配置文件路径
_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "weight_matrix.yaml"

# 模块级缓存
_matrix_cache = None


def load_weight_matrix() -> dict:
    """加载权重矩阵配置（带缓存）。"""
    global _matrix_cache
    if _matrix_cache is not None:
        return _matrix_cache
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _matrix_cache = data if data else {}
    except Exception as e:
        log.warning("weight_matrix_load_failed", error=str(e))
        _matrix_cache = {}
    return _matrix_cache


def reload_weight_matrix() -> dict:
    """强制重新加载权重矩阵（用于配置热更新）。"""
    global _matrix_cache
    _matrix_cache = None
    return load_weight_matrix()


def get_weights_for_intent(intent_level: str, query_type: str = "") -> Tuple[float, float, float, float]:
    """根据意图级别返回 RRF 权重向量 (dense, sparse, graph, visual/community)。

    v17: query_type 优先查找（6-type mapping）；回退到 intent_level（v12 compat）。
    """
    matrix = load_weight_matrix()
    # priority 1: query_type (v17 6-type)
    entry = None
    if query_type and query_type in matrix:
        entry = matrix[query_type]
    # priority 2: intent_level (v12 L1/L2/L3 compat)
    if entry is None and intent_level:
        entry = matrix.get(intent_level)
    # priority 3: DEFAULT
    if entry is None:
        entry = matrix.get("DEFAULT")
    if not entry:
        return (0.4, 0.3, 0.15, 0.15)
    weights = entry.get("weights", [0.4, 0.3, 0.15, 0.15])
    while len(weights) < 4:
        weights.append(0.0)
    return tuple(weights[:4])
