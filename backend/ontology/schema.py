"""领域本体定义：实体类型、关系谓词、三元组合法规则。

通用企业知识领域，兼容现有图谱数据（Model/Data 类型保留）。
新增类型: Product, Event, Document, Metric
新增谓词: CAUSES, IMPLEMENTS, BELONGS_TO, COMPETES_WITH
"""

# ---------------------------------------------------------------------------
# Entity Types (11 种)
# ---------------------------------------------------------------------------

ENTITY_TYPES: list[str] = [
    # --- 现有 7 种 ---
    "Person",        # 人物：研究员、工程师、创始人等
    "Organization",  # 机构：公司、大学、研究院等
    "Technology",    # 技术：框架、语言、平台、工具
    "Concept",       # 概念：理论、思想、设计模式、架构风格
    "Model",         # 模型：LLM、预训练模型、算法模型（向后兼容）
    "Method",        # 方法：算法、技术路线、评测方法
    "Data",          # 数据：数据集、语料库、Benchmark（向后兼容）
    # --- 新增 4 种 ---
    "Product",       # 产品：商业化产品、服务、SaaS 平台
    "Event",         # 事件：会议、发布会、竞赛、版本发布
    "Document",      # 文档：论文、规范、标准、RFC
    "Metric",        # 指标：评测指标、业务 KPI、性能指标
]

ENTITY_TYPE_SET: set[str] = set(ENTITY_TYPES)

# ---------------------------------------------------------------------------
# Relation Predicates (11 种)
# ---------------------------------------------------------------------------

RELATION_PREDICATES: list[str] = [
    # --- 现有 7 种 ---
    "DEPENDS_ON",    # 依赖：A 依赖于 B
    "CONTAINS",      # 包含：A 包含 B
    "CITES",         # 引用：A 引用/参考 B
    "USES",          # 使用：A 使用 B
    "PART_OF",       # 隶属：A 是 B 的一部分
    "PROPOSES",      # 提出：A 提出/发布 B
    "EVALUATES",     # 评估：A 评估/评测 B
    # --- 新增 4 种 ---
    "CAUSES",        # 因果：A 导致/引起 B
    "IMPLEMENTS",    # 实现：A 实现了 B（代码/产品实现概念/标准）
    "BELONGS_TO",    # 归属：A 属于 B（人属于组织、产品属于公司）
    "COMPETES_WITH", # 竞争：A 与 B 存在竞争关系
    "RELATED_TO",    # 关联：A 与 B 存在一般性关联（通用兜底）
]

RELATION_PREDICATE_SET: set[str] = set(RELATION_PREDICATES)

# ---------------------------------------------------------------------------
# Relation Rules: (subject_type, predicate, object_type)
# "*" 表示任意类型
# ---------------------------------------------------------------------------

RELATION_RULES: list[tuple[str, str, str]] = [
    # --- Technology ↔ * ---
    ("Technology", "DEPENDS_ON", "Technology"),
    ("Technology", "CONTAINS", "Technology"),
    ("Technology", "CONTAINS", "Concept"),
    ("Technology", "CONTAINS", "Data"),
    ("Technology", "USES", "Technology"),
    ("Technology", "USES", "Method"),
    ("Technology", "USES", "Concept"),
    ("Technology", "USES", "Data"),
    ("Technology", "IMPLEMENTS", "Technology"),
    ("Technology", "IMPLEMENTS", "Concept"),
    ("Technology", "IMPLEMENTS", "Method"),
    ("Technology", "COMPETES_WITH", "Technology"),
    ("Technology", "PART_OF", "Technology"),
    ("Technology", "DEPENDS_ON", "Data"),

    # --- Organization → * ---
    ("Organization", "PROPOSES", "Technology"),
    ("Organization", "PROPOSES", "Model"),
    ("Organization", "PROPOSES", "Product"),
    ("Organization", "PROPOSES", "Document"),
    ("Organization", "PROPOSES", "Method"),
    ("Organization", "BELONGS_TO", "Organization"),
    ("Organization", "COMPETES_WITH", "Organization"),
    ("Organization", "USES", "Technology"),
    ("Organization", "USES", "Method"),

    # --- Person → * ---
    ("Person", "PROPOSES", "Technology"),
    ("Person", "PROPOSES", "Model"),
    ("Person", "PROPOSES", "Method"),
    ("Person", "PROPOSES", "Document"),
    ("Person", "PROPOSES", "Concept"),
    ("Person", "BELONGS_TO", "Organization"),
    ("Person", "BELONGS_TO", "Event"),
    ("Person", "USES", "Technology"),
    ("Person", "USES", "Method"),
    ("Person", "EVALUATES", "Model"),
    ("Person", "EVALUATES", "Method"),
    ("Person", "EVALUATES", "Product"),

    # --- Model ↔ * ---
    ("Model", "DEPENDS_ON", "Technology"),
    ("Model", "DEPENDS_ON", "Data"),
    ("Model", "PART_OF", "Technology"),
    ("Model", "EVALUATES", "Data"),
    ("Model", "IMPLEMENTS", "Method"),
    ("Model", "IMPLEMENTS", "Concept"),
    ("Model", "COMPETES_WITH", "Model"),
    ("Model", "USES", "Method"),
    ("Model", "USES", "Technology"),
    ("Model", "USES", "Data"),
    ("Model", "CONTAINS", "Technology"),

    # --- Method ↔ * ---
    ("Method", "PART_OF", "Technology"),
    ("Method", "DEPENDS_ON", "Data"),
    ("Method", "DEPENDS_ON", "Technology"),
    ("Method", "EVALUATES", "Model"),
    ("Method", "EVALUATES", "Data"),
    ("Method", "IMPLEMENTS", "Concept"),
    ("Method", "USES", "Technology"),
    ("Method", "USES", "Data"),
    ("Method", "CONTAINS", "Concept"),

    # --- Concept ↔ * ---
    ("Concept", "PART_OF", "Concept"),
    ("Concept", "PART_OF", "Technology"),
    ("Concept", "DEPENDS_ON", "Concept"),
    ("Concept", "DEPENDS_ON", "Technology"),
    ("Concept", "CAUSES", "Concept"),
    ("Concept", "IMPLEMENTS", "Technology"),
    ("Concept", "USES", "Method"),
    ("Concept", "USES", "Technology"),
    ("Concept", "RELATED_TO", "Concept"),
    ("Concept", "RELATED_TO", "Method"),

    # --- Product ↔ * ---
    ("Product", "IMPLEMENTS", "Technology"),
    ("Product", "DEPENDS_ON", "Technology"),
    ("Product", "BELONGS_TO", "Organization"),
    ("Product", "COMPETES_WITH", "Product"),
    ("Product", "USES", "Method"),
    ("Product", "USES", "Technology"),
    ("Product", "USES", "Model"),
    ("Product", "USES", "Concept"),
    ("Product", "CONTAINS", "Model"),
    ("Product", "CONTAINS", "Technology"),

    # --- Event ↔ * ---
    ("Event", "PROPOSES", "Technology"),
    ("Event", "PROPOSES", "Model"),
    ("Event", "PROPOSES", "Product"),
    ("Event", "CAUSES", "Concept"),
    ("Event", "BELONGS_TO", "Organization"),
    ("Event", "CONTAINS", "Document"),

    # --- Document ↔ * ---
    ("Document", "CITES", "Document"),
    ("Document", "CITES", "Model"),
    ("Document", "CITES", "Method"),
    ("Document", "PROPOSES", "Concept"),
    ("Document", "PROPOSES", "Method"),
    ("Document", "PART_OF", "Document"),
    ("Document", "BELONGS_TO", "Organization"),
    ("Document", "EVALUATES", "Model"),
    ("Document", "EVALUATES", "Method"),

    # --- Metric ↔ * ---
    ("Metric", "EVALUATES", "Model"),
    ("Metric", "EVALUATES", "Method"),
    ("Metric", "EVALUATES", "Product"),
    ("Metric", "PART_OF", "Method"),
    ("Metric", "BELONGS_TO", "Document"),

    # --- Data ↔ * ---
    ("Data", "PART_OF", "Technology"),
    ("Data", "BELONGS_TO", "Organization"),

    # --- 通配规则：通用关系 ---
    ("*", "CITES", "*"),
    ("*", "RELATED_TO", "*"),
]

# 构建 O(1) 查找集：展开通配符
_RELATION_RULES_SET: set[tuple[str, str, str]] = set()
_WILDCARD_PREDICATES: set[str] = set()

for s, p, o in RELATION_RULES:
    if s == "*" and o == "*":
        _WILDCARD_PREDICATES.add(p)
    else:
        _RELATION_RULES_SET.add((s, p, o))


def is_valid_relation(subject_type: str, predicate: str, object_type: str) -> bool:
    """校验 (subject_type, predicate, object_type) 三元组是否合法。"""
    if predicate in _WILDCARD_PREDICATES:
        return True
    if (subject_type, predicate, object_type) in _RELATION_RULES_SET:
        return True
    return False


def get_allowed_predicates(subject_type: str, object_type: str) -> list[str]:
    """返回给定主宾类型对下所有合法的谓词列表（诊断工具）。"""
    allowed = []
    for p in RELATION_PREDICATES:
        if is_valid_relation(subject_type, p, object_type):
            allowed.append(p)
    return allowed
