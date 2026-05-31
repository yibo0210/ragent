"""图谱三元组抽取器。利用 LLM 结构化输出从 L2 文本块提取实体关系。

v10: 引入本体约束层，将自由抽取收敛为受控填空。
- EntityInfo.type / RelationInfo.predicate 通过 field_validator 白名单校验 + 归一化
- _validate_extraction() 后置拦截器过滤违规实体/关系
- EXTRACTION_PROMPT 明确列出合法类型和谓词
"""
from typing import List
from pydantic import BaseModel, Field, field_validator
from langchain_core.messages import SystemMessage, HumanMessage

from backend.ontology.schema import (
    ENTITY_TYPES, ENTITY_TYPE_SET,
    RELATION_PREDICATES, RELATION_PREDICATE_SET,
    is_valid_relation,
)


# ---------------------------------------------------------------------------
# 归一化映射表
# ---------------------------------------------------------------------------

_ENTITY_TYPE_MAP: dict[str, str] = {
    # 英文别名
    "company": "Organization", "firm": "Organization", "enterprise": "Organization",
    "university": "Organization", "institution": "Organization", "startup": "Organization",
    "person": "Person", "researcher": "Person", "author": "Person", "engineer": "Person",
    "scientist": "Person", "founder": "Person",
    "tool": "Technology", "framework": "Technology", "library": "Technology",
    "platform": "Technology", "language": "Technology", "software": "Technology",
    "system": "Technology", "api": "Technology", "sdk": "Technology",
    "idea": "Concept", "theory": "Concept", "approach": "Concept", "notion": "Concept",
    "paradigm": "Concept", "architecture": "Concept", "pattern": "Concept",
    "algorithm": "Method", "technique": "Method", "strategy": "Method",
    "procedure": "Method", "pipeline": "Method",
    "product": "Product", "service": "Product", "saas": "Product", "app": "Product",
    "dataset": "Data", "corpus": "Data", "benchmark": "Data", "corpora": "Data",
    "paper": "Document", "article": "Document", "specification": "Document",
    "standard": "Document", "report": "Document", "spec": "Document",
    "conference": "Event", "workshop": "Event", "summit": "Event", "release": "Event",
    "metric": "Metric", "indicator": "Metric", "kpi": "Metric", "measure": "Metric",
    # 中文别名
    "公司": "Organization", "企业": "Organization", "机构": "Organization", "大学": "Organization",
    "人物": "Person", "研究员": "Person", "工程师": "Person", "科学家": "Person",
    "工具": "Technology", "框架": "Technology", "平台": "Technology", "语言": "Technology",
    "概念": "Concept", "理论": "Concept", "思想": "Concept", "模式": "Concept",
    "算法": "Method", "方法": "Method", "技术": "Method", "策略": "Method",
    "产品": "Product", "服务": "Product", "应用": "Product",
    "数据集": "Data", "语料库": "Data", "基准": "Data",
    "论文": "Document", "文档": "Document", "规范": "Document", "标准": "Document",
    "会议": "Event", "发布会": "Event", "竞赛": "Event",
    "指标": "Metric", "评测": "Metric",
}

_PREDICATE_MAP: dict[str, str] = {
    # 英文
    "uses": "USES", "utilizes": "USES", "employs": "USES", "adopts": "USES",
    "leverages": "USES", "applies": "USES",
    "depends": "DEPENDS_ON", "relies_on": "DEPENDS_ON", "requires": "DEPENDS_ON",
    "includes": "CONTAINS", "contains": "CONTAINS", "has": "CONTAINS",
    "encompasses": "CONTAINS", "comprises": "CONTAINS",
    "belongs": "BELONGS_TO", "member_of": "BELONGS_TO", "part_of": "PART_OF",
    "implements": "IMPLEMENTS", "builds": "IMPLEMENTS", "realizes": "IMPLEMENTS",
    "causes": "CAUSES", "leads_to": "CAUSES", "results_in": "CAUSES",
    "triggers": "CAUSES",
    "competes": "COMPETES_WITH", "rivals": "COMPETES_WITH",
    "proposes": "PROPOSES", "introduces": "PROPOSES", "publishes": "PROPOSES",
    "releases": "PROPOSES", "announces": "PROPOSES",
    "evaluates": "EVALUATES", "measures": "EVALUATES", "assesses": "EVALUATES",
    "benchmarks": "EVALUATES",
    "cites": "CITES", "references": "CITES", "mentions": "CITES",
    # 中文
    "使用": "USES", "采用": "USES", "利用": "USES", "应用": "USES",
    "依赖": "DEPENDS_ON", "依靠": "DEPENDS_ON",
    "包含": "CONTAINS", "含有": "CONTAINS", "包括": "CONTAINS",
    "属于": "BELONGS_TO", "隶属": "BELONGS_TO",
    "实现": "IMPLEMENTS", "构建": "IMPLEMENTS",
    "导致": "CAUSES", "引起": "CAUSES", "造成": "CAUSES",
    "竞争": "COMPETES_WITH", "对标": "COMPETES_WITH",
    "提出": "PROPOSES", "发布": "PROPOSES", "推出": "PROPOSES",
    "评估": "EVALUATES", "评测": "EVALUATES", "衡量": "EVALUATES",
    "引用": "CITES", "参考": "CITES",
}


def _normalize_entity_type(raw: str) -> str:
    """将 LLM 输出的实体类型归一化到本体白名单，兜底 Concept。"""
    key = raw.strip().lower()
    if key in ENTITY_TYPE_SET:
        return raw.strip()
    return _ENTITY_TYPE_MAP.get(key, "Concept")


def _normalize_predicate(raw: str) -> str:
    """将 LLM 输出的关系谓词归一化到本体白名单，兜底 USES。"""
    key = raw.strip().lower()
    if key in RELATION_PREDICATE_SET:
        return raw.strip()
    return _PREDICATE_MAP.get(key, "USES")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class EntityInfo(BaseModel):
    name: str = Field(description="实体名")
    type: str = Field(description="实体类型，必须是以下之一: " + ", ".join(ENTITY_TYPES))
    description: str = Field(default="", description="简短描述")
    valid_from: str = Field(default="", description="知识生效起始年份，如 '2023'")
    valid_to: str = Field(default="", description="知识生效截止年份，如 '2025'，仍有效则留空")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v in ENTITY_TYPE_SET:
            return v
        return _normalize_entity_type(v)


class RelationInfo(BaseModel):
    subject: str = Field(description="源实体名")
    predicate: str = Field(description="关系，必须是以下之一: " + ", ".join(RELATION_PREDICATES))
    object: str = Field(description="目标实体名")
    description: str = Field(default="", description="关系描述")
    weight: float = Field(default=0.5, description="置信度 0-1")
    valid_from: str = Field(default="", description="关系生效起始年份")
    valid_to: str = Field(default="", description="关系生效截止年份")

    @field_validator("predicate")
    @classmethod
    def validate_predicate(cls, v: str) -> str:
        if v in RELATION_PREDICATE_SET:
            return v
        return _normalize_predicate(v)


class ExtractionResult(BaseModel):
    entities: List[EntityInfo] = Field(default_factory=list)
    relations: List[RelationInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 受控抽取 Prompt
# ---------------------------------------------------------------------------

_ENTITY_TYPE_DESC = "\n".join(f"- {t}" for t in ENTITY_TYPES)
_PREDICATE_DESC = "\n".join(f"- {p}" for p in RELATION_PREDICATES)

EXTRACTION_PROMPT = f"""你是一个知识图谱构建专家。请从以下文本中提取实体和关系。

## 实体类型（只能使用以下类型，不得发明新类型）
{_ENTITY_TYPE_DESC}

## 关系谓词（只能使用以下谓词，不得发明新谓词）
{_PREDICATE_DESC}

## 规则
1. 只提取文本中明确出现的重要名词，并归类到上述实体类型。
2. 只识别文本中明确的语义关系，并使用上述关系谓词。
3. 消歧: 如果多个名称指向同一实体，使用最规范的名称。
4. 时间: 如果文本中明确了该知识的时间范围（如"2023年发布"、"2020-2024年间使用"），填写 valid_from 和 valid_to。不明确则留空。
5. 只提取文本中明确出现的信息，不要编造。
6. 限制: 最多提取 10 个实体和 15 条关系。
7. 以 JSON 格式输出结果。"""


# ---------------------------------------------------------------------------
# 合规拦截器
# ---------------------------------------------------------------------------

def _validate_extraction(result: ExtractionResult) -> ExtractionResult:
    """后置过滤：丢弃不在白名单中的实体/关系，验证关系流向合法性。"""
    # Step 1: 过滤 type 不在白名单的实体（field_validator 已归一化，这里是安全网）
    valid_entities = {}
    dropped_entities = 0
    for e in result.entities:
        if e.type in ENTITY_TYPE_SET:
            valid_entities[e.name] = e
        else:
            print(f"[VALIDATE] Dropped entity: {e.name} (type={e.type} not in whitelist)")
            dropped_entities += 1

    # Step 2: 过滤关系
    valid_names = set(valid_entities.keys())
    valid_relations = []
    dropped_relations = 0
    for r in result.relations:
        # 检查 subject/object 是否存在于本批实体
        if r.subject not in valid_names:
            print(f"[VALIDATE] Dropped relation: {r.subject} --{r.predicate}--> {r.object} (subject not in entities)")
            dropped_relations += 1
            continue
        if r.object not in valid_names:
            print(f"[VALIDATE] Dropped relation: {r.subject} --{r.predicate}--> {r.object} (object not in entities)")
            dropped_relations += 1
            continue
        # 检查三元组合法性
        s_type = valid_entities[r.subject].type
        o_type = valid_entities[r.object].type
        if not is_valid_relation(s_type, r.predicate, o_type):
            print(f"[VALIDATE] Dropped relation: {s_type}:{r.subject} --{r.predicate}--> {o_type}:{r.object} (rule violation)")
            dropped_relations += 1
            continue
        valid_relations.append(r)

    if dropped_entities or dropped_relations:
        print(f"[VALIDATE] Filtered: {dropped_entities} entities, {dropped_relations} relations")

    return ExtractionResult(entities=list(valid_entities.values()), relations=valid_relations)


# ---------------------------------------------------------------------------
# 抽取逻辑
# ---------------------------------------------------------------------------

async def _extract_one(text: str, filename: str) -> ExtractionResult:
    """单个文本块的受控抽取。使用手动 JSON 解析兼容 Qwen。"""
    import json as _json
    import re as _re
    from backend.agent.orchestrator import _get_worker_model

    model = _get_worker_model()
    prompt = f"文件名: {filename}\n\n文本:\n{text[:1200]}"
    messages = [
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(content=prompt),
    ]
    try:
        # 手动 JSON 解析，兼容 Qwen with_structured_output 字段名不一致问题
        response = model.invoke(messages)
        raw_text = response.content if hasattr(response, "content") else str(response)

        # 提取 JSON 块
        json_match = _re.search(r'\{[\s\S]*\}', raw_text)
        if not json_match:
            print(f"[EXTRACT] No JSON found in response")
            return ExtractionResult()

        raw_data = _json.loads(json_match.group())

        # 映射字段名: Qwen 可能返回 source/target/id 而非 subject/object/name
        entities = []
        for e in raw_data.get("entities", []):
            entities.append(EntityInfo(
                name=e.get("name") or e.get("id", ""),
                type=e.get("type", "Concept"),
                description=e.get("description", ""),
                valid_from=e.get("valid_from", ""),
                valid_to=e.get("valid_to", ""),
            ))

        relations = []
        for r in raw_data.get("relations", []):
            relations.append(RelationInfo(
                subject=r.get("subject") or r.get("source", ""),
                predicate=r.get("predicate", "USES"),
                object=r.get("object") or r.get("target", ""),
                description=r.get("description", ""),
                weight=float(r.get("weight", 0.5)),
                valid_from=r.get("valid_from", ""),
                valid_to=r.get("valid_to", ""),
            ))

        result = ExtractionResult(entities=entities, relations=relations)
        # 后置合规校验
        result = _validate_extraction(result)
        print(f"[EXTRACT] OK: {len(result.entities)} entities, {len(result.relations)} relations from text[{len(text)}]")
        return result
    except Exception as e:
        import traceback
        print(f"[EXTRACT] LLM call failed: {e}")
        traceback.print_exc()
        return ExtractionResult()


async def extract_from_l2_chunks(
    l2_chunks: List[dict], filename: str, progress_callback=None
) -> ExtractionResult:
    """从一批 L2 块提取实体关系，去重合并。"""

    # Sequential extraction to avoid asyncio issues in FastAPI context
    all_results = []
    for chunk in l2_chunks:
        result = await _extract_one(chunk["text"], filename)
        all_results.append(result)
    print(f"[EXTRACT] Processed {len(all_results)} chunks, entities={sum(len(r.entities) for r in all_results)}, relations={sum(len(r.relations) for r in all_results)}")

    all_entities: dict[str, EntityInfo] = {}
    all_relations: list[RelationInfo] = []
    for r in all_results:
        if isinstance(r, Exception):
            continue
        for e in r.entities:
            if e.name not in all_entities:
                all_entities[e.name] = e
        all_relations.extend(r.relations)

    return ExtractionResult(
        entities=list(all_entities.values()),
        relations=all_relations,
    )
