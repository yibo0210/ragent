"""图谱三元组抽取器。利用 LLM 结构化输出从 L2 文本块提取实体关系。"""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage


class EntityInfo(BaseModel):
    name: str = Field(description="实体名")
    type: str = Field(description="实体类型: Person|Organization|Technology|Concept|Model|Method|Data")
    description: str = Field(default="", description="简短描述")
    valid_from: str = Field(default="", description="知识生效起始年份，如 '2023'")
    valid_to: str = Field(default="", description="知识生效截止年份，如 '2025'，仍有效则留空")


class RelationInfo(BaseModel):
    subject: str = Field(description="源实体名")
    predicate: str = Field(description="关系: DEPENDS_ON|CONTAINS|CITES|USES|PART_OF|PROPOSES|EVALUATES")
    object: str = Field(description="目标实体名")
    description: str = Field(default="", description="关系描述")
    weight: float = Field(default=0.5, description="置信度 0-1")
    valid_from: str = Field(default="", description="关系生效起始年份")
    valid_to: str = Field(default="", description="关系生效截止年份")


class ExtractionResult(BaseModel):
    entities: List[EntityInfo] = Field(default_factory=list)
    relations: List[RelationInfo] = Field(default_factory=list)


EXTRACTION_PROMPT = """你是一个知识图谱构建专家。请从以下文本中提取实体和关系。

规则:
1. 实体: 提取文本中出现的所有重要名词——包含人名、机构、技术、概念、产品、模型、算法、数据、方法。
2. 关系: 识别实体之间的语义关系。
3. 消歧: 如果多个名称指向同一实体，使用最规范的名称。
4. 时间: 如果文本中明确了该知识的时间范围（如"2023年发布"、"2020-2024年间使用"），填写 valid_from 和 valid_to。不明确则留空。
5. 只提取文本中明确出现的信息，不要编造。
6. 限制: 最多提取 10 个实体和 15 条关系。"""

async def _extract_one(text: str, filename: str) -> ExtractionResult:
    from backend.agent.orchestrator import _get_worker_model

    model = _get_worker_model()
    prompt = f"文件名: {filename}\n\n文本:\n{text[:1200]}"
    messages = [
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(content=prompt),
    ]
    try:
        structured = model.with_structured_output(ExtractionResult)
        result = structured.invoke(messages)
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
