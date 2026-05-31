"""v10 受控图谱抽取 — 全流程测试。

覆盖:
- Phase 1: 本体 Schema 定义（类型、谓词、规则校验）
- Phase 2: 抽取引擎（Pydantic 验证器、归一化、拦截器）
- Phase 3: 实体消歧类型过滤
- Phase 4: 图谱拓扑统计脚本
"""
import pytest
import sys

sys.path.insert(0, ".")

# ======================================================================
# Phase 1: Ontology Schema
# ======================================================================

class TestOntologySchema:
    """测试本体定义的完整性与正确性。"""

    def test_entity_types_count(self):
        from backend.ontology.schema import ENTITY_TYPES
        assert len(ENTITY_TYPES) == 11

    def test_entity_types_include_legacy(self):
        from backend.ontology.schema import ENTITY_TYPE_SET
        # 向后兼容旧类型
        assert "Model" in ENTITY_TYPE_SET
        assert "Data" in ENTITY_TYPE_SET

    def test_entity_types_include_new(self):
        from backend.ontology.schema import ENTITY_TYPE_SET
        assert "Product" in ENTITY_TYPE_SET
        assert "Event" in ENTITY_TYPE_SET
        assert "Document" in ENTITY_TYPE_SET
        assert "Metric" in ENTITY_TYPE_SET

    def test_relation_predicates_count(self):
        from backend.ontology.schema import RELATION_PREDICATES
        assert len(RELATION_PREDICATES) == 12

    def test_relation_predicates_include_new(self):
        from backend.ontology.schema import RELATION_PREDICATE_SET
        assert "CAUSES" in RELATION_PREDICATE_SET
        assert "IMPLEMENTS" in RELATION_PREDICATE_SET
        assert "BELONGS_TO" in RELATION_PREDICATE_SET
        assert "COMPETES_WITH" in RELATION_PREDICATE_SET
        assert "RELATED_TO" in RELATION_PREDICATE_SET

    def test_valid_relation_tech_uses_tech(self):
        from backend.ontology.schema import is_valid_relation
        assert is_valid_relation("Technology", "USES", "Technology") is True

    def test_valid_relation_person_proposes_model(self):
        from backend.ontology.schema import is_valid_relation
        assert is_valid_relation("Person", "PROPOSES", "Model") is True

    def test_invalid_relation_person_causes_metric(self):
        from backend.ontology.schema import is_valid_relation
        assert is_valid_relation("Person", "CAUSES", "Metric") is False

    def test_valid_relation_wildcard_cites(self):
        from backend.ontology.schema import is_valid_relation
        assert is_valid_relation("Anything", "CITES", "Whatever") is True

    def test_valid_relation_wildcard_related_to(self):
        from backend.ontology.schema import is_valid_relation
        assert is_valid_relation("X", "RELATED_TO", "Y") is True

    def test_valid_relation_model_depends_tech(self):
        from backend.ontology.schema import is_valid_relation
        assert is_valid_relation("Model", "DEPENDS_ON", "Technology") is True

    def test_valid_relation_data_belongs_org(self):
        from backend.ontology.schema import is_valid_relation
        assert is_valid_relation("Data", "BELONGS_TO", "Organization") is True

    def test_invalid_relation_metric_uses_person(self):
        from backend.ontology.schema import is_valid_relation
        assert is_valid_relation("Metric", "USES", "Person") is False

    def test_get_allowed_predicates(self):
        from backend.ontology.schema import get_allowed_predicates
        preds = get_allowed_predicates("Person", "Technology")
        assert "USES" in preds
        assert "PROPOSES" in preds

    def test_get_allowed_predicates_specific_only(self):
        from backend.ontology.schema import get_allowed_predicates
        preds = get_allowed_predicates("Metric", "Person")
        # Metric -> Person 没有特定规则，但有通配 CITES 和 RELATED_TO
        specific = [p for p in preds if p not in ("CITES", "RELATED_TO")]
        assert len(specific) == 0


# ======================================================================
# Phase 2: Graph Extractor — Validators & Normalization
# ======================================================================

class TestEntityInfoValidator:
    """测试 EntityInfo 的 Pydantic 验证器和归一化。"""

    def test_valid_type_passes(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="BERT", type="Technology")
        assert e.type == "Technology"

    def test_normalize_company_to_organization(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="OpenAI", type="company")
        assert e.type == "Organization"

    def test_normalize_tool_to_technology(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="PyTorch", type="tool")
        assert e.type == "Technology"

    def test_normalize_framework_to_technology(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="LangChain", type="framework")
        assert e.type == "Technology"

    def test_normalize_algorithm_to_method(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="Leiden", type="algorithm")
        assert e.type == "Method"

    def test_normalize_paper_to_document(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="Attention Paper", type="paper")
        assert e.type == "Document"

    def test_normalize_conference_to_event(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="NeurIPS", type="conference")
        assert e.type == "Event"

    def test_normalize_metric_to_metric(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="BLEU", type="metric")
        assert e.type == "Metric"

    def test_normalize_product_to_product(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="ChatGPT", type="product")
        assert e.type == "Product"

    def test_normalize_dataset_to_data(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="ImageNet", type="dataset")
        assert e.type == "Data"

    def test_normalize_unknown_to_concept(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="Something", type="SomeRandomType")
        assert e.type == "Concept"

    def test_normalize_chinese_company(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="阿里巴巴", type="公司")
        assert e.type == "Organization"

    def test_normalize_chinese_tool(self):
        from backend.documents.graph_extractor import EntityInfo
        e = EntityInfo(name="Milvus", type="工具")
        assert e.type == "Technology"


class TestRelationInfoValidator:
    """测试 RelationInfo 的 Pydantic 验证器和归一化。"""

    def test_valid_predicate_passes(self):
        from backend.documents.graph_extractor import RelationInfo
        r = RelationInfo(subject="A", predicate="USES", object="B")
        assert r.predicate == "USES"

    def test_normalize_uses(self):
        from backend.documents.graph_extractor import RelationInfo
        r = RelationInfo(subject="A", predicate="uses", object="B")
        assert r.predicate == "USES"

    def test_normalize_utilizes(self):
        from backend.documents.graph_extractor import RelationInfo
        r = RelationInfo(subject="A", predicate="utilizes", object="B")
        assert r.predicate == "USES"

    def test_normalize_depends(self):
        from backend.documents.graph_extractor import RelationInfo
        r = RelationInfo(subject="A", predicate="depends", object="B")
        assert r.predicate == "DEPENDS_ON"

    def test_normalize_implements(self):
        from backend.documents.graph_extractor import RelationInfo
        r = RelationInfo(subject="A", predicate="builds", object="B")
        assert r.predicate == "IMPLEMENTS"

    def test_normalize_causes(self):
        from backend.documents.graph_extractor import RelationInfo
        r = RelationInfo(subject="A", predicate="leads_to", object="B")
        assert r.predicate == "CAUSES"

    def test_normalize_chinese_uses(self):
        from backend.documents.graph_extractor import RelationInfo
        r = RelationInfo(subject="A", predicate="采用", object="B")
        assert r.predicate == "USES"

    def test_normalize_chinese_depends(self):
        from backend.documents.graph_extractor import RelationInfo
        r = RelationInfo(subject="A", predicate="依赖", object="B")
        assert r.predicate == "DEPENDS_ON"

    def test_normalize_unknown_to_uses(self):
        from backend.documents.graph_extractor import RelationInfo
        r = RelationInfo(subject="A", predicate="something_random", object="B")
        assert r.predicate == "USES"


class TestValidateExtraction:
    """测试 _validate_extraction 后置拦截器。"""

    def test_valid_extraction_passes(self):
        from backend.documents.graph_extractor import (
            EntityInfo, RelationInfo, ExtractionResult, _validate_extraction
        )
        result = ExtractionResult(
            entities=[
                EntityInfo(name="PyTorch", type="Technology"),
                EntityInfo(name="AutoDiff", type="Concept"),
            ],
            relations=[
                RelationInfo(subject="PyTorch", predicate="IMPLEMENTS", object="AutoDiff"),
            ]
        )
        validated = _validate_extraction(result)
        assert len(validated.entities) == 2
        assert len(validated.relations) == 1

    def test_invalid_entity_type_filtered(self):
        from backend.documents.graph_extractor import (
            EntityInfo, RelationInfo, ExtractionResult, _validate_extraction
        )
        # _validate_extraction checks ENTITY_TYPE_SET AFTER field_validator
        # field_validator normalizes "tool" -> "Technology", so it passes
        # To test filtering, we need to bypass the validator
        result = ExtractionResult(
            entities=[
                EntityInfo(name="PyTorch", type="Technology"),
            ],
            relations=[]
        )
        # Manually set an invalid type to test the filter
        result.entities[0].type = "InvalidType"
        validated = _validate_extraction(result)
        assert len(validated.entities) == 0

    def test_relation_with_missing_subject_filtered(self):
        from backend.documents.graph_extractor import (
            EntityInfo, RelationInfo, ExtractionResult, _validate_extraction
        )
        result = ExtractionResult(
            entities=[
                EntityInfo(name="PyTorch", type="Technology"),
            ],
            relations=[
                RelationInfo(subject="MissingEntity", predicate="USES", object="PyTorch"),
            ]
        )
        validated = _validate_extraction(result)
        assert len(validated.relations) == 0

    def test_relation_with_missing_object_filtered(self):
        from backend.documents.graph_extractor import (
            EntityInfo, RelationInfo, ExtractionResult, _validate_extraction
        )
        result = ExtractionResult(
            entities=[
                EntityInfo(name="PyTorch", type="Technology"),
            ],
            relations=[
                RelationInfo(subject="PyTorch", predicate="USES", object="MissingEntity"),
            ]
        )
        validated = _validate_extraction(result)
        assert len(validated.relations) == 0

    def test_invalid_relation_direction_filtered(self):
        from backend.documents.graph_extractor import (
            EntityInfo, RelationInfo, ExtractionResult, _validate_extraction
        )
        result = ExtractionResult(
            entities=[
                EntityInfo(name="Alice", type="Person"),
                EntityInfo(name="BLEU", type="Metric"),
            ],
            relations=[
                RelationInfo(subject="Alice", predicate="CAUSES", object="BLEU"),
            ]
        )
        validated = _validate_extraction(result)
        assert len(validated.relations) == 0

    def test_tech_uses_concept_passes(self):
        from backend.documents.graph_extractor import (
            EntityInfo, RelationInfo, ExtractionResult, _validate_extraction
        )
        result = ExtractionResult(
            entities=[
                EntityInfo(name="React", type="Technology"),
                EntityInfo(name="VirtualDOM", type="Concept"),
            ],
            relations=[
                RelationInfo(subject="React", predicate="USES", object="VirtualDOM"),
            ]
        )
        validated = _validate_extraction(result)
        assert len(validated.relations) == 1

    def test_mixed_valid_and_invalid(self):
        from backend.documents.graph_extractor import (
            EntityInfo, RelationInfo, ExtractionResult, _validate_extraction
        )
        result = ExtractionResult(
            entities=[
                EntityInfo(name="PyTorch", type="Technology"),
                EntityInfo(name="AutoDiff", type="Concept"),
                EntityInfo(name="BERT", type="Model"),
            ],
            relations=[
                # Valid: Technology IMPLEMENTS Concept
                RelationInfo(subject="PyTorch", predicate="IMPLEMENTS", object="AutoDiff"),
                # Invalid: Model CAUSES Concept (not in rules)
                RelationInfo(subject="BERT", predicate="CAUSES", object="AutoDiff"),
                # Valid: Model DEPENDS_ON Technology
                RelationInfo(subject="BERT", predicate="DEPENDS_ON", object="PyTorch"),
            ]
        )
        validated = _validate_extraction(result)
        assert len(validated.entities) == 3
        assert len(validated.relations) == 2


# ======================================================================
# Phase 3: Entity Resolution Type Filter
# ======================================================================

class TestEntityResolutionTypeFilter:
    """测试实体消歧的类型过滤 Cypher。"""

    def test_cypher_contains_type_filter(self):
        with open("backend/graph/entity_resolution.py", encoding="utf-8") as f:
            content = f.read()
        assert "a.type = b.type" in content


# ======================================================================
# Phase 4: Graph Topology Stats
# ======================================================================

class TestGraphTopologyStats:
    """测试图谱拓扑统计脚本的结构。"""

    def test_import(self):
        from scripts.graph_topology_stats import collect_topology_stats
        assert callable(collect_topology_stats)

    def test_script_has_main(self):
        import ast
        with open("scripts/graph_topology_stats.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert "collect_topology_stats" in funcs
        assert "main" in funcs


class TestRunEvaluationModes:
    """测试 run_evaluation.py 的新模式。"""

    def test_has_graph_mode(self):
        import ast
        with open("scripts/run_evaluation.py", encoding="utf-8") as f:
            source = f.read()
        assert '"graph"' in source
        assert '"graph_compare"' in source

    def test_has_compare_topology_func(self):
        import ast
        with open("scripts/run_evaluation.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert "_compare_topology" in funcs


class TestGenerateReportExtensions:
    """测试 generate_report.py 的拓扑图表函数。"""

    def test_has_topology_chart_func(self):
        import ast
        with open("scripts/generate_report.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert "_generate_topology_chart" in funcs
        assert "_generate_topology_comparison_chart" in funcs
        assert "_topology_section" in funcs
        assert "_topology_compare_section" in funcs


# ======================================================================
# Integration: EXTRACTION_PROMPT Content
# ======================================================================

class TestExtractionPrompt:
    """测试受控抽取 Prompt 的内容约束。"""

    def test_prompt_lists_entity_types(self):
        from backend.documents.graph_extractor import EXTRACTION_PROMPT
        for t in ["Person", "Organization", "Technology", "Concept", "Product", "Event"]:
            assert t in EXTRACTION_PROMPT, f"Missing entity type: {t}"

    def test_prompt_lists_predicates(self):
        from backend.documents.graph_extractor import EXTRACTION_PROMPT
        for p in ["USES", "DEPENDS_ON", "CONTAINS", "IMPLEMENTS", "CAUSES", "BELONGS_TO"]:
            assert p in EXTRACTION_PROMPT, f"Missing predicate: {p}"

    def test_prompt_forbids_inventing_types(self):
        from backend.documents.graph_extractor import EXTRACTION_PROMPT
        assert "不得发明" in EXTRACTION_PROMPT or "不能" in EXTRACTION_PROMPT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
