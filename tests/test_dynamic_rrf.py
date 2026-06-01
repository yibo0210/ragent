"""动态 RRF 权重矩阵单元测试。"""
import pytest
from backend.rag.dynamic_rrf import load_weight_matrix, get_weights_for_intent


class TestWeightMatrix:
    def test_load_matrix(self):
        matrix = load_weight_matrix()
        assert "L1_FACTUAL" in matrix
        assert "L2_REASONING" in matrix
        assert "L3_MACRO_SUMMARY" in matrix

    def test_l1_weights_dense_heavy(self):
        weights = get_weights_for_intent("L1_FACTUAL")
        assert weights[0] > weights[2]  # dense > graph
        assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_l2_weights_graph_heavy(self):
        weights = get_weights_for_intent("L2_REASONING")
        assert weights[2] > weights[0]  # graph > dense

    def test_l3_weights_balanced(self):
        weights = get_weights_for_intent("L3_MACRO_SUMMARY")
        assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_unknown_intent_returns_default(self):
        weights = get_weights_for_intent("UNKNOWN")
        assert len(weights) == 4
        assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_weights_tuple_length(self):
        for level in ["L1_FACTUAL", "L2_REASONING", "L3_MACRO_SUMMARY"]:
            weights = get_weights_for_intent(level)
            assert len(weights) == 4


class TestDynamicRRFIntegration:
    def test_retrieve_documents_accepts_intent(self):
        """retrieve_documents 应接受 intent_level 参数。"""
        from backend.rag.utils import retrieve_documents
        # 这个测试只验证函数签名正确，不实际调用（需要 Milvus）
        import inspect
        sig = inspect.signature(retrieve_documents)
        assert "intent_level" in sig.parameters

    def test_run_rag_graph_accepts_intent(self):
        """run_rag_graph 应接受 intent_level 参数。"""
        from backend.rag.pipeline import run_rag_graph
        import inspect
        sig = inspect.signature(run_rag_graph)
        assert "intent_level" in sig.parameters
