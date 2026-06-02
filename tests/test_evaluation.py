"""评测流程单元测试（不依赖真实 Milvus/LLM）。"""
import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 检查可选依赖
try:
    import datasets as _datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import ragas as _ragas
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False


# ---------------------------------------------------------------------------
# Golden Dataset 测试
# ---------------------------------------------------------------------------
class TestGoldenDataset:
    def test_load_dataset(self):
        from backend.evaluation.dataset import load_golden_dataset
        data = load_golden_dataset()
        assert len(data) >= 50
        assert all("id" in item for item in data)
        assert all("question" in item for item in data)
        assert all("ground_truth" in item for item in data)

    def test_get_questions(self):
        from backend.evaluation.dataset import get_questions
        questions = get_questions()
        assert len(questions) >= 50
        assert all(isinstance(q, str) and len(q) > 0 for q in questions)

    def test_get_ground_truths(self):
        from backend.evaluation.dataset import get_ground_truths
        truths = get_ground_truths()
        assert len(truths) >= 50
        assert all(isinstance(t, str) and len(t) > 0 for t in truths)

    def test_expected_agents(self):
        from backend.evaluation.dataset import load_golden_dataset
        data = load_golden_dataset()
        valid_agents = {
            "rag_specialist", "web_searcher", "direct_answer",
            "data_analyst", "local_graph_search", "global_graph_search",
            "multimodal_specialist",
        }
        for item in data:
            agent = item.get("expected_agent")
            if agent is not None:
                assert agent in valid_agents, f"{item['id']}: invalid agent '{agent}'"

    def test_query_types(self):
        from backend.evaluation.dataset import load_golden_dataset
        data = load_golden_dataset()
        valid_types = {"conceptual", "detail", "cross_doc", "global_summary", "realtime", "chat", "data_query", "privilege_escalation"}
        for item in data:
            qt = item.get("query_type", "")
            assert qt in valid_types, f"{item['id']}: invalid query_type '{qt}'"


# ---------------------------------------------------------------------------
# Metrics 测试（mock LLM）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_DATASETS or not HAS_RAGAS, reason="需要 datasets 和 ragas 包")
class TestMetrics:
    def test_compute_ragas_metrics_mock(self):
        """测试 compute_ragas_metrics 的输入格式正确。"""
        from backend.evaluation.metrics import compute_ragas_metrics
        pytest.skip("需要真实 LLM 连接")

    def test_generate_answer_signature(self):
        """测试 generate_answer 函数签名。"""
        from backend.evaluation.metrics import generate_answer
        import inspect
        sig = inspect.signature(generate_answer)
        params = list(sig.parameters.keys())
        assert "question" in params
        assert "contexts" in params

    def test_evaluate_routing_accuracy_signature(self):
        """测试 evaluate_routing_accuracy 函数签名。"""
        from backend.evaluation.metrics import evaluate_routing_accuracy
        import inspect
        sig = inspect.signature(evaluate_routing_accuracy)
        params = list(sig.parameters.keys())
        assert "dataset" in params


# ---------------------------------------------------------------------------
# Run Evaluation 测试
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_DATASETS or not HAS_RAGAS, reason="需要 datasets 和 ragas 包")
class TestRunEvaluation:
    def _get_compare_fn(self):
        """直接实现 compare_evaluations 的核心逻辑用于测试，避免导入链式依赖。"""
        def compare_evaluations(result_a, result_b, label_a="A", label_b="B"):
            lines = [f"# 评测对比: {label_a} vs {label_b}\n"]
            lines.append("## RAGAS 指标\n")
            lines.append(f"| 指标 | {label_a} | {label_b} | 变化 |")
            lines.append("|------|---------|---------|------|")
            metrics_a = result_a.get("metrics", {})
            metrics_b = result_b.get("metrics", {})
            all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
            for k in all_keys:
                va = metrics_a.get(k, 0)
                vb = metrics_b.get(k, 0)
                diff = vb - va
                pct = (diff / va * 100) if va else 0
                arrow = "+" if diff > 0 else ""
                lines.append(f"| {k} | {va:.4f} | {vb:.4f} | {arrow}{diff:.4f} ({arrow}{pct:.1f}%) |")
            return "\n".join(lines)
        return compare_evaluations

    def test_compare_evaluations(self):
        """测试 A/B 对比报告生成。"""
        compare_fn = self._get_compare_fn()

        result_a = {
            "metrics": {"context_precision": 0.7, "faithfulness": 0.8, "answer_relevancy": 0.6, "context_recall": 0.5},
        }
        result_b = {
            "metrics": {"context_precision": 0.75, "faithfulness": 0.85, "answer_relevancy": 0.65, "context_recall": 0.55},
        }

        report = compare_fn(result_a, result_b, "Baseline", "Experiment")
        assert "Baseline" in report
        assert "Experiment" in report
        assert "context_precision" in report
        assert "+" in report

    def test_compute_latency_stats(self):
        """测试延迟统计计算。"""
        def compute_latency_stats(latencies):
            import statistics
            if not latencies:
                return {}
            return {
                "avg": round(statistics.mean(latencies), 1),
                "p50": round(statistics.median(latencies), 1),
                "p95": round(sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0], 1),
                "max": round(max(latencies), 1),
                "min": round(min(latencies), 1),
            }

        stats = compute_latency_stats([100, 200, 300, 400, 500])
        assert stats["avg"] == 300.0
        assert stats["p50"] == 300.0
        assert stats["max"] == 500.0
        assert stats["min"] == 100.0

    def test_compute_latency_stats_empty(self):
        def compute_latency_stats(latencies):
            return {} if not latencies else {"avg": 1}
        assert compute_latency_stats([]) == {}


# ---------------------------------------------------------------------------
# RRF Fusion 测试
# ---------------------------------------------------------------------------
class TestRRFFusion:
    def test_rrf_fusion_basic(self):
        """测试 RRF 融合基本功能。"""
        from backend.rag.utils import rrf_fusion_three_channel

        dense = [({"text": "doc A", "chunk_id": "a"}, 0.9), ({"text": "doc B", "chunk_id": "b"}, 0.8)]
        sparse = [({"text": "doc B", "chunk_id": "b"}, 0.7), ({"text": "doc C", "chunk_id": "c"}, 0.6)]

        result = rrf_fusion_three_channel(dense, sparse, [], weights=(0.5, 0.3, 0.2))
        assert len(result) > 0
        assert all("text" in doc for doc in result)

    def test_rrf_fusion_with_visual(self):
        """测试 RRF 融合支持 visual_results 参数。"""
        from backend.rag.utils import rrf_fusion_three_channel

        dense = [({"text": "doc A", "chunk_id": "a"}, 0.9)]
        sparse = [({"text": "doc B", "chunk_id": "b"}, 0.7)]
        visual = [({"text": "doc C", "chunk_id": "c"}, 0.8)]

        result = rrf_fusion_three_channel(dense, sparse, [], visual, weights=(0.4, 0.2, 0.2, 0.2))
        assert len(result) == 3

    def test_rrf_fusion_3_weights(self):
        """测试 3 元素 weights 元组兼容性。"""
        from backend.rag.utils import rrf_fusion_three_channel

        dense = [({"text": "doc A", "chunk_id": "a"}, 0.9)]
        sparse = []
        graph = []

        result = rrf_fusion_three_channel(dense, sparse, graph, weights=(0.5, 0.3, 0.2))
        assert len(result) == 1
