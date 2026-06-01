"""v12 Query Profiler — 轻量级意图分类器测试。

覆盖:
- L1 闲聊/简单事实查询分类
- L2 多跳推理查询分类
- L3 宏观全局总结查询分类
- 边界条件（空查询、短查询）
- QueryIntent 数据类序列化
"""
import pytest
import sys

sys.path.insert(0, ".")

from backend.agent.query_profiler import QueryProfiler, QueryIntent


class TestQueryProfiler:
    """测试 QueryProfiler 的意图分类能力。"""

    def setup_method(self):
        """每个测试前创建纯关键词模式的 profiler（不依赖 Embedding 服务）。"""
        self.profiler = QueryProfiler(use_embedding=False)

    def test_l1_factual_greeting(self):
        """问候语应被分类为 L1_FACTUAL。"""
        intent = self.profiler.profile("你好")
        assert intent.level == "L1_FACTUAL"
        assert "你好" in intent.matched_keywords

    def test_l1_factual_simple_question(self):
        """简单定义型问题应被分类为 L1_FACTUAL。"""
        intent = self.profiler.profile("Python 是什么？")
        assert intent.level == "L1_FACTUAL"
        assert "是什么" in intent.matched_keywords

    def test_l2_reasoning_relation(self):
        """涉及关系分析的问题应被分类为 L2_REASONING。"""
        intent = self.profiler.profile("Milvus 和 Neo4j 之间有什么关系？")
        assert intent.level == "L2_REASONING"
        assert "关系" in intent.matched_keywords

    def test_l2_reasoning_multi_hop(self):
        """多跳推理问题应被分类为 L2_REASONING。"""
        intent = self.profiler.profile("GraphRAG 依赖哪些组件来实现多跳推理？")
        assert intent.level == "L2_REASONING"
        assert "多跳" in intent.matched_keywords or "推理" in intent.matched_keywords

    def test_l3_macro_summary(self):
        """宏观全局总结问题应被分类为 L3_MACRO_SUMMARY。"""
        intent = self.profiler.profile("系统整体技术架构是怎样的？请全面总结。")
        assert intent.level == "L3_MACRO_SUMMARY"
        # 应匹配到 "全面" 和 "总结"
        assert "全面" in intent.matched_keywords or "总结" in intent.matched_keywords

    def test_l3_macro_compare(self):
        """全局对比问题应被分类为 L3_MACRO_SUMMARY。"""
        intent = self.profiler.profile("所有文档中的方法有什么区别？")
        assert intent.level == "L3_MACRO_SUMMARY"
        assert "所有" in intent.matched_keywords or "区别" in intent.matched_keywords

    def test_intent_to_dict(self):
        """to_dict() 应包含所有必要字段。"""
        intent = self.profiler.profile("你好")
        d = intent.to_dict()
        assert "level" in d
        assert "complexity_score" in d
        assert "matched_keywords" in d
        assert "embedding_similarity" in d
        assert "reason" in d
        assert isinstance(d["matched_keywords"], list)
        assert isinstance(d["embedding_similarity"], dict)

    def test_empty_query_defaults_l1(self):
        """空查询应默认为 L1_FACTUAL。"""
        intent = self.profiler.profile("")
        assert intent.level == "L1_FACTUAL"
        assert intent.complexity_score == 0.0

    def test_short_query_forces_l1(self):
        """短查询（< 5 字符）应强制为 L1_FACTUAL。"""
        intent = self.profiler.profile("你好")
        assert intent.level == "L1_FACTUAL"

    def test_complexity_score_range(self):
        """复杂度分数应在 0.0~1.0 范围内。"""
        queries = [
            "你好",
            "Milvus 和 Neo4j 之间有什么关系？",
            "系统整体技术架构是怎样的？请全面总结。",
            "GraphRAG 依赖哪些组件来实现多跳推理？",
        ]
        for q in queries:
            intent = self.profiler.profile(q)
            assert 0.0 <= intent.complexity_score <= 1.0

    def test_l2_higher_complexity_than_l1(self):
        """L2 查询的复杂度应高于 L1。"""
        l1 = self.profiler.profile("你好")
        l2 = self.profiler.profile("Milvus 和 Neo4j 之间有什么关系？")
        assert l2.complexity_score > l1.complexity_score

    def test_l3_higher_complexity_than_l2(self):
        """L3 查询的复杂度应高于 L2（或至少不低于）。"""
        l2 = self.profiler.profile("Milvus 和 Neo4j 之间有什么关系？")
        l3 = self.profiler.profile("系统整体技术架构是怎样的？请全面总结。")
        assert l3.complexity_score >= l2.complexity_score


class TestProfilerIntegration:
    """集成测试：验证 Query Profiler 与 Supervisor 路由的集成。"""

    def test_profiler_result_in_supervisor_return(self):
        """验证 supervisor_node 返回值包含 query_intent。"""
        from unittest.mock import patch, MagicMock
        from backend.agent.orchestrator import supervisor_node
        from langchain_core.messages import HumanMessage

        # Mock LLM 返回一个合法的路由 JSON
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"routes": ["direct_answer"], "reason": "测试"}'
        mock_model.invoke.return_value = mock_response

        state = {
            "messages": [HumanMessage(content="你好，这是一个测试问题")],
            "user_query": "你好，这是一个测试问题",
            "next_worker": "",
            "next_workers": [],
            "route_reason": "",
            "rag_trace": None,
            "web_search_trace": None,
            "agent_trace": None,
            "worker_outputs": {},
            "human_interfered_input": "",
            "query_plan": None,
            "critique_result": None,
            "retry_count": 0,
            "draft_answer": "",
            "is_hallucinated": False,
            "plan_steps_completed": [],
            "tool_outputs": {},
            "query_intent": None,
        }

        with patch("backend.agent.orchestrator._get_supervisor_model", return_value=mock_model):
            result = supervisor_node(state)

        assert "query_intent" in result
        intent = result["query_intent"]
        assert "level" in intent
        assert intent["level"] in ("L1_FACTUAL", "L2_REASONING", "L3_MACRO_SUMMARY")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
