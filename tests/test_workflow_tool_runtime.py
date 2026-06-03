"""Tests for WorkflowTool and ToolRegistry."""

import pytest
from backend.workflow.tool_runtime import WorkflowTool, ToolResult, ToolRegistry


class TestToolResult:
    def test_success_result(self):
        result = ToolResult(success=True, data={"answer": "hello"})
        assert result.success is True
        assert result.data["answer"] == "hello"
        assert result.error == ""
        assert result.tokens_used == 0

    def test_error_result(self):
        result = ToolResult(success=False, error="timeout")
        assert result.success is False
        assert result.error == "timeout"

    def test_to_dict(self):
        result = ToolResult(success=True, data={"x": 1}, tokens_used=100)
        d = result.to_dict()
        assert d == {"success": True, "data": {"x": 1}, "error": "", "tokens_used": 100}


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = WorkflowTool(name="test", description="test tool", invoke_fn=lambda **kw: {"ok": True})
        registry.register(tool)
        assert registry.get("test") is tool
        assert registry.get("nonexistent") is None

    def test_list_tools(self):
        registry = ToolRegistry()
        tool = WorkflowTool(name="agent1", description="d1", invoke_fn=lambda **kw: {}, tool_type="agent")
        registry.register(tool)
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "agent1"
        assert tools[0]["type"] == "agent"


class TestWorkflowTool:
    def test_invoke_with_dict_return(self):
        async def _test():
            def sync_fn(query, user_context=None, **kwargs):
                return {"response": f"answered: {query}"}
            tool = WorkflowTool(name="test", description="d", invoke_fn=sync_fn)
            result = await tool.invoke("hello")
            assert result.success is True
            assert "answered: hello" in result.data["response"]
        asyncio = pytest.importorskip("asyncio")
        asyncio.run(_test())

    def test_invoke_with_tool_result_return(self):
        async def _test():
            def sync_fn(query, **kwargs):
                return ToolResult(success=True, data={"custom": "value"})
            tool = WorkflowTool(name="test", description="d", invoke_fn=sync_fn)
            result = await tool.invoke("hello")
            assert result.success is True
            assert result.data["custom"] == "value"
        import asyncio
        asyncio.run(_test())

    def test_invoke_exception_handling(self):
        async def _test():
            def sync_fn(query, **kwargs):
                raise RuntimeError("boom")
            tool = WorkflowTool(name="test", description="d", invoke_fn=sync_fn)
            result = await tool.invoke("hello")
            assert result.success is False
            assert "boom" in result.error
        import asyncio
        asyncio.run(_test())

    def test_from_agent(self):
        tool = WorkflowTool.from_agent("rag_specialist", lambda **kw: {})
        assert tool.name == "rag_specialist"
        assert tool.tool_type == "agent"

    def test_from_mcp(self):
        tool = WorkflowTool.from_mcp("my_mcp", server_name="srv", tool_name="query")
        assert tool.name == "my_mcp"
        assert tool.tool_type == "mcp"
        assert tool.server_name == "srv"
        assert tool.tool_name == "query"
