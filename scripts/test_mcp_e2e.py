#!/usr/bin/env python3
"""MCP 端到端测试

测试流程:
1. 连接到测试 MCP Server (stdio)
2. 拉取工具列表
3. 调用 query_employees 工具
4. 调用 execute_sql 工具
5. 测试工具转换 (mcp_tools_to_langchain_tools)
6. 测试 Data Analyst MCP 查询
"""
import asyncio
import sys
import json
sys.path.insert(0, ".")


async def test_mcp_connection():
    """测试 MCP 连接和工具调用。"""
    from backend.agent.mcp_client import MCPConnectionManager

    manager = MCPConnectionManager()

    print("=" * 60)
    print("1. 连接到测试 MCP Server (stdio)")
    print("=" * 60)

    success = await manager.connect(
        server_name="test_sqlite",
        url="python",
        transport="stdio",
        args=["scripts/test_mcp_server.py"],
    )
    print(f"   连接结果: {'成功' if success else '失败'}")
    assert success, "连接失败"

    print("\n" + "=" * 60)
    print("2. 拉取工具列表")
    print("=" * 60)

    tools = manager.get_available_tools("test_sqlite")
    print(f"   工具数量: {len(tools)}")
    for t in tools:
        print(f"   - {t['name']}: {t['description'][:50]}...")
    assert len(tools) == 3, f"期望 3 个工具，实际 {len(tools)}"

    print("\n" + "=" * 60)
    print("3. 调用 query_employees 工具")
    print("=" * 60)

    result = await manager.call_tool("test_sqlite", "query_employees", {"department": "engineering"})
    print(f"   结果: {result.get('content', '')[:200]}...")
    assert not result.get("error"), f"调用失败: {result.get('error')}"
    data = json.loads(result["content"])
    print(f"   返回 {len(data)} 条记录")
    assert len(data) > 0, "期望有数据"

    print("\n" + "=" * 60)
    print("4. 调用 execute_sql 工具")
    print("=" * 60)

    result = await manager.call_tool("test_sqlite", "execute_sql", {"sql": "SELECT department, COUNT(*) as cnt, AVG(salary) as avg_salary FROM employees GROUP BY department"})
    print(f"   结果:\n{result.get('content', '')[:300]}")
    assert not result.get("error"), f"调用失败: {result.get('error')}"

    print("\n" + "=" * 60)
    print("5. 测试工具转换 (mcp_tools_to_langchain_tools)")
    print("=" * 60)

    from backend.agent.tools import mcp_tools_to_langchain_tools
    langchain_tools = mcp_tools_to_langchain_tools(tools, "test_sqlite")
    print(f"   转换后工具数量: {len(langchain_tools)}")
    for t in langchain_tools:
        print(f"   - {t.name}: {t.description[:50]}...")
    assert len(langchain_tools) == 3, f"期望 3 个工具，实际 {len(langchain_tools)}"

    print("\n" + "=" * 60)
    print("6. 测试 Data Analyst MCP 查询")
    print("=" * 60)

    from backend.agent.data_analyst import get_mcp_data_sources, format_mcp_result
    sources = await get_mcp_data_sources()
    print(f"   发现 MCP 数据源: {len(sources)} 个")
    for s in sources:
        print(f"   - {s['server_name']}/{s['tool_name']}: {s['description'][:40]}...")

    print("\n" + "=" * 60)
    print("7. 测试 MCP 结果格式化")
    print("=" * 60)

    formatted = format_mcp_result(result)
    print(f"   格式化结果:\n{formatted[:200]}...")

    # 清理
    await manager.disconnect_all()

    print("\n" + "=" * 60)
    print("全部测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_mcp_connection())
