"""测试用 MCP Server (stdio 模式)

提供 3 个工具:
- query_employees: 查询员工表
- query_departments: 查询部门表
- execute_sql: 执行任意 SELECT SQL
"""
import sqlite3
import json
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

DB_PATH = "data/test_mcp.db"

server = Server("test-sqlite")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="query_employees",
            description="查询员工信息表，支持按部门、薪资范围过滤",
            inputSchema={
                "type": "object",
                "properties": {
                    "department": {"type": "string", "description": "部门名称过滤 (engineering/marketing/sales/hr)"},
                    "min_salary": {"type": "number", "description": "最低薪资"},
                    "max_salary": {"type": "number", "description": "最高薪资"},
                    "limit": {"type": "integer", "description": "返回条数，默认 20"},
                },
            },
        ),
        Tool(
            name="query_departments",
            description="查询部门信息表，包含预算和经理信息",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="execute_sql",
            description="执行任意 SELECT SQL 查询",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "要执行的 SELECT SQL 语句"},
                },
                "required": ["sql"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if name == "query_employees":
            sql = "SELECT * FROM employees WHERE 1=1"
            if arguments.get("department"):
                sql += f" AND department = '{arguments['department']}'"
            if arguments.get("min_salary"):
                sql += f" AND salary >= {arguments['min_salary']}"
            if arguments.get("max_salary"):
                sql += f" AND salary <= {arguments['max_salary']}"
            sql += f" LIMIT {arguments.get('limit', 20)}"

        elif name == "query_departments":
            sql = "SELECT * FROM departments"

        elif name == "execute_sql":
            sql = arguments.get("sql", "")
            if not sql.strip().upper().startswith("SELECT"):
                return [TextContent(type="text", text="错误: 仅允许 SELECT 查询")]
        else:
            return [TextContent(type="text", text=f"未知工具: {name}")]

        rows = conn.execute(sql).fetchall()
        result = [dict(row) for row in rows]
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"查询错误: {str(e)}")]
    finally:
        conn.close()


async def main():
    print("Starting test MCP Server (stdio mode)", flush=True)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
