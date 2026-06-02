"""Data Analyst Agent - Text-to-SQL Worker

将自然语言问题转为 MySQL 只读 SQL 查询并执行，以结构化数据回答用户。
v9: 支持 MCP 外部数据源（PostgreSQL、Salesforce 等）。
"""
import os
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text

from backend.storage.database import SessionLocal

load_dotenv()


# ---------------------------------------------------------------------------
# v9: MCP 数据源支持
# ---------------------------------------------------------------------------
async def get_mcp_data_sources() -> list[dict]:
    """从 MCP Manager 获取可用的数据库类工具。"""
    from .mcp_client import get_mcp_manager
    manager = get_mcp_manager()
    all_tools = manager.get_all_tools()
    db_tools = []
    for server_name, tools in all_tools.items():
        for tool_def in tools:
            name = tool_def.get("name", "")
            desc = tool_def.get("description", "")
            # 识别数据库类工具（名称或描述包含 query/sql/database/fetch）
            keywords = ["query", "sql", "database", "fetch", "select", "search", "get", "list", "find"]
            if any(kw in name.lower() or kw in desc.lower() for kw in keywords):
                db_tools.append({
                    "server_name": server_name,
                    "tool_name": name,
                    "description": desc,
                    "input_schema": tool_def.get("input_schema", {}),
                })
    return db_tools


async def generate_mcp_query(question: str, tool_schema: dict) -> dict:
    """根据 MCP 工具的 inputSchema 生成查询参数。"""
    from .orchestrator import _get_worker_model

    model = _get_worker_model()
    schema_desc = tool_schema.get("description", "")
    properties = tool_schema.get("input_schema", {}).get("properties", {})
    required = tool_schema.get("input_schema", {}).get("required", [])

    prompt = f"""你是一个数据分析师。根据用户问题和工具的输入规范，生成查询参数。

工具名称: {tool_schema.get('tool_name', '')}
工具描述: {schema_desc}
输入参数:
"""
    for param_name, param_def in properties.items():
        req = " (必填)" if param_name in required else " (可选)"
        prompt += f"  - {param_name}: {param_def.get('description', param_def.get('type', 'string'))}{req}\n"

    prompt += f"\n用户问题: {question}\n\n请输出 JSON 格式的参数，只输出 JSON，不要解释。"

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        # 提取 JSON
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            import json
            return json.loads(json_match.group())
    except Exception:
        pass
    return {}


async def execute_mcp_query(server_name: str, tool_name: str, arguments: dict) -> dict:
    """调用 MCP Server 执行查询。"""
    from .mcp_client import get_mcp_manager
    manager = get_mcp_manager()
    return await manager.call_tool(server_name, tool_name, arguments)

SQL_GENERATION_PROMPT = """你是一个数据分析师。根据用户问题和数据库 schema，生成一条 MySQL SELECT 查询。

规则：
- 只生成 SELECT 语句（只读）
- 利用 schema 中的表名和字段名
- 查询尽量简洁高效
- 只输出 SQL，不要附带解释

数据库 schema：
{schema}

用户问题：{question}"""


def get_schema_info() -> str:
    """读取数据库 schema 信息供 LLM 生成 SQL 时参考。"""
    db = SessionLocal()
    try:
        rows = db.execute(
            text(
                "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_COMMENT "
                "FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "ORDER BY TABLE_NAME, ORDINAL_POSITION"
            )
        ).fetchall()

        if not rows:
            return _fallback_schema()

        parts = []
        current_table = None
        for row in rows:
            table_name, col_name, data_type, comment = row
            if table_name != current_table:
                current_table = table_name
                parts.append(f"\n表: {table_name}")
            comment_str = f" -- {comment}" if comment else ""
            parts.append(f"  {col_name} ({data_type}){comment_str}")
        return "\n".join(parts)
    except Exception:
        return _fallback_schema()
    finally:
        db.close()


def _fallback_schema() -> str:
    return """
表: chat_sessions
  id (int)
  session_id (varchar)
  metadata_json (json)
  updated_at (datetime)
  created_at (datetime)

表: chat_messages
  id (int)
  session_ref_id (int)
  message_type (varchar)
  content (text)
  timestamp (datetime)
  rag_trace (json)
  agent_trace (json)
"""


def generate_sql(question: str, schema: str, tenant_id: int = None) -> str:
    """调用 LLM 将自然语言转换为 SQL。"""
    import traceback
    from .orchestrator import _get_worker_model

    model = _get_worker_model()
    extra_constraint = ""
    if tenant_id is not None:
        extra_constraint = f"\nIMPORTANT: Only query rows where tenant_id = {tenant_id}. Always add 'WHERE tenant_id = {tenant_id}' (or 'AND tenant_id = {tenant_id}' if there's already a WHERE clause) to every SELECT statement that queries tables with a tenant_id column."
    prompt = SQL_GENERATION_PROMPT.format(schema=schema, question=question) + extra_constraint
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
    except Exception:
        print(f"[ERROR] generate_sql LLM 调用失败: {traceback.format_exc()}")
        return "SELECT 'LLM 调用失败' AS error"

    # 提取 SQL（去除 markdown 代码块包裹）
    sql = content.strip()
    code_match = re.search(r"```(?:sql)?\s*\n?(.*?)```", sql, re.DOTALL | re.IGNORECASE)
    if code_match:
        sql = code_match.group(1).strip()
    return sql


def execute_sql(sql: str, tenant_id: int = 0, user_id: int = 0) -> dict:
    """执行只读 SQL 并返回结果。非 SELECT 语句会被拒绝。"""
    import datetime
    import decimal

    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned.upper().startswith("SELECT"):
        result = {
            "error": "non_select",
            "sql": cleaned,
            "message": "仅允许执行 SELECT 查询",
        }
        from backend.billing.audit import log_audit_event as _log
        _db = SessionLocal()
        try:
            _log(db=_db, tenant_id=tenant_id, user_id=user_id,
                 action="sql_execute", target=sql[:200],
                 result_summary=result.get("message"),
                 risk_level="high")
        finally:
            _db.close()
        return result

    # Reject multi-statement queries (e.g. "SELECT 1; DROP TABLE users;")
    if ";" in cleaned:
        result = {
            "error": "non_select",
            "sql": cleaned,
            "message": "不允许执行多条语句",
        }
        from backend.billing.audit import log_audit_event as _log
        _db = SessionLocal()
        try:
            _log(db=_db, tenant_id=tenant_id, user_id=user_id,
                 action="sql_execute", target=sql[:200],
                 result_summary=result.get("message"),
                 risk_level="high")
        finally:
            _db.close()
        return result

    # Defense-in-depth: verify tenant_id constraint is present for tenant-scoped tables
    tenant_scoped_tables = ["chat_sessions", "chat_messages", "document_index", "parent_chunks"]
    sql_lower = cleaned.lower()
    for table in tenant_scoped_tables:
        if table in sql_lower and "tenant_id" not in sql_lower:
            result = {"error": sql, "message": f"SECURITY: Query on {table} must include tenant_id filter"}
            from backend.billing.audit import log_audit_event as _log
            _db = SessionLocal()
            try:
                _log(db=_db, tenant_id=tenant_id, user_id=user_id,
                     action="sql_execute", target=sql[:200],
                     result_summary=result.get("message"),
                     risk_level="high")
            finally:
                _db.close()
            return result

    db = SessionLocal()
    try:
        db.execute(text("SET TRANSACTION READ ONLY"))
        result = db.execute(text(cleaned))
        columns = list(result.keys())
        rows = []
        for row in result.fetchall():
            serialized = {}
            for c, v in zip(columns, row):
                if isinstance(v, (datetime.datetime, datetime.date)):
                    serialized[c] = v.isoformat()
                elif isinstance(v, decimal.Decimal):
                    serialized[c] = float(v)
                elif v is None:
                    serialized[c] = None
                else:
                    serialized[c] = v
            rows.append(serialized)
        result_dict = {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "sql": cleaned,
        }
    except Exception as e:
        result_dict = {"error": "execution_failed", "sql": cleaned, "message": str(e)}
    finally:
        db.close()

    from backend.billing.audit import log_audit_event as _log
    _db = SessionLocal()
    try:
        risk_level = "high" if result_dict.get("error") else "low"
        _log(db=_db, tenant_id=tenant_id, user_id=user_id,
             action="sql_execute", target=sql[:200],
             result_summary=f"rows: {result_dict.get('row_count', 0)}" if not result_dict.get("error") else result_dict.get("error"),
             risk_level=risk_level)
    finally:
        _db.close()

    return result_dict


def format_sql_result(result: dict) -> str:
    """将 SQL 执行结果格式化为 LLM 可读文本。"""
    if result.get("error"):
        return f"SQL 执行失败: {result.get('message')}\nSQL: {result.get('sql')}"

    rows = result.get("rows", [])
    cols = result.get("columns", [])
    if not rows:
        return "查询结果为空。"

    # 表格格式化
    lines = [" | ".join(cols), "-" * 40]
    for row in rows[:20]:  # 最多展示 20 行
        lines.append(" | ".join(str(row.get(c, "")) for c in cols))
    if len(rows) > 20:
        lines.append(f"... 还有 {len(rows) - 20} 行未显示")
    return "\n".join(lines)


def format_mcp_result(result: dict) -> str:
    """将 MCP 工具返回结果格式化为 LLM 可读文本。"""
    if result.get("error"):
        return f"MCP 调用失败: {result['error']}"
    content = result.get("content", "")
    if not content:
        return "MCP 返回结果为空。"
    # 截断过长内容
    if len(content) > 3000:
        content = content[:3000] + f"\n... (截断，共 {len(content)} 字符)"
    return content
