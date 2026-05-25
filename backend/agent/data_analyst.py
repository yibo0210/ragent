"""Data Analyst Agent - Text-to-SQL Worker

将自然语言问题转为 MySQL 只读 SQL 查询并执行，以结构化数据回答用户。
"""
import os
import re

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text

from backend.storage.database import SessionLocal

load_dotenv()

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


def generate_sql(question: str, schema: str) -> str:
    """调用 LLM 将自然语言转换为 SQL。"""
    import traceback
    from .orchestrator import _get_worker_model

    model = _get_worker_model()
    prompt = SQL_GENERATION_PROMPT.format(schema=schema, question=question)
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


def execute_sql(sql: str) -> dict:
    """执行只读 SQL 并返回结果。非 SELECT 语句会被拒绝。"""
    import datetime
    import decimal

    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned.upper().startswith("SELECT"):
        return {
            "error": "non_select",
            "sql": cleaned,
            "message": "仅允许执行 SELECT 查询",
        }

    db = SessionLocal()
    try:
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
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "sql": cleaned,
        }
    except Exception as e:
        return {"error": "execution_failed", "sql": cleaned, "message": str(e)}
    finally:
        db.close()


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
