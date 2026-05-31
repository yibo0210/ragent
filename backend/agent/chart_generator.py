"""Echarts 图表生成器

根据数据和用户问题，生成 Echarts JSON 配置。
前端 Vue 渲染器检测 ```echarts 代码块并渲染为图表。
"""
import json
import re

from langchain_core.messages import HumanMessage


CHART_TYPE_PROMPT = """你是一个数据可视化专家。根据用户问题和数据，判断最适合的图表类型。

可用类型: bar (柱状图), line (折线图), pie (饼图), table (表格)

用户问题: {question}
数据预览: {data_preview}

只输出图表类型名称，不要解释。"""


ECHARTS_PROMPT = """你是一个 Echarts 配置生成专家。根据数据生成 Echarts JSON 配置。

规则：
- 只输出 JSON，不要 markdown 代码块
- 使用标准 Echarts 配置格式
- 标题、坐标轴标签使用中文
- 颜色方案使用: ['#5470C6', '#91CC75', '#FAC858', '#EE6666', '#73C0DE', '#3BA272']

图表类型: {chart_type}
用户问题: {question}
数据:
{data}

输出 Echarts JSON 配置:"""


def detect_chart_type(question: str, data: dict) -> str:
    """LLM 判断最适合的图表类型。"""
    from .orchestrator import _get_worker_model

    model = _get_worker_model()
    data_preview = json.dumps(data, ensure_ascii=False, default=str)[:500]
    prompt = CHART_TYPE_PROMPT.format(question=question, data_preview=data_preview)

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip().lower()
        for t in ["bar", "line", "pie", "table"]:
            if t in content:
                return t
    except Exception:
        pass
    return "bar"


def generate_echarts_config(data: dict, chart_type: str, question: str = "") -> dict:
    """根据数据生成 Echarts JSON 配置。"""
    from .orchestrator import _get_worker_model

    model = _get_worker_model()
    data_str = json.dumps(data, ensure_ascii=False, default=str)[:2000]
    prompt = ECHARTS_PROMPT.format(chart_type=chart_type, question=question, data=data_str)

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        # 提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return {}


def format_chart_markdown(echarts_config: dict, chart_type: str) -> str:
    """将 Echarts 配置格式化为 markdown 代码块。"""
    if not echarts_config:
        return ""
    config_str = json.dumps(echarts_config, ensure_ascii=False, indent=2)
    return f"\n\n```echarts\n{config_str}\n```\n"


def format_data_table(data: dict) -> str:
    """将结构化数据格式化为 markdown 表格。"""
    rows = data.get("rows", [])
    cols = data.get("columns", [])
    if not rows or not cols:
        return ""

    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows[:30]:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    if len(rows) > 30:
        lines.append(f"| ... 还有 {len(rows) - 30} 行 |")
    return "\n".join(lines)
