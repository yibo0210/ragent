"""联网搜索模块

基于 Tavily API 实现实时网络搜索能力，供 Web Searcher Worker 调用。

功能：
- 调用 Tavily Search API 获取实时网络信息
- 通过 emit_rag_step 向前端推送搜索进度
- 返回结构化搜索结果供 LLM 生成回答
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Tavily API 配置
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))


def run_web_search(query: str, max_results: int = None) -> dict:
    """执行联网搜索，返回结构化结果。

    Args:
        query: 搜索查询
        max_results: 最大返回结果数，默认从环境变量读取

    Returns:
        dict: {
            "results": [{"title", "url", "snippet", "score"}],
            "answer": "Tavily 生成的摘要答案",
            "query": "原始查询",
            "error": "错误信息（仅失败时）"
        }
    """
    from .tools import emit_rag_step

    if max_results is None:
        max_results = WEB_SEARCH_MAX_RESULTS

    # 检查 API Key 配置
    if not TAVILY_API_KEY:
        emit_rag_step(
            "⚠️", "联网搜索未配置", "缺少 TAVILY_API_KEY",
            agent="web_searcher",
        )
        return {
            "results": [],
            "answer": "",
            "query": query,
            "error": "TAVILY_API_KEY 未配置，无法执行联网搜索",
        }

    emit_rag_step(
        "🌐", "正在联网搜索...",
        f"查询: {query[:80]}",
        agent="web_searcher",
    )

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=max_results)

        # 解析搜索结果
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "score": item.get("score", 0),
            })

        answer = response.get("answer", "")

        emit_rag_step(
            "📄", f"找到 {len(results)} 条搜索结果",
            f"答案摘要: {answer[:60]}..." if answer else "",
            agent="web_searcher",
        )
        emit_rag_step(
            "✅", "联网搜索完成", "",
            agent="web_searcher",
        )

        return {
            "results": results,
            "answer": answer,
            "query": query,
        }

    except ImportError:
        emit_rag_step(
            "❌", "搜索依赖缺失", "请安装 tavily-python: uv add tavily-python",
            agent="web_searcher",
        )
        return {
            "results": [],
            "answer": "",
            "query": query,
            "error": "tavily-python 未安装",
        }
    except Exception as e:
        error_msg = str(e)[:100]
        emit_rag_step(
            "❌", f"联网搜索失败", error_msg,
            agent="web_searcher",
        )
        return {
            "results": [],
            "answer": "",
            "query": query,
            "error": error_msg,
        }


def format_web_search_context(search_result: dict) -> str:
    """将搜索结果格式化为 LLM 可读的上下文文本。

    Args:
        search_result: run_web_search 的返回值

    Returns:
        str: 格式化的上下文文本
    """
    results = search_result.get("results", [])
    answer = search_result.get("answer", "")

    if not results and not answer:
        return ""

    parts = []

    # Tavily 生成的摘要答案
    if answer:
        parts.append(f"【搜索摘要】\n{answer}")

    # 详细搜索结果
    if results:
        parts.append("【搜索结果详情】")
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            parts.append(f"[{i}] {title}\n链接: {url}\n摘要: {snippet}")

    return "\n\n".join(parts)
