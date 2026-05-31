"""MCP 工具语义检索器

当企业接入大量 MCP Server（20+）和工具（100+）时，
不能将所有工具描述塞给 LLM（会超 Token 限制）。
本模块将工具描述向量化存入 Milvus，查询时先语义检索 Top-K 最相关工具。
"""
import logging
from typing import Optional

from backend.milvus.client import MilvusManager
from backend.embedding.service import EmbeddingService

log = logging.getLogger(__name__)

MCP_TOOLS_COLLECTION = "mcp_tools"


class ToolRetriever:
    """MCP 工具语义检索器。"""

    def __init__(self, milvus_manager: MilvusManager = None, embedding_service: EmbeddingService = None):
        self.milvus = milvus_manager or MilvusManager()
        self.embedding = embedding_service or EmbeddingService()

    def index_tools(self, tools: list[dict], server_name: str):
        """将 MCP 工具描述向量化存入 Milvus。

        Args:
            tools: MCP 工具列表 [{name, description, input_schema}]
            server_name: MCP Server 名称
        """
        if not tools:
            return

        # 构建文档
        docs = []
        for tool_def in tools:
            name = tool_def.get("name", "")
            desc = tool_def.get("description", "")
            # 合并名称和描述作为索引文本
            text = f"{name}: {desc}"
            docs.append({
                "text": text,
                "tool_name": name,
                "server_name": server_name,
                "input_schema": str(tool_def.get("input_schema", {})),
                "file_type": "mcp_tool",
                "filename": f"mcp://{server_name}/{name}",
            })

        # 生成向量
        texts = [d["text"] for d in docs]
        vectors = self.embedding.get_embeddings(texts)

        # 写入 Milvus（使用已有的 write_documents 逻辑）
        for doc, vec in zip(docs, vectors):
            doc["dense_vector"] = vec

        self._upsert_tools(docs)
        log.info("索引 MCP 工具: server=%s, count=%d", server_name, len(docs))

    def _upsert_tools(self, docs: list[dict]):
        """将工具文档写入 Milvus。"""
        try:
            self.milvus.init_collection()
            for doc in docs:
                self.milvus.insert(doc)
        except Exception as e:
            log.error("写入 MCP 工具失败: %s", e)

    def retrieve_tools(self, query: str, top_k: int = 5) -> list[dict]:
        """语义检索最相关的 MCP 工具。

        Args:
            query: 用户查询
            top_k: 返回数量

        Returns:
            相关工具列表 [{name, server_name, description, score}]
        """
        try:
            query_vec = self.embedding.get_embeddings([query])[0]
            self.milvus.init_collection()
            results = self.milvus.search(
                data=[query_vec],
                limit=top_k,
                output_fields=["tool_name", "server_name", "description", "input_schema"],
                filter_expr='file_type == "mcp_tool"',
            )

            tools = []
            if results and results[0]:
                for hit in results[0]:
                    entity = hit.get("entity", {})
                    tools.append({
                        "name": entity.get("tool_name", ""),
                        "server_name": entity.get("server_name", ""),
                        "description": entity.get("description", ""),
                        "input_schema": entity.get("input_schema", "{}"),
                        "score": hit.get("distance", 0),
                    })
            return tools
        except Exception as e:
            log.error("MCP 工具检索失败: %s", e)
            return []

    def get_all_indexed_tools(self) -> list[dict]:
        """获取所有已索引的 MCP 工具。"""
        try:
            self.milvus.init_collection()
            results = self.milvus.query(
                filter_expr='file_type == "mcp_tool"',
                output_fields=["tool_name", "server_name", "description"],
                limit=1000,
            )
            return [
                {
                    "name": r.get("tool_name", ""),
                    "server_name": r.get("server_name", ""),
                    "description": r.get("description", ""),
                }
                for r in results
            ]
        except Exception:
            return []


# 全局单例
_tool_retriever: Optional[ToolRetriever] = None


def get_tool_retriever() -> ToolRetriever:
    """获取全局工具检索器单例。"""
    global _tool_retriever
    if _tool_retriever is None:
        _tool_retriever = ToolRetriever()
    return _tool_retriever
