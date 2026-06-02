"""文档向量化并写入 Milvus 模块

将文档数据进行向量化（生成密集 + 稀疏混合向量）并批量写入 Milvus，支持混合检索场景。
"""
from backend.embedding.service import EmbeddingService
from .client import MilvusManager

class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusManager = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.milvus_manager = milvus_manager or MilvusManager()

    def write_documents(self, documents: list[dict], batch_size: int = 25, progress_callback=None):
        """批量写入文档到 Milvus（同时生成密集和稀疏向量）。

        Args:
            documents: 文档列表
            batch_size: 批次大小（DashScope 限制最大 25）
            progress_callback: 进度回调函数 callback(current, total, status)
        """
        if not documents:
            return

        self.milvus_manager.init_collection()

        # 拟合语料库（用于 BM25 IDF 计算）
        all_texts = [doc["text"] for doc in documents]
        self.embedding_service.fit_corpus(all_texts)

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]

            if progress_callback:
                progress_callback(min(i + batch_size, total), total, f"正在向量化 {min(i + batch_size, total)}/{total}...")

            # 生成稠密向量
            dense_embeddings = self.embedding_service.get_embeddings(texts)
            # 生成稀疏向量
            sparse_embeddings = self.embedding_service.get_sparse_embeddings(texts)

            insert_data = [
                {
                    "dense_embedding": dense_emb,
                    "sparse_embedding": sparse_emb,
                    "text": doc["text"],
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": int(doc.get("page_number", 0)),
                    "chunk_idx": int(doc.get("chunk_idx", 0)),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": int(doc.get("chunk_level", 0)),
                    "is_deleted": False,
                    "tenant_id": int(doc.get("tenant_id", 0)),
                }
                for doc, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings)
            ]

            self.milvus_manager.insert(insert_data)
