"""文档向量化并写入 Milvus 模块

将文档数据进行向量化（生成密集 + 稀疏混合向量）并批量写入 Milvus，支持混合检索场景。
"""
from embedding import EmbeddingService
from milvus_client import MilvusManager

class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusManager = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.milvus_manager = milvus_manager or MilvusManager()

    def write_documents(self, documents: list[dict], batch_size: int = 50):
        """
        批量写入文档到 Milvus（同时生成密集和稀疏向量）
        :param documents: 文档列表
        :param batch_size: 批次大小
        空值校验	若传入的documents为空，直接返回，避免无效操作
初始化 Milvus 集合	调用milvus_manager.init_collection()，确保写入的集合已创建
语料库拟合	提取所有文档的文本，调用embedding_service.fit_corpus()，为稀疏向量（BM25）计算 IDF 值
分批处理	按batch_size（默认 50）拆分文档，避免单次处理数据量过大
向量生成	对每批文档：
- 调用get_embeddings()生成密集向量
- 调用get_sparse_embeddings()生成稀疏向量
数据组装	按 Milvus 入库格式组装数据，包含：
① 向量字段：dense_embedding（密集）、sparse_embedding（稀疏）
② 元数据字段：文本、文件名、文件类型、路径、页码、分块 ID / 层级等（保留文档溯源信息）
批量插入	调用milvus_manager.insert()将组装后的数据写入 Milvus
        """
        if not documents:
            return

        self.milvus_manager.init_collection()
        
        # 先拟合语料库（用于 BM25 IDF 计算）
        all_texts = [doc["text"] for doc in documents]
        self.embedding_service.fit_corpus(all_texts)

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]
            
            # ✅ 修复：分开调用正确的方法（删除了错误的 get_all_embeddings）
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
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                }
                for doc, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings)
            ]

            self.milvus_manager.insert(insert_data)