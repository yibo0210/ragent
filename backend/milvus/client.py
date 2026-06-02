"""Milvus 向量数据库客户端管理类"""
import os
import warnings
import logging
from dotenv import load_dotenv

_log = logging.getLogger("ragent.milvus")
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

class MilvusManager:
    def __init__(self):
        self.uri = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        self._client_instance = None

    def _client(self):
        if self._client_instance is None:
            self._client_instance = MilvusClient(uri=self.uri)
        return self._client_instance

    def _ensure_connected(self):
        """每次查询前检查连接健康，必要时重连。"""
        try:
            self._client().get_load_state(self.collection_name)
        except Exception:
            self._reconnect()

    def _reconnect(self):
        """重置 gRPC 连接，用于断线重连。"""
        if self._client_instance:
            try:
                self._client_instance.close()
            except Exception as e:
                _log.warning("milvus_close_failed", error=str(e))
        self._client_instance = MilvusClient(uri=self.uri)

#集合初始化
    def init_collection(self):
        self._ensure_connected()
        client = self._client()
        if not client.has_collection(self.collection_name):
            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=1536)
            schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("text", DataType.VARCHAR, max_length=2000)
            schema.add_field("filename", DataType.VARCHAR, max_length=255)
            schema.add_field("file_type", DataType.VARCHAR, max_length=50)
            schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
            schema.add_field("page_number", DataType.INT64)
            schema.add_field("chunk_idx", DataType.INT64)
            schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("is_deleted", DataType.BOOL)
            schema.add_field("tenant_id", DataType.INT64, description="Tenant ID for multi-tenant isolation")

            index_params = client.prepare_index_params()
            index_params.add_index(field_name="dense_embedding", index_type="HNSW", metric_type="IP")
            index_params.add_index(field_name="sparse_embedding", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

            client.create_collection(collection_name=self.collection_name, schema=schema, index_params=index_params)
            client.load_collection(self.collection_name)
            print("[INFO] Milvus 集合首次创建完成！")
        else:
            client.load_collection(self.collection_name)
            print("[INFO] Milvus 集合已存在，直接加载使用")
#数据写入
    def insert(self, data):
        client = self._client()
        if not client.has_collection(self.collection_name):
            self.init_collection()
        res = client.insert(self.collection_name, data)
        print(f"[INFO] 成功写入Milvus：{res['insert_count']} 条数据")
        return res
#数据查询
    def query(self, filter_expr="", output_fields=None, limit=10000):
        self._ensure_connected()
        for attempt in range(3):
            try:
                client = self._client()
                return client.query(
                    collection_name=self.collection_name,
                    filter=filter_expr,
                    output_fields=output_fields or ["filename"],
                    limit=limit
                )
            except Exception as e:
                if attempt < 2 and ("closed channel" in str(e).lower() or "RPC" in str(e)):
                    self._reconnect()
                    continue
                if attempt >= 2:
                    return []
        return []
#数据删除
    def delete(self, filter_expr: str):
        client = self._client()
        return client.delete(collection_name=self.collection_name, filter=filter_expr)

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        """按 chunk_id 列表批量删除向量。返回删除条数。"""
        if not chunk_ids:
            return 0
        ids_str = ", ".join(f'"{cid}"' for cid in chunk_ids)
        filter_expr = f"chunk_id in [{ids_str}]"
        res = self.delete(filter_expr)
        return res.get("delete_count", 0) if isinstance(res, dict) else 0
    CACHE_COLLECTION = "semantic_cache_collection"

    def init_cache_collection(self):
        client = self._client()
        if not client.has_collection(self.CACHE_COLLECTION):
            schema = client.create_schema(auto_id=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)
            schema.add_field("query_hash", DataType.VARCHAR, max_length=64)
            schema.add_field("query_text", DataType.VARCHAR, max_length=2000)
            schema.add_field("source_doc", DataType.VARCHAR, max_length=255)
            index_params = client.prepare_index_params()
            index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE")
            client.create_collection(collection_name=self.CACHE_COLLECTION, schema=schema, index_params=index_params)
            client.load_collection(self.CACHE_COLLECTION)
        else:
            client.load_collection(self.CACHE_COLLECTION)

    def search_cache(self, query_vector: list[float], top_k: int = 3) -> list[dict]:
        client = self._client()
        results = client.search(
            collection_name=self.CACHE_COLLECTION,
            data=[query_vector],
            anns_field="embedding",
            search_params={"metric_type": "COSINE"},
            limit=top_k,
            output_fields=["query_hash", "query_text", "source_doc"],
        )
        return results[0] if results else []

    def insert_cache(self, query_vector, query_hash, query_text, source_doc=""):
        client = self._client()
        data = [{"embedding": query_vector, "query_hash": query_hash,
                 "query_text": query_text[:2000], "source_doc": source_doc}]
        return client.insert(self.CACHE_COLLECTION, data)

    def delete_cache_by_source(self, source_doc: str) -> int:
        client = self._client()
        res = client.delete(
            collection_name=self.CACHE_COLLECTION,
            filter=f'source_doc == "{source_doc}"',
        )
        return res.get("delete_count", 0) if isinstance(res, dict) else 0

#混合向量检索（hybrid_retrieve）
#RAG 场景核心方法，支持稠密 + 稀疏向量混合检索：
#构建两个向量检索请求（稠密向量 dense_embedding、稀疏向量 sparse_embedding），均使用 IP 度量方式；
#采用 RRFRanker 排序器对两类检索结果融合排序，返回 Top-K 结果（默认 5 条），输出文本内容和文件名。
    def hybrid_retrieve(self, dense_embedding, sparse_embedding, top_k=5, filter_expr=""):
        import time
        from backend.observability import get_tracer, Metrics

        self._ensure_connected()
        client = self._client()
        tracer = get_tracer("ragent.milvus")
        t0 = time.time()
        with tracer.start_as_current_span("milvus.hybrid_retrieve") as span:
            span.set_attribute("top_k", top_k)
            reqs = [
                AnnSearchRequest(data=[dense_embedding], anns_field="dense_embedding", param={"metric_type":"IP"}, limit=top_k),
                AnnSearchRequest(data=[sparse_embedding], anns_field="sparse_embedding", param={"metric_type":"IP"}, limit=top_k)
            ]
            if filter_expr:
                for req in reqs:
                    req.filter = filter_expr
            results = client.hybrid_search(
                collection_name=self.collection_name,
                reqs=reqs,
                ranker=RRFRanker(),
                limit=top_k,
                output_fields=["text", "filename", "chunk_id", "page_number"]
            )
            if results and isinstance(results[0], list):
                results = results[0]
            dt = time.time() - t0
            span.set_attribute("duration_ms", dt * 1000)
            span.set_attribute("result_count", len(results) if results else 0)
            Metrics.record_vector_search(dt)
            return results

    def get_chunks_by_ids(self, chunk_ids):
        if not chunk_ids:
            return []
        client = self._client()
        try:
            ids_str = ", ".join(f'"{cid}"' for cid in chunk_ids if cid)
            filter_expr = f"chunk_id in [{ids_str}]"
            return client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["text", "filename", "page_number", "chunk_id", "parent_chunk_id"],
                limit=len(chunk_ids)
            )
        except Exception:
            return []

    def dense_retrieve(self, dense_embedding, top_k=5, filter_expr=""):
        self._ensure_connected()
        client = self._client()
        try:
            results = client.search(
                collection_name=self.collection_name,
                data=[dense_embedding],
                anns_field="dense_embedding",
                search_params={"metric_type": "IP"},
                limit=top_k,
                filter=filter_expr,
                output_fields=["text", "filename", "page_number"]
            )
            # 展平嵌套列表 [[...]] -> [...]
            if results and isinstance(results[0], list):
                return results[0]
            return results
        except Exception:
            return []