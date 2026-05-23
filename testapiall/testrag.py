from backend.milvus_client import MilvusManager  # 以backend为顶层包，规范引用
# 模拟RAG检索时的向量输入（实际由embedding.py生成，此处仅用于测试）
dense_embedding = [0.1]*1536  # 稠密向量（维度1536，与Milvus字段一致）
sparse_embedding = {}  # 稀疏向量（可空，不影响测试）

milvus = MilvusManager()
# 执行与RAG相同的混合检索
retrieve_result = milvus.hybrid_retrieve(dense_embedding, sparse_embedding, top_k=5)
print(f"RAG检索结果条数：{len(retrieve_result)}")
if len(retrieve_result) > 0:
    print("✅ RAG检索有效（已获取文档内容）：")
    for res in retrieve_result:
        filename = res.fields.get("filename")  # 关键：加.fields
        text = res.fields.get("text", "")[:50]  # 避免text为空报错
        print(f"- 文档：{filename}，内容片段：{text}...")