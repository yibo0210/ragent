# 检查 Milvus 向量库是否有数据（直接运行，无需修改）
from pymilvus import MilvusClient

# 连接你本地的 Milvus
client = MilvusClient(uri="http://127.0.0.1:19530")
COLLECTION = "embeddings_collection"

print("🔍 正在检查向量数据库...")

# 1. 检查集合是否存在
if not client.has_collection(COLLECTION):
    print("❌ 集合不存在，向量库为空")
else:
    print("✅ 集合已存在")

    # 2. 查询总数据量（最关键！）
    result = client.query(
        collection_name=COLLECTION,
        output_fields=["count(*)"]
    )
    total = result[0]["count(*)"]
    print(f"📊 向量库总数据条数：{total}")

    # 3. 判断结果
    if total > 0:
        print("🎉 数据存在！前端不显示是因为 Milvus 没加载数据！")
    else:
        print("⚠️  集合存在，但无数据，写入失败")