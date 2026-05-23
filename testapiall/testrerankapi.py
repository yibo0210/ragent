from dotenv import load_dotenv
import os

load_dotenv()

print("=== Rerank 环境变量检测 ===")
print("RERANK_MODEL =", os.getenv("RERANK_MODEL"))
print("RERANK_BINDING_HOST =", os.getenv("RERANK_BINDING_HOST"))
print("RERANK_API_KEY =", os.getenv("RERANK_API_KEY"))

# 判断是否配置
if os.getenv("RERANK_MODEL"):
    print("\n✅ 已配置 Rerank，系统会自动使用重排")
else:
    print("\n❌ 未配置 Rerank，系统自动降级为普通检索")