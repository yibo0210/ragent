#!/usr/bin/env python3
"""并发压测：对比缓存开启/关闭的 QPS 和 Token 消耗。"""
import time, json, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ".")

TEST_QUERIES = [
    "什么是 Ragent AI？", "系统用了什么向量数据库？", "如何配置 Neo4j？",
    "HITL 是什么？", "Leiden 算法怎么用？", "文档上传支持哪些格式？",
    "milvus 端口是多少？", "embedding 模型叫什么？", "Supervisor 怎么工作？",
    "什么是语义缓存？",
]
CONCURRENT_USERS = 5
REQUESTS_PER_USER = 2


def simulate_request(query: str) -> dict:
    t0 = time.time()
    from backend.cache import query_cache
    cache_result = query_cache(query)
    if cache_result:
        return {"cached": True, "latency_ms": (time.time() - t0) * 1000, "tokens": 0}

    from backend.rag.utils import retrieve_documents
    result = retrieve_documents(query)
    latency = (time.time() - t0) * 1000
    return {
        "cached": False, "latency_ms": latency,
        "docs": len(result.get("docs", [])),
        "tokens_estimated": sum(len(d.get("text", "")) for d in result.get("docs", [])) // 4,
    }


def run_benchmark():
    total = CONCURRENT_USERS * REQUESTS_PER_USER
    print(f"并发用户: {CONCURRENT_USERS}, 每用户请求: {REQUESTS_PER_USER}, 总数: {total}")

    with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        futures = [
            executor.submit(simulate_request, TEST_QUERIES[(i * REQUESTS_PER_USER + j) % len(TEST_QUERIES)])
            for i in range(CONCURRENT_USERS) for j in range(REQUESTS_PER_USER)
        ]
        results = [f.result() for f in futures]

    cached = [r for r in results if r["cached"]]
    uncached = [r for r in results if not r["cached"]]

    print(f"\n===== 压测结果 =====")
    print(f"缓存命中: {len(cached)}/{total} ({len(cached)/total*100:.0f}%)")
    if cached:
        print(f"命中平均延迟: {sum(r['latency_ms'] for r in cached)/len(cached):.0f}ms")
    if uncached:
        print(f"穿透平均延迟: {sum(r['latency_ms'] for r in uncached)/len(uncached):.0f}ms")
        print(f"估算 Token: {sum(r['tokens_estimated'] for r in uncached)}")

    return {"total": total, "cache_hits": len(cached), "cache_misses": len(uncached)}


if __name__ == "__main__":
    result = run_benchmark()
