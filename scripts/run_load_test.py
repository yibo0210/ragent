#!/usr/bin/env python3
"""Locust 压测脚本 — 模拟不同 QPS 并发流量。

用法:
    # 使用 Locust Web UI
    locust -f scripts/run_load_test.py --host http://localhost:8000

    # 无头模式压测
    locust -f scripts/run_load_test.py --host http://localhost:8000 \
        --users 50 --spawn-rate 10 --run-time 2m --headless
"""
import json
import random
from locust import HttpUser, task, between, events


# 测试查询集（覆盖不同意图类型）
TEST_QUERIES = [
    # L1_FACTUAL
    "你好",
    "Python 是什么？",
    "今天天气怎么样？",
    "谢谢",
    # L2_REASONING
    "Milvus 和 Neo4j 之间有什么关系？",
    "GraphRAG 依赖哪些组件？",
    "系统如何实现多跳推理？",
    "Dense 和 Sparse 检索的区别是什么？",
    # L3_MACRO_SUMMARY
    "请总结系统的整体技术架构",
    "所有文档中涉及的主要技术有哪些？",
    "系统的核心模块概览",
    "请综述各方面的设计思路",
]


class RagentUser(HttpUser):
    """模拟 Ragent AI 用户。"""
    wait_time = between(0.5, 2.0)

    @task(3)
    def chat_stream(self):
        """SSE 流式对话（主要测试场景）。"""
        query = random.choice(TEST_QUERIES)
        with self.client.post(
            "/api/chat/stream",
            json={"message": query, "session_id": f"load_test_{random.randint(1, 1000)}"},
            stream=True,
            name="/api/chat/stream",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                full_content = ""
                for line in response.iter_lines():
                    line = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                            if event.get("type") == "content":
                                full_content += event.get("content", "")
                        except json.JSONDecodeError:
                            pass
                if full_content:
                    response.success()
                else:
                    response.failure("Empty response content")
            elif response.status_code == 423:
                response.failure("HITL lock (423)")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def chat_sync(self):
        """同步对话。"""
        query = random.choice(TEST_QUERIES)
        self.client.post(
            "/api/chat",
            json={"message": query, "session_id": f"load_test_{random.randint(1, 1000)}"},
            name="/api/chat",
        )

    @task(1)
    def health_check(self):
        """健康检查。"""
        self.client.get("/api/health", name="/api/health")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("=" * 60)
    print("Ragent AI 压测开始")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("=" * 60)
    print("Ragent AI 压测结束")
    print("=" * 60)
