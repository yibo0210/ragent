#!/usr/bin/env python3
"""A/B 评测对比脚本 — 静态链路 vs 自适应动态链路。

用法:
    # 先运行静态链路（v11 默认权重）
    python scripts/run_ab_evaluation.py --mode static --limit 20 --output static_result.json

    # 再运行动态链路（v12 意图驱动权重）
    python scripts/run_ab_evaluation.py --mode dynamic --limit 20 --output dynamic_result.json

    # 对比两次结果
    python scripts/run_evaluation.py --compare static_result.json dynamic_result.json
"""
import argparse
import json
import sys
import time
import statistics
from collections import defaultdict

sys.path.insert(0, ".")

from backend.rag.utils import retrieve_documents
from backend.rag.pipeline import run_rag_graph
from backend.evaluation.dataset import load_golden_dataset
from backend.evaluation.metrics import compute_ragas_metrics


def _format_docs(docs: list[dict]) -> list[str]:
    return [d.get("text", "") for d in docs]


def _compute_latency_stats(latencies: list[float]) -> dict:
    if not latencies:
        return {}
    return {
        "avg": round(statistics.mean(latencies), 1),
        "p50": round(statistics.median(latencies), 1),
        "p95": round(sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0], 1),
        "max": round(max(latencies), 1),
        "min": round(min(latencies), 1),
    }


def run_static_evaluation(dataset: list[dict], limit: int = 0) -> dict:
    """静态链路评测（使用默认环境变量权重）。"""
    if limit > 0:
        dataset = dataset[:limit]

    samples = []
    latencies = []
    intent_distribution = defaultdict(int)

    for item in dataset:
        t0 = time.time()
        result = run_rag_graph(item["question"])
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        if result.get("force_interrupt"):
            continue

        docs = result.get("docs", [])
        samples.append({
            "question": item["question"],
            "answer": item["ground_truth"],
            "contexts": _format_docs(docs),
            "ground_truth": item["ground_truth"],
        })
        print(f"  [{item.get('query_type', '?')}] {item['id']}: {len(docs)} chunks, {latency:.0f}ms")

    metrics = compute_ragas_metrics(samples)

    return {
        "mode": "static",
        "metrics": metrics,
        "sample_count": len(samples),
        "latency": _compute_latency_stats(latencies),
        "intent_distribution": dict(intent_distribution),
    }


def run_dynamic_evaluation(dataset: list[dict], limit: int = 0) -> dict:
    """动态链路评测（使用 Query Profiler + 动态权重）。"""
    if limit > 0:
        dataset = dataset[:limit]

    from backend.agent.query_profiler import QueryProfiler
    from backend.rag.dynamic_rrf import get_weights_for_intent

    profiler = QueryProfiler(use_embedding=True)

    samples = []
    latencies = []
    intent_distribution = defaultdict(int)
    weights_used = {}

    for item in dataset:
        # 1. Query Profiler 分类
        intent = profiler.profile(item["question"])
        intent_distribution[intent.level] += 1
        weights_used[intent.level] = get_weights_for_intent(intent.level)

        # 2. 使用意图级别调用 RAG
        t0 = time.time()
        result = run_rag_graph(item["question"], intent_level=intent.level)
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        if result.get("force_interrupt"):
            continue

        docs = result.get("docs", [])
        samples.append({
            "question": item["question"],
            "answer": item["ground_truth"],
            "contexts": _format_docs(docs),
            "ground_truth": item["ground_truth"],
        })
        print(f"  [{intent.level}] {item['id']}: {len(docs)} chunks, {latency:.0f}ms")

    metrics = compute_ragas_metrics(samples)

    per_intent = defaultdict(list)
    for item, sample in zip(dataset[:len(samples)], samples):
        intent = profiler.profile(item["question"])
        per_intent[intent.level].append(sample)
    by_intent = {}
    for level, items in sorted(per_intent.items()):
        if len(items) >= 2:
            by_intent[level] = compute_ragas_metrics(items)

    return {
        "mode": "dynamic",
        "metrics": metrics,
        "sample_count": len(samples),
        "latency": _compute_latency_stats(latencies),
        "intent_distribution": dict(intent_distribution),
        "weights_used": {k: list(v) for k, v in weights_used.items()},
        "by_intent_level": by_intent,
    }


def main():
    parser = argparse.ArgumentParser(description="v12 A/B 评测对比")
    parser.add_argument("--mode", choices=["static", "dynamic"], required=True,
                        help="评测模式: static(v11默认), dynamic(v12自适应)")
    parser.add_argument("--limit", type=int, default=0, help="限制评测条数")
    parser.add_argument("--output", default="ab_result.json", help="输出文件路径")
    args = parser.parse_args()

    dataset = load_golden_dataset()
    print(f"数据集: {len(dataset)} 条, 模式: {args.mode}")

    t_start = time.time()

    if args.mode == "static":
        result = run_static_evaluation(dataset, args.limit)
    else:
        result = run_dynamic_evaluation(dataset, args.limit)

    result["total_time_seconds"] = round(time.time() - t_start, 1)

    print(f"\n===== 评估结果 [{args.mode}] =====")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
    print(f"\n样本数: {result['sample_count']}")
    print(f"总耗时: {result['total_time_seconds']}s")

    if result.get("intent_distribution"):
        print(f"\n===== 意图分布 =====")
        for level, count in sorted(result["intent_distribution"].items()):
            print(f"  {level}: {count}")

    if result.get("latency"):
        lat = result["latency"]
        print(f"\n===== 延迟 (ms) =====")
        print(f"  avg={lat['avg']}, p50={lat['p50']}, p95={lat['p95']}, max={lat['max']}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
