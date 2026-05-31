#!/usr/bin/env python3
"""RRF 权重网格搜索。

以 0.1 为步长遍历权重组合，用 RAGAS 全指标（context_precision + faithfulness + answer_relevancy）
的加权组合作为优化目标。支持图谱通道接入。

用法:
    python scripts/grid_search_rrf.py
    python scripts/grid_search_rrf.py --step 0.2 --limit 10 --graph
"""
import argparse, json, sys, itertools, time
sys.path.insert(0, ".")

from backend.evaluation.dataset import load_golden_dataset
from backend.evaluation.metrics import compute_ragas_metrics
from backend.rag.utils import retrieve_documents, rrf_fusion_three_channel
from backend.embedding.service import EmbeddingService
from backend.milvus.client import MilvusManager

embedding_service = EmbeddingService()
milvus = MilvusManager()

# 优化目标权重
COMPOSITE_WEIGHTS = {
    "context_precision": 0.4,
    "faithfulness": 0.3,
    "answer_relevancy": 0.3,
}


def _try_get_graph_results(query: str, top_k: int = 5) -> list:
    """尝试获取图谱检索结果，失败返回空列表。"""
    try:
        from backend.rag.graph_retriever import local_graph_search
        result = local_graph_search(query, top_k=top_k)
        triples = result.get("graph_triples", [])
        # 将三元组转为 (doc, score) 格式
        formatted = []
        for t in triples:
            text = f"{t.get('subject', '')} --{t.get('predicate', '')}--> {t.get('object', '')}"
            formatted.append(({"text": text, "chunk_id": f"graph_{hash(text)}"}, 1.0))
        return formatted
    except Exception:
        return []


def evaluate_with_weights(w1, w2, w3, w4, dataset, use_graph: bool = False) -> dict:
    """用指定权重跑完整评估，返回全部 RAGAS 指标和 composite score。"""
    samples = []
    for item in dataset:
        dense_vec = embedding_service.get_embeddings([item["question"]])[0]
        sparse_vec = embedding_service.get_sparse_embedding(item["question"])
        dense_result = milvus.dense_retrieve(dense_vec, top_k=10)
        sparse_result = milvus.hybrid_retrieve(dense_vec, sparse_vec, top_k=10)

        graph_result = []
        if use_graph:
            graph_result = _try_get_graph_results(item["question"])

        fused = rrf_fusion_three_channel(
            dense_result, sparse_result, graph_result,
            weights=(w1, w2, w3, w4), top_k=5,
        )
        context_texts = [d.get("text", "") for d in fused]

        samples.append({
            "question": item["question"],
            "answer": item["ground_truth"],
            "contexts": context_texts,
            "ground_truth": item["ground_truth"],
        })

    metrics = compute_ragas_metrics(samples)

    composite = sum(
        COMPOSITE_WEIGHTS.get(k, 0) * v
        for k, v in metrics.items()
        if k in COMPOSITE_WEIGHTS
    )
    metrics["composite_score"] = round(composite, 4)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="RRF 权重网格搜索")
    parser.add_argument("--step", type=float, default=0.1, help="权重步长 (默认 0.1)")
    parser.add_argument("--limit", type=int, default=0, help="限制数据集条数 (0=全部)")
    parser.add_argument("--graph", action="store_true", help="启用图谱通道")
    parser.add_argument("--output", default="grid_search_result.json", help="输出文件路径")
    args = parser.parse_args()

    dataset = load_golden_dataset()
    if args.limit > 0:
        dataset = dataset[:args.limit]

    values = [round(x * args.step, 2) for x in range(int(1 / args.step) + 1)]

    # 根据是否启用图谱通道决定搜索空间
    if args.graph:
        combos = [
            (w1, w2, w3, w4)
            for w1, w2, w3, w4 in itertools.product(values, repeat=4)
            if abs(w1 + w2 + w3 + w4 - 1.0) < 0.001
        ]
    else:
        combos = [
            (w1, w2, w3, 0.0)
            for w1, w2, w3 in itertools.product(values, repeat=3)
            if abs(w1 + w2 + w3 - 1.0) < 0.001
        ]

    print(f"数据集: {len(dataset)} 条")
    print(f"权重步长: {args.step}, 组合数: {len(combos)}")
    print(f"图谱通道: {'启用' if args.graph else '禁用'}")
    print(f"优化目标: composite = {COMPOSITE_WEIGHTS}\n")

    best_composite = 0.0
    best_weights = (0.4, 0.3, 0.3, 0.0)
    results = []

    for i, weights in enumerate(combos):
        w1, w2, w3, w4 = weights
        label = f"({w1}, {w2}, {w3}"
        if args.graph:
            label += f", {w4}"
        label += ")"

        try:
            t0 = time.time()
            metrics = evaluate_with_weights(w1, w2, w3, w4, dataset, use_graph=args.graph)
            elapsed = time.time() - t0

            composite = metrics["composite_score"]
            results.append({"weights": {"dense": w1, "sparse": w2, "graph": w3, "visual": w4}, **metrics})

            print(f"[{i+1}/{len(combos)}] {label} composite={composite:.4f} "
                  f"prec={metrics.get('context_precision',0):.4f} "
                  f"faith={metrics.get('faithfulness',0):.4f} "
                  f"rel={metrics.get('answer_relevancy',0):.4f} "
                  f"recall={metrics.get('context_recall',0):.4f} "
                  f"({elapsed:.1f}s)")

            if composite > best_composite:
                best_composite = composite
                best_weights = weights
        except Exception as e:
            print(f"[{i+1}/{len(combos)}] {label} error: {e}")

    print(f"\n===== 最佳权重 =====")
    print(f"DENSE={best_weights[0]}, SPARSE={best_weights[1]}, GRAPH={best_weights[2]}")
    if args.graph:
        print(f"VISUAL={best_weights[3]}")
    print(f"Composite Score: {best_composite:.4f}")

    output = {
        "best_weights": {
            "dense": best_weights[0],
            "sparse": best_weights[1],
            "graph": best_weights[2],
            "visual": best_weights[3],
        },
        "best_composite_score": best_composite,
        "composite_weights": COMPOSITE_WEIGHTS,
        "graph_enabled": args.graph,
        "dataset_size": len(dataset),
        "total_combos": len(combos),
        "all_results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
