#!/usr/bin/env python3
"""RRF 三通道权重网格搜索。以 0.1 为步长遍历权重组合。"""
import json, sys, itertools
sys.path.insert(0, ".")

from backend.evaluation.dataset import load_golden_dataset
from backend.evaluation.metrics import compute_ragas_metrics
from backend.rag.utils import retrieve_documents, rrf_fusion_three_channel
from backend.embedding.service import EmbeddingService
from backend.milvus.client import MilvusManager

embedding_service = EmbeddingService()
milvus = MilvusManager()


def evaluate_with_weights(w1, w2, w3, dataset) -> float:
    """用指定权重跑完整评估，返回平均 context_precision。"""
    samples = []
    for item in dataset:
        dense_vec = embedding_service.get_embeddings([item["question"]])[0]
        sparse_vec = embedding_service.get_sparse_embedding(item["question"])
        dense_result = milvus.dense_retrieve(dense_vec, top_k=10)
        sparse_result = milvus.hybrid_retrieve(dense_vec, sparse_vec, top_k=10)

        fused = rrf_fusion_three_channel(
            dense_result, sparse_result, [],
            weights=(w1, w2, w3), top_k=5,
        )
        context_texts = [d.get("text", "") for d in fused]

        samples.append({
            "question": item["question"],
            "answer": item["ground_truth"],
            "contexts": context_texts,
            "ground_truth": item["ground_truth"],
        })

    metrics = compute_ragas_metrics(samples)
    return metrics.get("context_precision", 0.0)


def main():
    dataset = load_golden_dataset()
    total_combos = sum(1 for _ in itertools.product(
        [round(x*0.1, 1) for x in range(0, 11)], repeat=3
    ) if abs(sum(_)-1.0) < 0.001)

    print(f"数据集: {len(dataset)} 条")
    print(f"权重步长: 0.1, 归一化组合数: {total_combos}")

    best_score = 0.0
    best_weights = (0.4, 0.3, 0.3)
    results = []

    values = [round(x * 0.1, 1) for x in range(0, 11)]
    for w1, w2, w3 in itertools.product(values, repeat=3):
        if abs(w1 + w2 + w3 - 1.0) > 0.001:
            continue
        print(f"测试权重: ({w1}, {w2}, {w3})...", end=" ")
        try:
            score = evaluate_with_weights(w1, w2, w3, dataset)
            results.append({"weights": (w1, w2, w3), "context_precision": score})
            print(f"score={score:.4f}")
            if score > best_score:
                best_score = score
                best_weights = (w1, w2, w3)
        except Exception as e:
            print(f"error: {e}")

    print(f"\n===== 最佳权重 =====")
    print(f"DENSE={best_weights[0]}, SPARSE={best_weights[1]}, GRAPH={best_weights[2]}")
    print(f"Context Precision: {best_score:.4f}")

    with open("grid_search_result.json", "w") as f:
        json.dump({
            "best_weights": {"dense": best_weights[0], "sparse": best_weights[1], "graph": best_weights[2]},
            "best_score": best_score,
            "all_results": results,
        }, f, indent=2, ensure_ascii=False)
    print("结果已保存到 grid_search_result.json")


if __name__ == "__main__":
    main()
