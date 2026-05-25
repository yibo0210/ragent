#!/usr/bin/env python3
"""自动化 RAG 评估入口脚本。"""
import json, sys, time
sys.path.insert(0, ".")

from backend.rag.utils import retrieve_documents
from backend.evaluation.dataset import load_golden_dataset
from backend.evaluation.metrics import compute_ragas_metrics


def run_evaluation() -> dict:
    dataset = load_golden_dataset()
    print(f"基准数据集: {len(dataset)} 条")

    samples = []
    for item in dataset:
        print(f"  评估 {item['id']}: {item['question'][:50]}...")
        t0 = time.time()
        result = retrieve_documents(item["question"])
        docs = result.get("docs", [])
        context_texts = [d.get("text", "") for d in docs]

        samples.append({
            "question": item["question"],
            "answer": item["ground_truth"],
            "contexts": context_texts,
            "ground_truth": item["ground_truth"],
        })
        print(f"    耗时 {time.time() - t0:.1f}s, 检索 {len(docs)} 条")

    metrics = compute_ragas_metrics(samples)
    return {"metrics": metrics, "sample_count": len(samples)}


if __name__ == "__main__":
    result = run_evaluation()
    print(f"\n===== 评估结果 =====")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
    print(f"\n样本数: {result['sample_count']}")

    with open("evaluation_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("结果已保存到 evaluation_result.json")
