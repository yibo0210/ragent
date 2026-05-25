#!/usr/bin/env python3
"""自动化 RAG 评估入口脚本。支持 Ragas 指标计算和雷达图/柱状图可视化。"""
import json, sys, time
from collections import defaultdict
sys.path.insert(0, ".")

from backend.rag.utils import retrieve_documents
from backend.evaluation.dataset import load_golden_dataset
from backend.evaluation.metrics import compute_ragas_metrics


def run_evaluation() -> dict:
    dataset = load_golden_dataset()
    print(f"基准数据集: {len(dataset)} 条")

    samples = []
    by_type = defaultdict(list)
    for item in dataset:
        qtype = item.get("query_type", "general")
        print(f"  [{qtype}] {item['id']}: {item['question'][:50]}...")
        t0 = time.time()
        result = retrieve_documents(item["question"])
        docs = result.get("docs", [])
        context_texts = [d.get("text", "") for d in docs]

        sample = {
            "question": item["question"],
            "answer": item["ground_truth"],
            "contexts": context_texts,
            "ground_truth": item["ground_truth"],
        }
        samples.append(sample)
        by_type[qtype].append(sample)
        print(f"    耗时 {time.time() - t0:.1f}s, 检索 {len(docs)} 条")

    metrics = compute_ragas_metrics(samples)

    per_type = {}
    for qtype, items in sorted(by_type.items()):
        if len(items) >= 2:
            per_type[qtype] = compute_ragas_metrics(items)

    return {
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "sample_count": len(samples),
        "by_query_type": {
            qtype: {k: round(float(v), 4) for k, v in m.items()}
            for qtype, m in per_type.items()
        },
    }


def generate_charts(result: dict, output_dir: str = "."):
    """生成雷达图和分组柱状图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not installed, skipping charts. pip install matplotlib")
        return

    metrics = result["metrics"]
    per_type = result.get("by_query_type", {})

    # --- Radar Chart ---
    labels = list(metrics.keys())
    values = list(metrics.values())
    n = len(labels)
    if n < 3:
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    labels += labels[:1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    subplot_kw=dict(polar=True))

    ax1.fill(angles, values, alpha=0.25, color='#4FC08D')
    ax1.plot(angles, values, linewidth=2, color='#4FC08D')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([l.replace("_", "\n") for l in labels[:-1]], fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"RAG Metrics Radar (n={result['sample_count']})", pad=20, fontsize=12)

    # --- Grouped Bar Chart (by query type) ---
    plt.subplot(1, 2, 2)
    if per_type:
        qtypes = list(per_type.keys())
        metric_names = list(list(per_type.values())[0].keys())
        x = np.arange(len(metric_names))
        width = 0.8 / len(qtypes)
        colors = ['#4FC08D', '#3776AB', '#F56C6C', '#E6A23C']

        for i, qt in enumerate(qtypes):
            vals = [per_type[qt].get(m, 0) for m in metric_names]
            plt.bar(x + i * width, vals, width, label=qt, color=colors[i % len(colors)])

        plt.xticks(x + width * (len(qtypes) - 1) / 2,
                    [m.replace("_", "\n") for m in metric_names], fontsize=9)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("Metrics by Query Type", fontsize=12)
        plt.legend(fontsize=8)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

    fig.suptitle("Ragent AI v4.0 — RAG Evaluation Report", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = f"{output_dir}/evaluation_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"可视化图表已保存到 {chart_path}")


if __name__ == "__main__":
    result = run_evaluation()
    print(f"\n===== 评估结果 =====")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
    print(f"\n样本数: {result['sample_count']}")

    if result.get("by_query_type"):
        print(f"\n===== 按问题类型 =====")
        for qt, m in result["by_query_type"].items():
            print(f"  [{qt}] ", end="")
            print(", ".join(f"{k}={v}" for k, v in m.items()))

    with open("evaluation_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("JSON 结果已保存到 evaluation_result.json")

    generate_charts(result)
