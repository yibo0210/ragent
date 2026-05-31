#!/usr/bin/env python3
"""自动化 RAG 评估入口脚本。

支持三种评测模式：
- retrieval: 仅评测初始检索质量（answer = ground_truth）
- pipeline: 评测完整 RAG pipeline 检索质量（answer = ground_truth，走 run_rag_graph）
- e2e: 端到端评测（LLM 真实生成 answer + 路由准确率 + 延迟统计）

用法:
    python scripts/run_evaluation.py --mode retrieval --limit 10
    python scripts/run_evaluation.py --mode e2e
    python scripts/run_evaluation.py --compare result_a.json result_b.json
"""
import argparse, json, sys, time, statistics
from collections import defaultdict
sys.path.insert(0, ".")

from backend.rag.utils import retrieve_documents
from backend.rag.pipeline import run_rag_graph
from backend.evaluation.dataset import load_golden_dataset
from backend.evaluation.metrics import (
    compute_ragas_metrics,
    generate_answer,
    evaluate_routing_accuracy,
)


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


def run_retrieval_evaluation(dataset: list[dict], limit: int = 0) -> dict:
    """retrieval 模式：仅初始检索，answer = ground_truth。"""
    if limit > 0:
        dataset = dataset[:limit]

    samples = []
    latencies = []
    for item in dataset:
        t0 = time.time()
        result = retrieve_documents(item["question"])
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        docs = result.get("docs", [])
        samples.append({
            "question": item["question"],
            "answer": item["ground_truth"],
            "contexts": _format_docs(docs),
            "ground_truth": item["ground_truth"],
        })
        print(f"  [{item.get('query_type', '?')}] {item['id']}: {len(docs)} chunks, {latency:.0f}ms")

    metrics = compute_ragas_metrics(samples)

    per_type = defaultdict(list)
    for item, sample in zip(dataset, samples):
        per_type[item.get("query_type", "unknown")].append(sample)
    by_type = {}
    for qt, items in sorted(per_type.items()):
        if len(items) >= 2:
            by_type[qt] = compute_ragas_metrics(items)

    return {
        "mode": "retrieval",
        "metrics": metrics,
        "sample_count": len(samples),
        "by_query_type": by_type,
        "latency": _compute_latency_stats(latencies),
    }


def run_pipeline_evaluation(dataset: list[dict], limit: int = 0) -> dict:
    """pipeline 模式：完整 RAG pipeline，answer = ground_truth。"""
    if limit > 0:
        dataset = dataset[:limit]

    samples = []
    latencies = []
    for item in dataset:
        t0 = time.time()
        result = run_rag_graph(item["question"])
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        if result.get("force_interrupt"):
            print(f"  [{item.get('query_type', '?')}] {item['id']}: HITL interrupt, skipped")
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

    per_type = defaultdict(list)
    for item, sample in zip(dataset[:len(samples)], samples):
        per_type[item.get("query_type", "unknown")].append(sample)
    by_type = {}
    for qt, items in sorted(per_type.items()):
        if len(items) >= 2:
            by_type[qt] = compute_ragas_metrics(items)

    return {
        "mode": "pipeline",
        "metrics": metrics,
        "sample_count": len(samples),
        "by_query_type": by_type,
        "latency": _compute_latency_stats(latencies),
    }


def run_e2e_evaluation(dataset: list[dict], limit: int = 0) -> dict:
    """e2e 模式：端到端评测（LLM 生成 answer + 路由准确率 + 延迟）。"""
    if limit > 0:
        dataset = dataset[:limit]

    samples = []
    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []

    for item in dataset:
        t_total = time.time()

        # 检索阶段
        t0 = time.time()
        result = run_rag_graph(item["question"])
        retrieval_latency = (time.time() - t0) * 1000

        if result.get("force_interrupt"):
            print(f"  [{item.get('query_type', '?')}] {item['id']}: HITL interrupt, skipped")
            continue

        docs = result.get("docs", [])
        contexts = _format_docs(docs)

        # 生成阶段
        answer, generation_latency = generate_answer(item["question"], contexts)

        total_latency = (time.time() - t_total) * 1000
        retrieval_latencies.append(retrieval_latency)
        generation_latencies.append(generation_latency)
        total_latencies.append(total_latency)

        samples.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })
        print(f"  [{item.get('query_type', '?')}] {item['id']}: "
              f"retrieval={retrieval_latency:.0f}ms, gen={generation_latency:.0f}ms, total={total_latency:.0f}ms")

    metrics = compute_ragas_metrics(samples)

    per_type = defaultdict(list)
    for item, sample in zip(dataset[:len(samples)], samples):
        per_type[item.get("query_type", "unknown")].append(sample)
    by_type = {}
    for qt, items in sorted(per_type.items()):
        if len(items) >= 2:
            by_type[qt] = compute_ragas_metrics(items)

    # 路由准确率
    routing = evaluate_routing_accuracy(dataset)

    return {
        "mode": "e2e",
        "metrics": metrics,
        "sample_count": len(samples),
        "by_query_type": by_type,
        "latency": {
            "retrieval": _compute_latency_stats(retrieval_latencies),
            "generation": _compute_latency_stats(generation_latencies),
            "total": _compute_latency_stats(total_latencies),
        },
        "routing_accuracy": {
            "accuracy": routing["accuracy"],
            "total": routing["total"],
            "correct": routing["correct"],
            "by_query_type": routing["by_query_type"],
        },
    }


def compare_evaluations(result_a: dict, result_b: dict, label_a: str = "A", label_b: str = "B") -> str:
    """对比两次评测结果，生成 markdown diff 报告。"""
    lines = [f"# 评测对比: {label_a} vs {label_b}\n"]

    # 指标对比
    lines.append("## RAGAS 指标\n")
    lines.append(f"| 指标 | {label_a} | {label_b} | 变化 |")
    lines.append("|------|---------|---------|------|")
    metrics_a = result_a.get("metrics", {})
    metrics_b = result_b.get("metrics", {})
    all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
    for k in all_keys:
        va = metrics_a.get(k, 0)
        vb = metrics_b.get(k, 0)
        diff = vb - va
        pct = (diff / va * 100) if va else 0
        arrow = "+" if diff > 0 else "" if diff == 0 else ""
        lines.append(f"| {k} | {va:.4f} | {vb:.4f} | {arrow}{diff:.4f} ({arrow}{pct:.1f}%) |")

    # 延迟对比
    lat_a = result_a.get("latency", {})
    lat_b = result_b.get("latency", {})
    if lat_a or lat_b:
        lines.append("\n## 延迟 (ms)\n")
        lines.append(f"| 指标 | {label_a} | {label_b} | 变化 |")
        lines.append("|------|---------|---------|------|")
        # 兼容 e2e 模式的嵌套结构
        lat_a_total = lat_a.get("total", lat_a) if isinstance(lat_a.get("total"), dict) else lat_a
        lat_b_total = lat_b.get("total", lat_b) if isinstance(lat_b.get("total"), dict) else lat_b
        for k in ["avg", "p50", "p95", "max"]:
            va = lat_a_total.get(k, 0)
            vb = lat_b_total.get(k, 0)
            diff = vb - va
            lines.append(f"| {k} | {va:.1f} | {vb:.1f} | {diff:+.1f} |")

    # 路由准确率对比
    rt_a = result_a.get("routing_accuracy", {})
    rt_b = result_b.get("routing_accuracy", {})
    if rt_a or rt_b:
        lines.append("\n## 路由准确率\n")
        lines.append(f"| 指标 | {label_a} | {label_b} | 变化 |")
        lines.append("|------|---------|---------|------|")
        acc_a = rt_a.get("accuracy", 0)
        acc_b = rt_b.get("accuracy", 0)
        lines.append(f"| accuracy | {acc_a:.4f} | {acc_b:.4f} | {acc_b - acc_a:+.4f} |")

    return "\n".join(lines)


def generate_charts(result: dict, output_dir: str = "."):
    """生成雷达图和分组柱状图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not installed, skipping charts.")
        return

    metrics = result["metrics"]
    per_type = result.get("by_query_type", {})
    mode = result.get("mode", "unknown")

    labels = list(metrics.keys())
    values = [v if (v == v and v is not None) else 0 for v in metrics.values()]  # NaN -> 0
    if not labels or all(v == 0 for v in values):
        return
    n = len(labels)
    if n < 3:
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    labels += labels[:1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), subplot_kw=dict(polar=True))

    ax1.fill(angles, values, alpha=0.25, color='#4FC08D')
    ax1.plot(angles, values, linewidth=2, color='#4FC08D')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([l.replace("_", "\n") for l in labels[:-1]], fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"RAG Metrics Radar [{mode}] (n={result['sample_count']})", pad=20, fontsize=12)

    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2)
    if per_type:
        qtypes = list(per_type.keys())
        metric_names = list(list(per_type.values())[0].keys())
        x = np.arange(len(metric_names))
        width = 0.8 / len(qtypes)
        colors = ['#4FC08D', '#3776AB', '#F56C6C', '#E6A23C']

        for i, qt in enumerate(qtypes):
            vals = []
            for m in metric_names:
                v = per_type[qt].get(m, 0)
                vals.append(v if (v == v and v is not None) else 0)
            plt.bar(x + i * width, vals, width, label=qt, color=colors[i % len(colors)])

        plt.xticks(x + width * (len(qtypes) - 1) / 2,
                    [m.replace("_", "\n") for m in metric_names], fontsize=9)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("Metrics by Query Type", fontsize=12)
        plt.legend(fontsize=8)
        ax2 = plt.gca()
        for spine_name in ['top', 'right']:
            if spine_name in ax2.spines:
                ax2.spines[spine_name].set_visible(False)

    fig.suptitle(f"Ragent AI — RAG Evaluation Report [{mode}]", fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = f"{output_dir}/evaluation_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存到 {chart_path}")


def _compare_topology(stats_a: dict, stats_b: dict, label_a: str, label_b: str) -> str:
    """对比两份图谱拓扑统计，输出 Markdown 表格。"""
    lines = [f"# 图谱拓扑对比\n", f"- **A**: {label_a}", f"- **B**: {label_b}", ""]

    # 概览指标
    lines.append("## 概览指标\n")
    lines.append("| 指标 | A | B | 变化 |")
    lines.append("|------|---|---|------|")
    overview_keys = [
        ("total_nodes", "总节点数"),
        ("total_edges", "总边数"),
        ("orphan_nodes", "孤岛节点"),
        ("orphan_rate", "孤岛率"),
        ("avg_degree", "平均度"),
        ("edge_density", "边密度"),
    ]
    for key, label in overview_keys:
        va = stats_a.get(key, 0)
        vb = stats_b.get(key, 0)
        if isinstance(va, float):
            diff = round(vb - va, 6)
            pct = f"{diff:+.4f}" if va == 0 else f"{diff:+.4f} ({diff/va*100:+.1f}%)"
            lines.append(f"| {label} | {va:.4f} | {vb:.4f} | {pct} |")
        else:
            diff = vb - va
            pct = f"{diff:+d}" if va == 0 else f"{diff:+d} ({diff/va*100:+.1f}%)"
            lines.append(f"| {label} | {va} | {vb} | {pct} |")

    # 类型分布
    lines.append("\n## 实体类型分布\n")
    lines.append("| 类型 | A | B | 变化 |")
    lines.append("|------|---|---|------|")
    all_types = sorted(set(list(stats_a.get("type_distribution", {}).keys()) +
                           list(stats_b.get("type_distribution", {}).keys())))
    for t in all_types:
        va = stats_a.get("type_distribution", {}).get(t, 0)
        vb = stats_b.get("type_distribution", {}).get(t, 0)
        diff = vb - va
        pct = f"{diff:+d}" if va == 0 else f"{diff:+d} ({diff/va*100:+.1f}%)"
        lines.append(f"| {t} | {va} | {vb} | {pct} |")

    # 谓词分布
    lines.append("\n## 关系谓词分布\n")
    lines.append("| 谓词 | A | B | 变化 |")
    lines.append("|------|---|---|------|")
    all_preds = sorted(set(list(stats_a.get("predicate_distribution", {}).keys()) +
                           list(stats_b.get("predicate_distribution", {}).keys())))
    for p in all_preds:
        va = stats_a.get("predicate_distribution", {}).get(p, 0)
        vb = stats_b.get("predicate_distribution", {}).get(p, 0)
        diff = vb - va
        pct = f"{diff:+d}" if va == 0 else f"{diff:+d} ({diff/va*100:+.1f}%)"
        lines.append(f"| {p} | {va} | {vb} | {pct} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Ragent AI RAG 评测")
    parser.add_argument("--mode", choices=["retrieval", "pipeline", "e2e", "graph", "graph_compare"], default="retrieval",
                        help="评测模式: retrieval(初始检索), pipeline(完整RAG), e2e(端到端), graph(图谱拓扑), graph_compare(图谱对比)")
    parser.add_argument("--limit", type=int, default=0, help="限制评测条数 (0=全部)")
    parser.add_argument("--output", default="evaluation_result.json", help="输出文件路径")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="对比两个评测结果")
    args = parser.parse_args()

    # 图谱拓扑模式
    if args.mode == "graph":
        from scripts.graph_topology_stats import collect_topology_stats
        stats = collect_topology_stats()
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"图谱拓扑统计已保存到 {args.output}")
        print(f"  节点: {stats['total_nodes']}, 边: {stats['total_edges']}, "
              f"孤岛: {stats['orphan_nodes']} ({stats['orphan_rate']:.1%})")
        return

    # 图谱对比模式
    if args.mode == "graph_compare":
        if not args.compare:
            print("错误: graph_compare 模式需要 --compare A.json B.json")
            return
        with open(args.compare[0]) as f:
            stats_a = json.load(f)
        with open(args.compare[1]) as f:
            stats_b = json.load(f)
        report = _compare_topology(stats_a, stats_b, args.compare[0], args.compare[1])
        print(report)
        with open("topology_comparison.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("\n对比报告已保存到 topology_comparison.md")
        return

    # RAG 评测对比模式
    if args.compare:
        with open(args.compare[0]) as f:
            result_a = json.load(f)
        with open(args.compare[1]) as f:
            result_b = json.load(f)
        report = compare_evaluations(result_a, result_b, args.compare[0], args.compare[1])
        print(report)
        with open("evaluation_comparison.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("\n对比报告已保存到 evaluation_comparison.md")
        return

    dataset = load_golden_dataset()
    print(f"数据集: {len(dataset)} 条, 模式: {args.mode}")
    if args.limit > 0:
        print(f"限制: {args.limit} 条")

    t_start = time.time()

    if args.mode == "retrieval":
        result = run_retrieval_evaluation(dataset, args.limit)
    elif args.mode == "pipeline":
        result = run_pipeline_evaluation(dataset, args.limit)
    elif args.mode == "e2e":
        result = run_e2e_evaluation(dataset, args.limit)
    else:
        print(f"未知模式: {args.mode}")
        return

    result["total_time_seconds"] = round(time.time() - t_start, 1)

    print(f"\n===== 评估结果 [{args.mode}] =====")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
    print(f"\n样本数: {result['sample_count']}")
    print(f"总耗时: {result['total_time_seconds']}s")

    if result.get("by_query_type"):
        print(f"\n===== 按问题类型 =====")
        for qt, m in result["by_query_type"].items():
            print(f"  [{qt}] ", end="")
            print(", ".join(f"{k}={v}" for k, v in m.items()))

    if result.get("latency"):
        print(f"\n===== 延迟统计 (ms) =====")
        lat = result["latency"]
        if isinstance(lat.get("total"), dict):
            for stage in ["retrieval", "generation", "total"]:
                if stage in lat:
                    s = lat[stage]
                    print(f"  {stage}: avg={s['avg']}, p50={s['p50']}, p95={s['p95']}, max={s['max']}")
        else:
            print(f"  avg={lat['avg']}, p50={lat['p50']}, p95={lat['p95']}, max={lat['max']}")

    if result.get("routing_accuracy"):
        rt = result["routing_accuracy"]
        print(f"\n===== 路由准确率 =====")
        print(f"  accuracy: {rt['accuracy']} ({rt['correct']}/{rt['total']})")
        for qt, m in rt.get("by_query_type", {}).items():
            print(f"  [{qt}] {m['accuracy']} ({m['correct']}/{m['total']})")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {args.output}")

    generate_charts(result)


if __name__ == "__main__":
    main()
