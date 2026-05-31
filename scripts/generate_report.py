#!/usr/bin/env python3
"""评测报告生成器。

生成包含指标卡片、雷达图、柱状图、路由准确率矩阵、延迟分布的 HTML 报告。

用法:
    python scripts/generate_report.py evaluation_result.json
    python scripts/generate_report.py --compare result_a.json result_b.json
"""
import argparse, json, sys, io, base64
from pathlib import Path


def _fig_to_base64(fig) -> str:
    """将 matplotlib figure 转为 base64 PNG。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _generate_radar_chart(metrics: dict, title: str = "RAGAS Metrics") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(metrics.keys())
    values = [v if (v == v and v is not None) else 0 for v in metrics.values()]
    n = len(labels)
    if n < 3 or all(v == 0 for v in values):
        return ""

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles_closed, values_closed, alpha=0.25, color="#4FC08D")
    ax.plot(angles_closed, values_closed, linewidth=2, color="#4FC08D")
    ax.set_xticks(angles)
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20, fontsize=12)

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _generate_bar_chart(by_type: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not by_type:
        return ""

    qtypes = list(by_type.keys())
    metric_names = list(list(by_type.values())[0].keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(qtypes)
    colors = ["#4FC08D", "#3776AB", "#F56C6C", "#E6A23C", "#9B59B6", "#1ABC9C"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, qt in enumerate(qtypes):
        vals = [by_type[qt].get(m, 0) or 0 for m in metric_names]
        ax.bar(x + i * width, vals, width, label=qt, color=colors[i % len(colors)])

    ax.set_xticks(x + width * (len(qtypes) - 1) / 2)
    ax.set_xticklabels([m.replace("_", "\n") for m in metric_names], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Metrics by Query Type", fontsize=12)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _generate_latency_chart(latency: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not latency:
        return ""

    # 兼容 e2e 模式的嵌套结构
    if "total" in latency and isinstance(latency["total"], dict):
        data = latency["total"]
    else:
        data = latency

    labels = [k for k in ["avg", "p50", "p95", "max", "min"] if k in data]
    values = [data[k] for k in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#4FC08D", "#3776AB", "#F56C6C", "#E6A23C", "#9B59B6"][:len(labels)])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Distribution", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"{val:.0f}ms", ha="center", fontsize=9)

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _generate_routing_matrix(routing: dict) -> str:
    """生成路由准确率 HTML 表格。"""
    if not routing:
        return ""

    by_type = routing.get("by_query_type", {})
    if not by_type:
        return ""

    rows = []
    for qt, m in sorted(by_type.items()):
        acc = m.get("accuracy", 0)
        color = "#4FC08D" if acc >= 0.8 else "#E6A23C" if acc >= 0.6 else "#F56C6C"
        rows.append(f"""
        <tr>
            <td>{qt}</td>
            <td>{m.get('correct', 0)}/{m.get('total', 0)}</td>
            <td><span style="color:{color};font-weight:bold">{acc:.1%}</span></td>
        </tr>""")

    overall = routing.get("accuracy", 0)
    overall_color = "#4FC08D" if overall >= 0.8 else "#E6A23C" if overall >= 0.6 else "#F56C6C"

    return f"""
    <table class="data-table">
        <thead><tr><th>Query Type</th><th>Correct/Total</th><th>Accuracy</th></tr></thead>
        <tbody>
            {''.join(rows)}
            <tr style="font-weight:bold;border-top:2px solid #333">
                <td>Overall</td>
                <td>{routing.get('correct', 0)}/{routing.get('total', 0)}</td>
                <td><span style="color:{overall_color}">{overall:.1%}</span></td>
            </tr>
        </tbody>
    </table>"""


def _topology_section(result: dict) -> str:
    """生成图谱拓扑 HTML 段落。"""
    stats = result.get("topology", {})
    if not stats:
        return ""

    chart_b64 = _generate_topology_chart(stats)

    pred_rows = ""
    for p, cnt in stats.get("predicate_distribution", {}).items():
        pred_rows += f"<tr><td>{p}</td><td>{cnt}</td></tr>"

    cards = "".join(
        f'<div class="metric-card"><div class="metric-value">{stats.get(k, 0)}</div><div class="metric-label">{l}</div></div>'
        for k, l in [
            ("total_nodes", "Total Nodes"),
            ("total_edges", "Total Edges"),
            ("orphan_nodes", "Orphan Nodes"),
            ("orphan_rate", "Orphan Rate"),
            ("avg_degree", "Avg Degree"),
        ]
    )

    return f"""
    <h2>Graph Topology</h2>
    <div class="metric-cards">{cards}</div>
    {f'<div class="chart-row"><div class="chart-box"><img src="data:image/png;base64,{chart_b64}"></div></div>' if chart_b64 else ''}
    {f'<h3>Predicate Distribution</h3><table class="data-table"><thead><tr><th>Predicate</th><th>Count</th></tr></thead><tbody>{pred_rows}</tbody></table>' if pred_rows else ''}
    """


def _generate_topology_chart(stats: dict) -> str:
    """生成图谱拓扑类型分布柱状图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    type_dist = stats.get("type_distribution", {})
    if not type_dist:
        return ""

    types = list(type_dist.keys())
    counts = list(type_dist.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(types, counts, color="#4FC08D")
    ax.set_ylabel("Count")
    ax.set_title("Entity Type Distribution", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=30, ha="right")

    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                str(val), ha="center", fontsize=9)

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _generate_topology_comparison_chart(stats_a: dict, stats_b: dict, label_a: str, label_b: str) -> str:
    """生成图谱拓扑前后对比分组柱状图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    types = sorted(set(list(stats_a.get("type_distribution", {}).keys()) +
                       list(stats_b.get("type_distribution", {}).keys())))
    if not types:
        return ""

    vals_a = [stats_a.get("type_distribution", {}).get(t, 0) for t in types]
    vals_b = [stats_b.get("type_distribution", {}).get(t, 0) for t in types]

    x = np.arange(len(types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, vals_a, width, label=label_a, color="#3776AB")
    ax.bar(x + width / 2, vals_b, width, label=label_b, color="#F56C6C")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Entity Type Distribution: Before vs After", fontsize=12)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_html_report(result: dict, output_path: str = "evaluation_report.html") -> str:
    """生成单次评测 HTML 报告。"""
    mode = result.get("mode", "unknown")
    metrics = result.get("metrics", {})
    by_type = result.get("by_query_type", {})
    latency = result.get("latency", {})
    routing = result.get("routing_accuracy", {})
    sample_count = result.get("sample_count", 0)
    total_time = result.get("total_time_seconds", 0)

    radar_b64 = _generate_radar_chart(metrics)
    bar_b64 = _generate_bar_chart(by_type)
    latency_b64 = _generate_latency_chart(latency)
    routing_html = _generate_routing_matrix(routing)

    metric_cards = "".join(
        f"""<div class="metric-card">
            <div class="metric-value">{v:.4f}</div>
            <div class="metric-label">{k.replace('_', ' ').title()}</div>
        </div>"""
        for k, v in metrics.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ragent AI Evaluation Report [{mode}]</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #4FC08D; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
.metric-cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }}
.metric-card {{ background: white; border-radius: 12px; padding: 20px; min-width: 140px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.metric-value {{ font-size: 28px; font-weight: 700; color: #4FC08D; }}
.metric-label {{ font-size: 12px; color: #666; margin-top: 4px; }}
.chart-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
.chart-box {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.chart-box img {{ max-width: 100%; height: auto; }}
.data-table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.data-table th, .data-table td {{ padding: 10px 16px; text-align: left; border-bottom: 1px solid #eee; }}
.data-table th {{ background: #f8f9fa; font-weight: 600; }}
.info-bar {{ display: flex; gap: 20px; margin: 10px 0; color: #666; font-size: 14px; }}
.info-bar span {{ background: white; padding: 6px 12px; border-radius: 6px; }}
</style>
</head>
<body>
<div class="container">
    <h1>Ragent AI Evaluation Report</h1>
    <div class="info-bar">
        <span>Mode: <strong>{mode}</strong></span>
        <span>Samples: <strong>{sample_count}</strong></span>
        <span>Total Time: <strong>{total_time}s</strong></span>
    </div>

    <h2>Metrics Overview</h2>
    <div class="metric-cards">{metric_cards}</div>

    {f'<div class="chart-row"><div class="chart-box"><img src="data:image/png;base64,{radar_b64}"></div></div>' if radar_b64 else ''}

    {f'<h2>Metrics by Query Type</h2><div class="chart-row"><div class="chart-box"><img src="data:image/png;base64,{bar_b64}"></div></div>' if bar_b64 else ''}

    {f'<h2>Latency Distribution</h2><div class="chart-row"><div class="chart-box"><img src="data:image/png;base64,{latency_b64}"></div></div>' if latency_b64 else ''}

    {f'<h2>Routing Accuracy</h2>{routing_html}' if routing_html else ''}

    {_topology_section(result) if result.get("topology") else ''}

    <footer style="margin-top:40px;padding-top:20px;border-top:1px solid #ddd;color:#999;font-size:12px;">
        Generated by Ragent AI Evaluation System
    </footer>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def generate_compare_html_report(result_a: dict, result_b: dict, label_a: str, label_b: str,
                                  output_path: str = "evaluation_comparison.html") -> str:
    """生成对比评测 HTML 报告。"""
    metrics_a = result_a.get("metrics", {})
    metrics_b = result_b.get("metrics", {})

    # 双雷达图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
    n = len(labels)
    if n >= 3:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        vals_a = [metrics_a.get(l, 0) for l in labels] + [metrics_a.get(labels[0], 0)]
        vals_b = [metrics_b.get(l, 0) for l in labels] + [metrics_b.get(labels[0], 0)]
        angles_closed = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles_closed, vals_a, alpha=0.15, color="#3776AB")
        ax.plot(angles_closed, vals_a, linewidth=2, color="#3776AB", label=label_a)
        ax.fill(angles_closed, vals_b, alpha=0.15, color="#F56C6C")
        ax.plot(angles_closed, vals_b, linewidth=2, color="#F56C6C", label=label_b)
        ax.set_xticks(angles)
        ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("RAGAS Comparison", pad=20, fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
        radar_b64 = _fig_to_base64(fig)
        plt.close(fig)
    else:
        radar_b64 = ""

    # Diff 表格
    diff_rows = []
    for k in labels:
        va = metrics_a.get(k, 0)
        vb = metrics_b.get(k, 0)
        diff = vb - va
        pct = (diff / va * 100) if va else 0
        color = "#4FC08D" if diff > 0 else "#F56C6C" if diff < 0 else "#666"
        arrow = "+" if diff > 0 else ""
        diff_rows.append(f"""
        <tr>
            <td>{k.replace('_', ' ').title()}</td>
            <td>{va:.4f}</td>
            <td>{vb:.4f}</td>
            <td style="color:{color};font-weight:bold">{arrow}{diff:.4f} ({arrow}{pct:.1f}%)</td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ragent AI Evaluation Comparison</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #4FC08D; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
.chart-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
.chart-box {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.data-table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.data-table th, .data-table td {{ padding: 10px 16px; text-align: left; border-bottom: 1px solid #eee; }}
.data-table th {{ background: #f8f9fa; font-weight: 600; }}
</style>
</head>
<body>
<div class="container">
    <h1>Ragent AI Evaluation Comparison</h1>
    <p><strong>{label_a}</strong> vs <strong>{label_b}</strong></p>

    <h2>Metrics Comparison</h2>
    <table class="data-table">
        <thead><tr><th>Metric</th><th>{label_a}</th><th>{label_b}</th><th>Change</th></tr></thead>
        <tbody>{''.join(diff_rows)}</tbody>
    </table>

    {f'<div class="chart-row"><div class="chart-box"><img src="data:image/png;base64,{radar_b64}"></div></div>' if radar_b64 else ''}

    {_topology_compare_section(result_a, result_b, label_a, label_b)}

    <footer style="margin-top:40px;padding-top:20px;border-top:1px solid #ddd;color:#999;font-size:12px;">
        Generated by Ragent AI Evaluation System
    </footer>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def _topology_compare_section(result_a: dict, result_b: dict, label_a: str, label_b: str) -> str:
    """生成图谱拓扑对比 HTML 段落。"""
    stats_a = result_a.get("topology", {})
    stats_b = result_b.get("topology", {})
    if not stats_a or not stats_b:
        return ""

    chart_b64 = _generate_topology_comparison_chart(stats_a, stats_b, label_a, label_b)

    overview_keys = [
        ("total_nodes", "Total Nodes"),
        ("total_edges", "Total Edges"),
        ("orphan_nodes", "Orphan Nodes"),
        ("orphan_rate", "Orphan Rate"),
        ("avg_degree", "Avg Degree"),
    ]
    diff_rows = ""
    for key, label in overview_keys:
        va = stats_a.get(key, 0)
        vb = stats_b.get(key, 0)
        diff = vb - va
        color = "#4FC08D" if diff > 0 else "#F56C6C" if diff < 0 else "#666"
        if key in ("orphan_rate",):
            diff_rows += f'<tr><td>{label}</td><td>{va:.4f}</td><td>{vb:.4f}</td><td style="color:{color};font-weight:bold">{diff:+.4f}</td></tr>'
        else:
            diff_rows += f'<tr><td>{label}</td><td>{va}</td><td>{vb}</td><td style="color:{color};font-weight:bold">{diff:+d}</td></tr>'

    return f"""
    <h2>Graph Topology Comparison</h2>
    <table class="data-table">
        <thead><tr><th>Metric</th><th>{label_a}</th><th>{label_b}</th><th>Change</th></tr></thead>
        <tbody>{diff_rows}</tbody>
    </table>
    {f'<div class="chart-row"><div class="chart-box"><img src="data:image/png;base64,{chart_b64}"></div></div>' if chart_b64 else ''}
    """


def main():
    parser = argparse.ArgumentParser(description="生成评测 HTML 报告")
    parser.add_argument("input", help="评测结果 JSON 文件")
    parser.add_argument("--compare", help="对比模式的第二个 JSON 文件")
    parser.add_argument("--output", default=None, help="输出 HTML 文件路径")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        result_a = json.load(f)

    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            result_b = json.load(f)
        output = args.output or "evaluation_comparison.html"
        path = generate_compare_html_report(result_a, result_b, args.input, args.compare, output)
    else:
        output = args.output or "evaluation_report.html"
        path = generate_html_report(result_a, output)

    print(f"报告已生成: {path}")


if __name__ == "__main__":
    main()
