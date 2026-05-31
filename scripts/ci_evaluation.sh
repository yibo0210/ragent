#!/bin/bash
# CI 评测脚本 — 快速检索模式，检查指标不低于阈值
# 用法: bash scripts/ci_evaluation.sh

set -e

echo "===== Ragent AI CI Evaluation ====="
echo "Mode: retrieval (fast, no LLM generation)"

python scripts/run_evaluation.py --mode retrieval --limit 20 --output ci_result.json

# 读取指标
PRECISION=$(python -c "import json; r=json.load(open('ci_result.json')); print(r['metrics']['context_precision'])")
FAITHFULNESS=$(python -c "import json; r=json.load(open('ci_result.json')); print(r['metrics']['faithfulness'])")
RELEVANCY=$(python -c "import json; r=json.load(open('ci_result.json')); print(r['metrics']['answer_relevancy'])")

echo ""
echo "===== Results ====="
echo "  context_precision: $PRECISION"
echo "  faithfulness:      $FAITHFULNESS"
echo "  answer_relevancy:  $RELEVANCY"

# 阈值检查
PASS=true

python -c "
import sys
prec = float('$PRECISION')
faith = float('$FAITHFULNESS')
rel = float('$RELEVANCY')

thresholds = {
    'context_precision': (prec, 0.6),
    'faithfulness': (faith, 0.7),
    'answer_relevancy': (rel, 0.6),
}

for name, (val, threshold) in thresholds.items():
    if val < threshold:
        print(f'  FAIL: {name}={val:.4f} < {threshold}')
        sys.exit(1)
    else:
        print(f'  PASS: {name}={val:.4f} >= {threshold}')
"

echo ""
echo "===== All thresholds passed ====="

# 清理
rm -f ci_result.json
