"""RAG 评估指标计算（基于 Ragas 框架）。

支持三种评测模式：
- retrieval: 仅评测检索质量（answer = ground_truth）
- pipeline: 评测 RAG pipeline 检索质量（answer = ground_truth，但走完整 pipeline）
- e2e: 端到端评测（LLM 真实生成 answer）

注意：使用 ragas 0.2.x（<0.3.0），0.4.x 的 prompt 格式与 DashScope API 不兼容。
DashScope 的 contents 字段格式限制导致 answer_relevancy 和 context_recall 可能返回 NaN。
"""
import os
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

_LLM = None
_EMBEDDINGS = None


def _get_llm():
    global _LLM
    if _LLM is None:
        api_key = os.getenv("ARK_API_KEY")
        base_url = os.getenv("BASE_URL")
        _LLM = ChatOpenAI(
            model=os.getenv("GRADE_MODEL", os.getenv("MODEL", "qwen-plus")),
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            model_kwargs={"extra_body": {"enable_thinking": False}},
        )
    return _LLM


def _get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        api_key = os.getenv("ARK_API_KEY")
        base_url = os.getenv("BASE_URL")
        _EMBEDDINGS = OpenAIEmbeddings(
            model=os.getenv("EMBEDDER", "text-embedding-v1"),
            api_key=api_key,
            base_url=base_url,
        )
    return _EMBEDDINGS


def compute_ragas_metrics(data_samples: list[dict]) -> dict:
    """计算 context_precision, context_recall, faithfulness, answer_relevancy。"""
    dataset = Dataset.from_list(data_samples)
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=_get_llm(),
        embeddings=_get_embeddings(),
    )
    # ragas 0.2.x 返回 dict，0.4.x 返回 EvaluationResult 对象
    if isinstance(result, dict):
        scores = result
    elif hasattr(result, "_scores_dict"):
        scores = result._scores_dict
    elif hasattr(result, "to_pandas"):
        df = result.to_pandas()
        scores = {col: list(df[col]) for col in df.columns if col not in ("question", "contexts", "answer", "ground_truth")}
    else:
        scores = {k: v for k, v in result.items()} if hasattr(result, "items") else {}
    # 每个指标可能是 list（per-sample）或 scalar，统一取均值
    final = {}
    for k, v in scores.items():
        if isinstance(v, list):
            numeric = [x for x in v if x is not None and isinstance(x, (int, float))]
            final[k] = round(sum(numeric) / len(numeric), 4) if numeric else 0.0
        else:
            final[k] = round(float(v), 4)
    return final


def generate_answer(question: str, contexts: list[str]) -> tuple[str, float]:
    """调用 Worker LLM 基于 contexts 生成回答。返回 (answer, latency_ms)。"""
    from backend.agent.orchestrator import _get_worker_model, RAG_SPECIALIST_PROMPT
    from langchain_core.messages import HumanMessage

    model = _get_worker_model()
    context_text = "\n\n---\n\n".join(contexts) if contexts else "未检索到相关文档"
    prompt = f"{RAG_SPECIALIST_PROMPT}\n\n## 检索到的文档\n\n{context_text}\n\n## 用户问题\n\n{question}"

    t0 = time.time()
    response = model.invoke([HumanMessage(content=prompt)])
    latency_ms = (time.time() - t0) * 1000
    answer = response.content if hasattr(response, "content") else str(response)
    return answer, latency_ms


def evaluate_routing_accuracy(dataset: list[dict]) -> dict:
    """评测 Supervisor 路由准确率。

    仅对含 expected_agent 字段的条目进行评测。
    复用 supervisor_node 的路由逻辑但不执行 Worker。
    """
    from backend.agent.orchestrator import _get_supervisor_model, SUPERVISOR_SYSTEM_PROMPT
    from langchain_core.messages import HumanMessage
    import re, json

    model = _get_supervisor_model()
    valid_agents = {
        "rag_specialist", "web_searcher", "direct_answer",
        "data_analyst", "local_graph_search", "global_graph_search",
        "multimodal_specialist",
    }

    results = []
    for item in dataset:
        expected = item.get("expected_agent")
        if not expected:
            continue

        question = item["question"]
        route_prompt = SUPERVISOR_SYSTEM_PROMPT + (
            '\n\n请严格输出JSON格式，不要包含其他内容：\n'
            '{"routes": ["agent_name"], "reason": "选择原因"}\n'
            f'\n用户问题：{question}'
        )

        t0 = time.time()
        try:
            response = model.invoke([HumanMessage(content=route_prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            json_match = re.search(r'\{[^{}]*"routes"\s*:\s*\[[^\]]*\][^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                routes = data.get("routes", [])
            else:
                routes = ["rag_specialist"]
        except Exception:
            routes = ["rag_specialist"]
        latency_ms = (time.time() - t0) * 1000

        routes = [r for r in routes if r in valid_agents]
        if not routes:
            routes = ["direct_answer"]

        predicted = routes[0]
        correct = predicted == expected
        results.append({
            "id": item.get("id", ""),
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "query_type": item.get("query_type", "unknown"),
            "latency_ms": round(latency_ms, 1),
        })

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = round(correct_count / total, 4) if total > 0 else 0.0

    by_type = {}
    for r in results:
        qt = r["query_type"]
        if qt not in by_type:
            by_type[qt] = {"total": 0, "correct": 0}
        by_type[qt]["total"] += 1
        if r["correct"]:
            by_type[qt]["correct"] += 1
    for qt in by_type:
        t = by_type[qt]
        t["accuracy"] = round(t["correct"] / t["total"], 4) if t["total"] > 0 else 0.0

    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct_count,
        "by_query_type": by_type,
        "details": results,
    }


def evaluate_security(dataset: list[dict], run_fn) -> dict:
    """Test that low-privilege users cannot access high-privilege content."""
    security_tests = [d for d in dataset if d.get("query_type") == "privilege_escalation"]
    results = {"total": len(security_tests), "passed": 0, "failed": 0, "details": []}
    for test in security_tests:
        response = run_fn(test["question"], test.get("test_role", "viewer"))
        refused = ("未找到" in response or "无权" in response or
                   "not found" in response.lower() or "unauthorized" in response.lower())
        if refused:
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["details"].append({
            "id": test["id"],
            "question": test["question"],
            "refused": refused,
            "response_preview": response[:200],
        })
    results["security_score"] = results["passed"] / max(results["total"], 1)
    return results
