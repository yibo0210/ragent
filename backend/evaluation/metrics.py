"""RAG 评估指标计算（基于 Ragas 框架）。"""
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain.chat_models import init_chat_model

LLM = init_chat_model(
    model=os.getenv("GRADE_MODEL", os.getenv("MODEL", "qwen-plus")),
    model_provider="openai",
    api_key=os.getenv("ARK_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0.0,
)


def compute_ragas_metrics(data_samples: list[dict]) -> dict:
    """计算 context_precision, faithfulness, answer_relevancy。"""
    dataset = Dataset.from_list(data_samples)
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, faithfulness, answer_relevancy],
        llm=LLM,
    )
    return {k: round(float(v), 4) for k, v in result.items()}
