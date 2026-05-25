"""RAG 评估指标计算（基于 Ragas 框架）。"""
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
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
    """计算 context_precision, faithfulness, answer_relevancy。"""
    dataset = Dataset.from_list(data_samples)
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, faithfulness, answer_relevancy],
        llm=_get_llm(),
        embeddings=_get_embeddings(),
    )
    return {k: round(float(v), 4) for k, v in result.items()}
