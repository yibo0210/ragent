"""动态模型路由：按任务复杂度分配算力。"""
import os
from langchain_openai import ChatOpenAI

MODEL_TURBO = os.getenv("MODEL_TURBO", "qwen-turbo")
MODEL_PLUS = os.getenv("MODEL", "qwen-plus")
MODEL_MAX = os.getenv("MODEL_MAX", "qwen-max")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("ARK_API_KEY")

ROUTE_MAP = {
    "supervisor": MODEL_PLUS,   # Supervisor needs structured output — turbo is too weak
    "direct_answer": MODEL_TURBO,
    "web_searcher": MODEL_PLUS,
    "rag_specialist": MODEL_PLUS,
    "synthesize": MODEL_PLUS,
    "data_analyst": os.getenv("DATA_ANALYST_MODEL", MODEL_PLUS),
    "local_graph_search": MODEL_PLUS,
    "global_graph_search": MODEL_PLUS,
    "complex_graph_reasoning": MODEL_MAX,
}


def get_model_for_agent(agent_name: str):
    """根据 Agent 角色获取对应模型实例。"""
    from langchain.chat_models import init_chat_model
    model_name = ROUTE_MAP.get(agent_name, MODEL_PLUS)
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.0,
        max_tokens=8192,
        timeout=120,
    )


def is_lightweight_task(routes: list[str]) -> bool:
    """判断是否轻量任务（闲聊、简单问答）。"""
    return set(routes).issubset({"direct_answer"})
