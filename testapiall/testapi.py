from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("ARK_API_KEY"),
    model=os.getenv("MODEL"),
    base_url=os.getenv("BASE_URL"),
    temperature=0.3
)

# 简单提问测试
resp = llm.invoke("一句话介绍你是谁")
print("模型返回：", resp.content)