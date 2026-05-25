"""Qwen-VL 视觉描述：为图片/表格生成 Markdown 文本描述。"""
import os
import requests

VLM_API_KEY = os.getenv("ARK_API_KEY")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl-plus")
VLM_BASE_URL = os.getenv("VLM_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1")


def describe_image(image_url: str) -> dict:
    """调用 Qwen-VL 为图片生成文本描述。"""
    if not VLM_API_KEY:
        return {"description": "", "status": "error:no_api_key"}

    try:
        response = requests.post(
            f"{VLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {VLM_API_KEY}"},
            json={
                "model": VLM_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请详细描述这张图片/图表中的内容，包括关键数据、趋势和结论。用中文回答，控制在150字以内。"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
                "max_tokens": 300,
                "temperature": 0.1,
            },
            timeout=30,
        )
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return {"description": content, "status": "ok"}
        return {"description": "", "status": f"error:{response.status_code}"}
    except Exception as e:
        return {"description": "", "status": f"error:{str(e)[:100]}"}
