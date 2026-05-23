import requests
import os
from dotenv import load_dotenv

# 加载你的 .env 配置文件
load_dotenv()

api_key = os.getenv("AMAP_API_KEY")
api_url = os.getenv("AMAP_WEATHER_API")

if not api_key or api_key == "your_amap_api_key":
    print("❌ 请先在 .env 文件里配置正确的 AMAP_API_KEY")
else:
    print(f"正在测试 API Key: {api_key[:10]}...")

    # 测试查询北京天气
    params = {
        "key": api_key,
        "city": "110000"  # 北京的城市编码
    }

    try:
        response = requests.get(api_url, params=params, timeout=5)
        result = response.json()

        if result.get("status") == "1":
            print("✅ API 有效！")
            weather = result["lives"][0]
            print(f"当前城市: {weather['city']}")
            print(f"当前天气: {weather['weather']}")
            print(f"当前温度: {weather['temperature']}°C")
        else:
            print(f"❌ API 无效！")
            print(f"错误信息: {result.get('info')}")
            print(f"错误代码: {result.get('infocode')}")

    except Exception as e:
        print(f"❌ 请求失败: {e}")