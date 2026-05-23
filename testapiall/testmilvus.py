from pymilvus import connections
try:
    connections.connect(host="127.0.0.1", port=19530)
    print("✅ 连接成功！！！")
except Exception as e:
    print("错误：", e)