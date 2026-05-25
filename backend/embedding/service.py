"""嵌入式服务模块

为文本生成两类向量：
- 稠密向量：基于通义千问 API 的语义向量
- 稀疏向量：基于 BM25 算法的关键词权重向量
"""
import os
import math
import httpx
import re
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self):
        # 通义千问向量模型配置稠密向量（通义千问）：从环境变量读取 API 密钥，配置阿里云通义千问向量模型的接口地址、模型名称
        self.api_key = os.getenv("ARK_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        self.model = os.getenv("EMBEDDER", "text-embedding-v1")

        # BM25 稀疏向量参数 初始化 BM25 算法的核心参数（k1=1.5、b=0.75，行业通用默认值），以及统计所需的变量（文档频率、总文档数、平均文档长度、词汇表）
        self.k1 = 1.5
        self.b = 0.75
        self.doc_freq = Counter()  # 文档频率
        self.total_docs = 0        # 总文档数
        self.avg_doc_len = 0       # 平均文档长度
        self.vocab = {}            # 词汇表

    def fit_corpus(self, texts: list[str]):
        """
        拟合语料库，计算IDF所需统计量（给BM25用）
        ✅ 修复缺失的 fit_corpus 方法
        为 BM25 算法预处理语料库，计算 IDF（逆文档频率）所需的统计信息：
遍历输入的文本列表，对每个文本分词后，统计「每个词在多少文档中出现（文档频率 doc_freq）」「总文档数」「平均文档长度」。
是生成稀疏向量的前提（必须先拟合语料库，才能计算合理的 IDF 值）。
        """
        doc_lengths = []
        self.doc_freq.clear()
        self.total_docs = len(texts)

        for text in texts:
            tokens = self.tokenize(text)
            doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freq[t] += 1

        self.avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

    def tokenize(self, text: str) -> list[str]:
        """简单分词：中文单字 + 英文单词
        实现轻量级的多语言分词逻辑，适配中文 + 英文：
中文：按单字拆分（正则匹配 [\u4e00-\u9fff]）；
英文：按单词拆分（正则匹配 [a-zA-Z]+）；
统一转为小写，忽略非中 / 英文的字符（如标点、数字）
        """
        text = text.lower()
        tokens = []
        # 匹配中文
        ch_pattern = re.compile(r"[\u4e00-\u9fff]")
        # 匹配英文单词
        en_pattern = re.compile(r"[a-zA-Z]+")

        i = 0
        while i < len(text):
            c = text[i]
            if ch_pattern.match(c):
                tokens.append(c)
                i += 1
            elif en_pattern.match(c):
                match = en_pattern.match(text[i:])
                if match:
                    tokens.append(match.group())
                    i += len(match.group())
            else:
                i += 1
        return tokens

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """调用通义千问生成稠密向量（含重试机制）"""
        import time

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": {"texts": texts},
            "parameters": {"text_type": "document"}
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = httpx.post(self.base_url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                result = response.json()
                return [item["embedding"] for item in result["output"]["embeddings"]]
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise

    def get_sparse_embedding(self, text: str) -> dict[int, float]:
        """生成单条稀疏向量（BM25）
        分为单条生成（get_sparse_embedding）和批量生成（get_sparse_embeddings）
        步骤 1：对文本分词，统计词频（TF）、文档长度；
步骤 2：计算 IDF：基于拟合语料库得到的文档频率，计算词的逆文档频率（衡量词的稀缺性）；
步骤 3：计算 BM25 得分：结合 TF、IDF、文档长度、平均文档长度，以及 BM25 的k1/b参数，得到每个词的权重；
步骤 4：构建稀疏向量：将词映射为唯一整数 ID（词汇表vocab），仅保留权重 > 0 的词，最终返回「ID - 权重」的字典（稀疏表示，节省空间）。"""
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)
        sparse_vec = {}

        for token, freq in tf.items():
            df = self.doc_freq.get(token, 0)
            # IDF
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            # TF
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1))
            score = idf * (numerator / denominator)
            if score > 0:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                sparse_vec[self.vocab[token]] = score
        return sparse_vec

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict[int, float]]:
        """批量生成稀疏向量"""
        return [self.get_sparse_embedding(t) for t in texts]