# Ragent AI v7.0 — 多模态大升级 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 攻克 PDF 复杂版面解析（表格/图表不截断），引入 VLM 视觉描述入图，升级为四通道 RRF 融合（+视觉召回），新增 Multimodal Specialist Agent。

**Architecture:** 解析层（Marker/Unstructured 版面分析）→ 抽取层（Qwen-VL 图表描述 + Neo4j ImageNode/TableNode）→ 检索层（四通道 RRF）→ 路由层（Multimodal Specialist Agent）。

**Tech Stack:** Marker · Unstructured.io · Qwen-VL · MinIO · Neo4j ImageNode/TableNode · 4-Channel RRF

---

## 文件结构概览

```
新增文件 (8):
  backend/documents/layout_analyzer.py      # 版面分析（Marker/Unstructured）
  backend/documents/media_extractor.py      # 图片/表格截取+MinIO上传
  backend/documents/vlm_descriptor.py       # Qwen-VL 图表描述生成
  backend/rag/visual_retriever.py           # 视觉通道召回（文本→图片描述）
  backend/agent/multimodal_specialist.py    # 多模态专家 Agent
  tests/test_layout_analyzer.py             # 版面分析单元测试
  scripts/migrate_graph_schema.py           # Neo4j Schema 迁移脚本
  scripts/run_multimodal_ingest.py          # 多模态文档批量导入脚本

修改文件 (14):
  backend/documents/loader.py               # +版面分析调用, +图片/表格分流
  backend/storage/models.py                 # +associated_media_urls 字段
  backend/storage/graph_schema.py           # +ImageNode/TableNode 约束
  backend/documents/graph_extractor.py      # +VLM 描述抽取
  backend/storage/graph_ingestion.py        # +ImageNode/TableNode MERGE
  backend/storage/graph_client.py           # +multimodal Cypher 查询
  backend/rag/utils.py                      # 三通道→四通道 RRF
  backend/rag/graph_retriever.py            # +visual_search 函数
  backend/agent/orchestrator.py             # +multimodal_specialist 节点, +Supervisor路由
  backend/agent/tools.py                    # +emit_multimodal_step
  backend/schemas.py                        # +MultimodalChunk, ImageDescription
  scripts/grid_search_rrf.py               # 三权重→四权重
  docker-compose.yml                        # MinIO bucket 初始化
  .env.example                              # +VLM/Marker 配置
```

---

## Phase 1: 解析层 — 版面分析

### Task 1.1: 版面分析器 — Marker/Unstructured 集成

**Files:**
- Create: `backend/documents/layout_analyzer.py`
- Modify: `backend/documents/loader.py`

- [ ] **Step 1: 实现版面分析器**

```python
# backend/documents/layout_analyzer.py
"""文档版面分析：使用 Marker 识别标题/段落/表格/图片区域。"""
import os
from pathlib import Path

LAYOUT_ENABLED = os.getenv("LAYOUT_ANALYSIS_ENABLED", "true").lower() != "false"


def analyze_pdf_layout(file_path: str) -> list[dict]:
    """分析 PDF 版面，返回元素列表 [{type, bbox, page_number}]。"""
    if not LAYOUT_ENABLED:
        return _fallback_layout(file_path)

    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(file_path)
        elements = []
        for block in rendered.blocks:
            elements.append({
                "type": block.block_type,
                "text": block.raw_text() if hasattr(block, "raw_text") else "",
                "page_number": block.page_num,
                "bbox": block.bbox if hasattr(block, "bbox") else None,
            })
        return elements
    except Exception as e:
        print(f"[LAYOUT] Marker failed, falling back: {e}")
        return _fallback_layout(file_path)


def _fallback_layout(file_path: str) -> list[dict]:
    """降级方案：整页作为段落处理。"""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return [
        {"type": "paragraph", "text": d.page_content,
         "page_number": d.metadata.get("page", 0)}
        for d in docs
    ]


def is_visual_element(elem_type: str) -> bool:
    """判断是否为图片/表格等可视化元素。"""
    return elem_type in ("table", "picture", "figure", "image")
```

- [ ] **Step 2: 修改 DocumentLoader — 分流文本与图片**

在 `backend/documents/loader.py` 的 `load_document` 方法中，PDF 路径改用版面分析:

```python
def load_document(self, file_path: str, filename: str) -> list[dict]:
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        from backend.documents.layout_analyzer import analyze_pdf_layout, is_visual_element
        elements = analyze_pdf_layout(file_path)

        text_docs = []
        media_docs = []
        for elem in elements:
            if is_visual_element(elem["type"]):
                media_docs.append(elem)
            else:
                # 文本元素走原有三级分块
                doc = Document(page_content=elem["text"], metadata={
                    "page": elem["page_number"], "source": filename,
                    "element_type": elem["type"],
                })
                text_docs.append(doc)

        # 文本部分：三级分块
        chunks = self._chunk_documents(text_docs, filename)

        # 图片/表格：提取 + 上传 MinIO
        if media_docs:
            from backend.documents.media_extractor import extract_and_upload
            media_chunks = extract_and_upload(file_path, media_docs, filename)
            chunks.extend(media_chunks)

        return chunks

    # 其他格式保持原有逻辑
    return self._load_other(file_path, filename)
```

- [ ] **Step 3: 验证版面分析**

```bash
uv run python -c "
from backend.documents.layout_analyzer import analyze_pdf_layout
elements = analyze_pdf_layout('data/documents/test.pdf')
types = {}
for e in elements:
    types[e['type']] = types.get(e['type'], 0) + 1
print(f'Element types: {types}')
"
```

- [ ] **Step 4: Commit**

```bash
git add backend/documents/layout_analyzer.py backend/documents/loader.py
git commit -m "feat: add Marker-based PDF layout analysis, separate text from tables/images"
```

---

### Task 1.2: 图片/表格提取 + MinIO 存储

**Files:**
- Create: `backend/documents/media_extractor.py`
- Modify: `backend/storage/models.py` (+associated_media_urls)

- [ ] **Step 1: 实现图片/表格提取与上传**

```python
# backend/documents/media_extractor.py
"""从 PDF 中截取图片/表格并上传 MinIO。"""
import os
import uuid
from pathlib import Path
import fitz  # PyMuPDF

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ragent-media")


def _get_minio_client():
    from minio import Minio
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS,
                 secret_key=MINIO_SECRET, secure=False)


def extract_and_upload(pdf_path: str, media_elements: list[dict],
                       filename: str) -> list[dict]:
    """截取 PDF 中的图片/表格区域，上传 MinIO，返回 chunk 记录。"""
    client = _get_minio_client()
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)

    doc = fitz.open(pdf_path)
    chunks = []

    for elem in media_elements:
        page_num = elem.get("page_number", 0)
        bbox = elem.get("bbox")

        media_id = str(uuid.uuid4())[:12]
        ext = "png"

        # 截取页面区域
        page = doc[page_num]
        if bbox:
            rect = fitz.Rect(*bbox)
            pix = page.get_pixmap(clip=rect, dpi=150)
        else:
            pix = page.get_pixmap(dpi=150)

        img_bytes = pix.tobytes("png")
        object_name = f"{filename}/{media_id}.{ext}"

        client.put_object(MINIO_BUCKET, object_name, img_bytes, len(img_bytes))
        url = f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{object_name}"

        chunks.append({
            "chunk_id": f"{filename}::media::{media_id}",
            "text": elem.get("text", f"[{elem['type']} on page {page_num}]"),
            "filename": filename,
            "file_type": elem["type"],
            "page_number": page_num,
            "chunk_level": 3,  # 作为 L3 叶子
            "associated_media_urls": url,
            "is_media": True,
        })

    doc.close()
    return chunks
```

- [ ] **Step 2: ParentChunk 增加 associated_media_urls 字段**

在 `backend/storage/models.py` 的 `ParentChunk` 类中追加:

```python
associated_media_urls: Mapped[str] = mapped_column(String(2048), default="", nullable=False)
```

- [ ] **Step 3: 添加 minio 到 pyproject.toml**

```toml
"minio>=7.2.0",
"pymupdf>=1.24.0",
"marker-pdf>=1.0.0",
```

- [ ] **Step 4: Commit**

```bash
git add backend/documents/media_extractor.py backend/storage/models.py pyproject.toml
git commit -m "feat: PDF image/table extraction and MinIO upload, add associated_media_urls"
```

---

## Phase 2: 抽取层 — VLM 多模态入图

### Task 2.1: VLM 图表描述生成

**Files:**
- Create: `backend/documents/vlm_descriptor.py`

- [ ] **Step 1: Qwen-VL 图表描述**

```python
# backend/documents/vlm_descriptor.py
"""Qwen-VL 视觉描述：为图片/表格生成 Markdown 文本描述。"""
import os
import base64
import requests

VLM_API_KEY = os.getenv("ARK_API_KEY")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen-vl-plus")
VLM_BASE_URL = os.getenv("VLM_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1")


def describe_image(image_url: str) -> dict:
    """调用 Qwen-VL 为图片生成文本描述。"""
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
```

- [ ] **Step 2: 验证 VLM 调用**

```bash
uv run python -c "
from backend.documents.vlm_descriptor import describe_image
result = describe_image('https://example.com/chart.png')
print(result)
"
```

- [ ] **Step 3: Commit**

```bash
git add backend/documents/vlm_descriptor.py
git commit -m "feat: Qwen-VL image/table description generator"
```

---

### Task 2.2: Neo4j ImageNode/TableNode + 图摄入升级

**Files:**
- Modify: `backend/storage/graph_schema.py` (+ImageNode/TableNode 约束)
- Modify: `backend/storage/graph_ingestion.py` (+multimodal MERGE)

- [ ] **Step 1: 图 Schema 扩展**

```python
# graph_schema.py 新增
"CREATE CONSTRAINT imagenode_name_unique IF NOT EXISTS FOR (n:ImageNode) REQUIRE n.media_id IS UNIQUE",
"CREATE CONSTRAINT tablenode_name_unique IF NOT EXISTS FOR (n:TableNode) REQUIRE n.media_id IS UNIQUE",
```

- [ ] **Step 2: 多模态节点写入**

在 `graph_ingestion.py` 中新增:

```python
def ingest_multimodal_nodes(
    media_chunks: list[dict], vlm_descriptions: dict
) -> dict:
    """将图片/表格节点和 VLM 描述写入 Neo4j。"""
    stats = {"images": 0, "tables": 0, "relations": 0}

    for chunk in media_chunks:
        media_id = chunk.get("chunk_id", "")
        media_type = chunk.get("file_type", "image")
        url = chunk.get("associated_media_urls", "")
        desc = vlm_descriptions.get(media_id, {}).get("description", "")
        label = "ImageNode" if media_type in ("picture", "figure", "image") else "TableNode"

        write_cypher(f"""
            MERGE (n:{label} {{media_id: $media_id}})
            ON CREATE SET n.url = $url, n.description = $desc,
                n.filename = $filename, n.page_number = $page
            ON MATCH SET n.description = CASE WHEN $desc <> ''
                THEN $desc ELSE n.description END
        """, {"media_id": media_id, "url": url, "desc": desc,
               "filename": chunk.get("filename"), "page": chunk.get("page_number")})
        stats["images" if label == "ImageNode" else "tables"] += 1

    # 连接 VLM 提取的实体到文本实体
    for chunk in media_chunks:
        entities = _extract_entities_from_description(
            vlm_descriptions.get(chunk["chunk_id"], {}).get("description", "")
        )
        for entity_name in entities:
            write_cypher("""
                MATCH (media {media_id: $media_id})
                MATCH (e:Entity {name: $entity_name})
                MERGE (media)-[:ILLUSTRATES]->(e)
            """, {"media_id": chunk["chunk_id"], "entity_name": entity_name})
            stats["relations"] += 1

    return stats
```

- [ ] **Step 3: Commit**

```bash
git add backend/storage/graph_schema.py backend/storage/graph_ingestion.py
git commit -m "feat: Neo4j ImageNode/TableNode + VLM description ingestion"
```

---

## Phase 3: 检索层 — 四通道 RRF

### Task 3.1: 视觉召回通道 + RRF 扩展

**Files:**
- Create: `backend/rag/visual_retriever.py`
- Modify: `backend/rag/utils.py` (三通道→四通道 RRF)
- Modify: `scripts/grid_search_rrf.py` (三权重→四权重)

- [ ] **Step 1: 视觉召回通道**

```python
# backend/rag/visual_retriever.py
"""文本→视觉描述语义检索通道。"""
from backend.embedding.service import EmbeddingService
from backend.milvus.client import MilvusManager

_embedding = EmbeddingService()
_milvus = MilvusManager()


def retrieve_visual(query: str, top_k: int = 5) -> list[dict]:
    """检索与查询相关的图片/表格描述。"""
    query_vec = _embedding.get_embeddings([query])[0]
    results = _milvus.dense_retrieve(
        query_vec, top_k=top_k,
        filter_expr='is_media == true',
    )
    return results if results else []
```

- [ ] **Step 2: RRF 三通道→四通道**

在 `backend/rag/utils.py` 的 `rrf_fusion_three_channel` 重命名为 `rrf_fusion_four_channel`:

```python
RRF_WEIGHT_VISUAL = float(os.getenv("RRF_WEIGHT_VISUAL", "0.15"))

def rrf_fusion_four_channel(
    dense_results, sparse_results, graph_results, visual_results=None,
    k=60, weights=None, top_k=10,
) -> list:
    if weights is None:
        weights = (RRF_WEIGHT_DENSE, RRF_WEIGHT_SPARSE,
                    RRF_WEIGHT_GRAPH, RRF_WEIGHT_VISUAL)
    w1, w2, w3, w4 = weights
    scores = {}
    all_docs = {}

    # 前三路保持不变，追加第四路
    if visual_results:
        for rank, (doc, _) in enumerate(visual_results, 1):
            key = doc.get("chunk_id") or doc.get("text", "")[:50]
            scores[key] = scores.get(key, 0) + w4 / (k + rank)
            all_docs[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [all_docs[k] for k, _ in ranked[:top_k]]
```

- [ ] **Step 3: 更新网格搜索脚本**

`scripts/grid_search_rrf.py` — `evaluate_with_weights` 改为接受 4 个权重参数，iterate 改为四元组。

- [ ] **Step 4: Commit**

```bash
git add backend/rag/visual_retriever.py backend/rag/utils.py scripts/grid_search_rrf.py
git commit -m "feat: 4-channel RRF with visual/text-to-image retrieval"
```

---

## Phase 4: 路由层 — Multimodal Specialist

### Task 4.1: 多模态专家 Agent

**Files:**
- Create: `backend/agent/multimodal_specialist.py`
- Modify: `backend/agent/orchestrator.py` (+节点, +Supervisor路由)
- Modify: `backend/schemas.py` (+MultimodalChunk)

- [ ] **Step 1: Multimodal Specialist 节点**

```python
# backend/agent/multimodal_specialist.py
"""多模态专家 Agent：视觉检索 + 图片解读 + 回答生成。"""
from backend.rag.visual_retriever import retrieve_visual
from backend.agent.orchestrator import _get_worker_model

MULTIMODAL_KEYWORDS = ["图表", "曲线", "图像", "图片", "趋势图", "柱状图", "饼图"]


def is_multimodal_query(query: str) -> bool:
    return any(kw in query for kw in MULTIMODAL_KEYWORDS)


def multimodal_specialist_node(state: dict) -> dict:
    """多模态专家节点：检索视觉内容 + 生成带有图表引用的回答。"""
    user_query = state.get("user_query", "")
    model = _get_worker_model()

    # 1. 视觉召回
    visual_results = retrieve_visual(user_query, top_k=5)

    # 2. 组装上下文（文本+图片描述+URL）
    context_parts = []
    for v in visual_results:
        entity = v.get("entity", v)
        url = entity.get("associated_media_urls", "")
        text = entity.get("text", "")[:400]
        context_parts.append(f"[{text}]\n(Image: {url})")

    context = "\n\n".join(context_parts) if context_parts else "未找到相关图表。"
    prompt = f"根据以下图表信息回答问题。如有图片URL，请在回答中引用。\n\n{context}\n\n问题：{user_query}"

    answer = model.invoke(prompt)
    if hasattr(answer, "content"):
        answer = answer.content

    return {
        "messages": [AIMessage(content=str(answer))],
        "agent_trace": {"multimodal_sources": len(visual_results)},
    }
```

- [ ] **Step 2: Supervisor 路由扩展**

在 `orchestrator.py` 中:
- 路由白名单新增 `"multimodal_specialist"`
- Supervisor Prompt 新增路由规则
- 图节点中新增 `multimodal_specialist_node`

- [ ] **Step 3: 验证新 Agent 路由**

```bash
uv run python -c "
from backend.agent.multimodal_specialist import is_multimodal_query
print(is_multimodal_query('这个图表说明了什么趋势？'))  # True
print(is_multimodal_query('什么是GraphRAG？'))            # False
"
```

- [ ] **Step 4: Commit**

```bash
git add backend/agent/multimodal_specialist.py backend/agent/orchestrator.py backend/schemas.py
git commit -m "feat: add Multimodal Specialist Agent with visual retrieval"
```

---

## Phase 5: 集成 + 迁移 + 配置

### Task 5.1: 数据库迁移 + 环境变量

**Files:**
- Create: `scripts/migrate_graph_schema.py`
- Modify: `docker-compose.yml` (+MinIO bucket init), `.env.example`

- [ ] **Step 1: PyMuPDF column 迁移**

```bash
uv run python -c "
from backend.storage.database import engine
from sqlalchemy import text
with engine.connect() as c:
    c.execute(text('ALTER TABLE parent_chunks ADD COLUMN associated_media_urls VARCHAR(2048) DEFAULT \"\"'))
    c.commit()
    print('Migration done')
"
```

- [ ] **Step 2: .env.example 追加**

```env
# ===== v7.0 Multimodal =====
LAYOUT_ANALYSIS_ENABLED=true
VLM_MODEL=qwen-vl-plus
VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MINIO_ENDPOINT=localhost:9000
MINIO_BUCKET=ragent-media
RRF_WEIGHT_VISUAL=0.15
```

- [ ] **Step 3: 全链路导入验证**

```bash
uv run python -c "
from backend.documents.layout_analyzer import analyze_pdf_layout
from backend.documents.vlm_descriptor import describe_image
from backend.rag.visual_retriever import retrieve_visual
from backend.rag.utils import rrf_fusion_four_channel
from backend.agent.multimodal_specialist import is_multimodal_query
print('All v7.0 modules OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/migrate_graph_schema.py docker-compose.yml .env.example pyproject.toml uv.lock
git commit -m "feat: v7.0 integration — migration, env config, dependencies"
```

---

## 验收标准

### Phase 1 — 版面分析
- [ ] PDF 上传后，版面分析正确区分标题/段落/表格/图片
- [ ] 表格不被截断（完整保留在同一个元素中）
- [ ] 图片/表格截图成功上传 MinIO

### Phase 2 — 多模态入图
- [ ] ImageNode/TableNode 正确写入 Neo4j
- [ ] VLM 为图片生成可读的中文描述
- [ ] ILLUSTRATES 边连接图片节点和文本实体

### Phase 3 — 四通道 RRF
- [ ] 视觉通道权重 w4 参与 RRF 融合
- [ ] 网格搜索脚本支持四参数遍历
- [ ] 视觉召回能返回图片描述和 MinIO URL

### Phase 4 — Multimodal Specialist
- [ ] "这个图表"类查询被路由到 multimodal_specialist
- [ ] 回答中包含图片引用 URL
- [ ] HITL 机制对低置信度 VLM 描述触发人工审核

---

## 执行顺序

```
Phase 1 (Layout Analysis) ──► Phase 2 (VLM + Neo4j) ──► Phase 3 (4-Channel RRF)
                                                                      │
                                                                      ▼
                                                          Phase 4 (Multimodal Agent)
                                                                      │
                                                                      ▼
                                                          Phase 5 (Integration)
```

Phase 1→2 强依赖（版面分析输出→VLM 输入）。Phase 3→4 依赖（视觉召回→Agent 使用）。Phase 1 和 Phase 3 可部分并行（检索层改造不依赖版面分析完成）。
