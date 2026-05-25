"""文档版面分析：提取文本块 + 识别图片/表格区域。"""
import os

LAYOUT_ENABLED = os.getenv("LAYOUT_ANALYSIS_ENABLED", "true").lower() != "false"


def analyze_pdf_layout(file_path: str) -> list[dict]:
    """分析 PDF 版面，返回元素列表 [{type, text, page_number, bbox}]。"""
    if not LAYOUT_ENABLED:
        return _fallback_layout(file_path)

    try:
        import fitz
        doc = fitz.open(file_path)
        elements = []
        for page_num in range(len(doc)):
            page = doc[page_num]

            # 先尝试提取文本块
            blocks = page.get_text("blocks")
            for block in blocks:
                x0, y0, x1, y1, text, block_type, _ = block
                if text.strip():
                    elements.append({
                        "type": "image" if block_type == 1 else "paragraph",
                        "text": text.strip(),
                        "page_number": page_num,
                        "bbox": [x0, y0, x1, y1],
                    })

            # 提取页面中的图片
            images = page.get_images(full=True)
            for img_info in images:
                elements.append({
                    "type": "image",
                    "text": "",
                    "page_number": page_num,
                    "bbox": None,
                    "xref": img_info[0],
                })

        doc.close()
        return elements
    except ImportError:
        return _fallback_layout(file_path)
    except Exception as e:
        print(f"[LAYOUT] Analysis failed, falling back: {e}")
        return _fallback_layout(file_path)


def _fallback_layout(file_path: str) -> list[dict]:
    """降级方案：整页作为段落处理。"""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return [
        {"type": "paragraph", "text": d.page_content,
         "page_number": d.metadata.get("page", 0), "bbox": None}
        for d in docs
    ]


def is_visual_element(elem_type: str) -> bool:
    return elem_type in ("table", "picture", "figure", "image")
