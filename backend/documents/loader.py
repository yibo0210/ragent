"""文档加载和分片服务模块

支持 PDF/Word/Excel 格式，提供三级分层分块（1200 → 600 → 300），
并构建块之间的父子层级关系，用于知识库构建和长文档问答场景。
"""
import os
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_core.documents import Document

class DocumentLoader:
    """文档加载和分片服务

    接收外部传入的 chunk_size（分块尺寸）和 chunk_overlap（重叠尺寸），并基于此计算三级分块的最终尺寸（保证不低于默认值：一级≥1200、二级≥600、三级≥300）。
初始化三个层级的 RecursiveCharacterTextSplitter 拆分器，分别对应三级分块规则
"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # 保留原有参数以兼容外部调用；默认启用三层滑动窗口分块。
        level_1_size = max(1200, chunk_size * 2)
        level_1_overlap = max(240, chunk_overlap * 2)
        level_2_size = max(600, chunk_size)
        level_2_overlap = max(120, chunk_overlap)
        level_3_size = max(300, chunk_size // 2)
        level_3_overlap = max(60, chunk_overlap // 2)

        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size=level_1_size,
            chunk_overlap=level_1_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size=level_2_size,
            chunk_overlap=level_2_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size=level_3_size,
            chunk_overlap=level_3_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        )
#生成唯一的分块 ID，格式为：文件名::p页码::l层级::索引（例如 report.pdf::p3::l2::5），确保每个分块可唯一标识
    @staticmethod
    def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
        return f"{filename}::p{page_number}::l{level}::{index}"
#一级分块：将整页文本拆分为大粒度块，作为「根块」（root_chunk_id 指向自身，parent_chunk_id 为空）。
#二级分块：将每个一级块拆分为中粒度块，parent_chunk_id 指向所属一级块，root_chunk_id 继承一级块 ID。
#三级分块：将每个二级块拆分为小粒度块，parent_chunk_id 指向所属二级块，root_chunk_id 仍继承一级块 ID。
#每个分块记录核心元信息：文本内容、分块 ID、父子 / 根 ID、层级（1/2/3）、全局索引（chunk_idx）。
    def _split_raw_docs(self, raw_docs, filename: str, doc_type: str) -> list[dict]:
        """将原始文档列表分片为三级块。"""
        documents = []
        page_global_chunk_idx = 0
        for doc in raw_docs:
            base_doc = {
                "filename": filename,
                "file_path": "",
                "file_type": doc_type,
                "page_number": doc.metadata.get("page", 0),
            }
            page_chunks = self._split_page_to_three_levels(
                text=(doc.page_content or "").strip(),
                base_doc=base_doc,
                page_global_chunk_idx=page_global_chunk_idx,
            )
            page_global_chunk_idx += len(page_chunks)
            documents.extend(page_chunks)
        return documents

#过滤空文本块，避免无效数据。
    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
    ) -> List[Dict]:
        if not text:
            return []

        root_chunks: List[Dict] = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]

        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
        level_1_counter = 0
        level_2_counter = 0
        level_3_counter = 0

        for level_1_doc in level_1_docs:
            level_1_text = (level_1_doc.page_content or "").strip()
            if not level_1_text:
                continue
            level_1_id = self._build_chunk_id(filename, page_number, 1, level_1_counter)
            level_1_counter += 1

            level_1_chunk = {
                **base_doc,
                "text": level_1_text,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": page_global_chunk_idx,
            }
            page_global_chunk_idx += 1
            root_chunks.append(level_1_chunk)

            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = (level_2_doc.page_content or "").strip()
                if not level_2_text:
                    continue
                level_2_id = self._build_chunk_id(filename, page_number, 2, level_2_counter)
                level_2_counter += 1

                level_2_chunk = {
                    **base_doc,
                    "text": level_2_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": page_global_chunk_idx,
                }
                page_global_chunk_idx += 1
                root_chunks.append(level_2_chunk)

                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = (level_3_doc.page_content or "").strip()
                    if not level_3_text:
                        continue
                    level_3_id = self._build_chunk_id(filename, page_number, 3, level_3_counter)
                    level_3_counter += 1
                    root_chunks.append({
                        **base_doc,
                        "text": level_3_text,
                        "chunk_id": level_3_id,
                        "parent_chunk_id": level_2_id,
                        "root_chunk_id": level_1_id,
                        "chunk_level": 3,
                        "chunk_idx": page_global_chunk_idx,
                    })
                    page_global_chunk_idx += 1

        return root_chunks
#文件类型判断：根据后缀识别 PDF/Word/Excel，调用对应 Loader 加载文档。
#原始文档处理：加载后打印日志（原始页数、每页文本长度），便于调试。
#分块执行：遍历文档每页，基于每页文本生成三级分块，累加全局索引，最终返回所有分块列表。
#异常处理：捕获加载 / 分块异常，打印错误日志并向上抛出。
    def load_document(self, file_path: str, filename: str) -> list[dict]:
        """加载单个文档并分片"""
        file_lower = filename.lower()
        print(f"[INFO] 开始读取文件: {filename}")

        if file_lower.endswith(".pdf"):
            doc_type = "PDF"
            # --- v7.0 版面分析 ---
            from backend.documents.layout_analyzer import analyze_pdf_layout, is_visual_element
            elements = analyze_pdf_layout(file_path)
            text_blocks = [e for e in elements if not is_visual_element(e["type"])]
            media_elements = [e for e in elements if is_visual_element(e["type"])]

            raw_docs = [Document(page_content=e["text"], metadata={
                "page": e.get("page_number", 0),
                "element_type": e["type"],
            }) for e in text_blocks if e.get("text", "").strip()]

            documents = self._split_raw_docs(raw_docs, filename, doc_type)
            for d in documents:
                d["file_path"] = file_path

            # 图片/表格：提取上传
            if media_elements:
                try:
                    from backend.documents.media_extractor import extract_and_upload
                    media_chunks = extract_and_upload(file_path, media_elements, filename)
                    documents.extend(media_chunks)
                except Exception as e:
                    print(f"[WARN] Media extraction failed: {e}")

            print(f"[INFO] 最终生成分块总数: {len(documents)} (含{len(media_elements)}个图表)")
            return documents
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"
            loader = Docx2txtLoader(file_path)
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"
            loader = UnstructuredExcelLoader(file_path)
        elif file_lower.endswith((".md", ".markdown")):
            doc_type = "Markdown"
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            raw_docs = [Document(page_content=text, metadata={"page": 0})]
            return self._split_raw_docs(raw_docs, filename, doc_type)
        elif file_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
            doc_type = "Image"
            name_without_ext = os.path.splitext(filename)[0]
            description = name_without_ext.replace("_", " ").replace("-", " ")
            raw_docs = [Document(page_content=description, metadata={"page": 0})]
            return self._split_raw_docs(raw_docs, filename, doc_type)
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

        try:
            raw_docs = loader.load()
            print(f"[INFO] 原始文档页数: {len(raw_docs)}")
            for i, d in enumerate(raw_docs):
                print(f"第{i + 1}页文本长度: {len(d.page_content)}")
            documents = self._split_raw_docs(raw_docs, filename, doc_type)
            # 补填 file_path 到已存在的 base_doc（_split_raw_docs 不填路径）
            for d in documents:
                d["file_path"] = file_path
            print(f"[INFO] 最终生成分块总数: {len(documents)}")
            return documents
        except Exception as e:
            print(f"[ERROR] 读取文件失败: {str(e)}")
            raise Exception(f"处理文档失败: {str(e)}")
#遍历目标文件夹，筛选出 PDF/Word/Excel 格式的文件。
#逐个调用 load_document 加载分块，忽略单个文件的异常（保证批量处理不中断），最终返回所有文件的分块列表。
    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        """
        从文件夹加载所有文档并分片
        :param folder_path: 文件夹路径
        :return: 所有分片后的文档列表
        """
        all_documents = []

        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            supported = (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".md", ".markdown", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
            if not file_lower.endswith(supported):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                documents = self.load_document(file_path, filename)
                all_documents.extend(documents)
            except Exception:
                continue

        return all_documents
