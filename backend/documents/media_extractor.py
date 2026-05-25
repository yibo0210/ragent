"""从 PDF 中截取图片/表格区域并上传 MinIO。"""
import os
import uuid

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
    try:
        import fitz
    except ImportError:
        return _extract_fallback(media_elements, filename)

    try:
        client = _get_minio_client()
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
    except Exception:
        return _extract_fallback(media_elements, filename)

    doc = fitz.open(pdf_path)
    chunks = []

    for elem in media_elements:
        page_num = elem.get("page_number", 0)
        bbox = elem.get("bbox")
        xref = elem.get("xref")
        media_id = str(uuid.uuid4())[:12]

        try:
            page = doc[page_num]
            if xref:
                # 提取嵌入图片
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]
            elif bbox and len(bbox) == 4:
                rect = fitz.Rect(*bbox)
                pix = page.get_pixmap(clip=rect, dpi=150)
                img_bytes = pix.tobytes("png")
                ext = "png"
            else:
                continue

            object_name = f"{filename}/{media_id}.{ext}"
            from io import BytesIO
            client.put_object(MINIO_BUCKET, object_name,
                              BytesIO(img_bytes), len(img_bytes))
            url = f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{object_name}"

            chunks.append({
                "chunk_id": f"{filename}::media::{media_id}",
                "text": elem.get("text", f"[{elem['type']} on page {page_num}]"),
                "filename": filename,
                "file_type": elem["type"],
                "page_number": page_num,
                "chunk_level": 3,
                "chunk_idx": 0,
                "parent_chunk_id": f"{filename}::p{page_num}::l1::0",
                "root_chunk_id": f"{filename}::p{page_num}::l1::0",
                "associated_media_urls": url,
                "is_media": True,
            })
        except Exception as e:
            print(f"[MEDIA] Extract error for {elem['type']} on p{page_num}: {e}")

    doc.close()
    return chunks


def _extract_fallback(media_elements: list[dict], filename: str) -> list[dict]:
    """无 PyMuPDF/MinIO 时的降级方案：标记但不提取。"""
    chunks = []
    for elem in media_elements:
        page_num = elem.get("page_number", 0)
        media_id = str(uuid.uuid4())[:12]
        chunks.append({
            "chunk_id": f"{filename}::media::{media_id}",
            "text": f"[{elem['type']} on page {page_num} — 需安装 pymupdf+minio 以提取]",
            "filename": filename,
            "file_type": elem["type"],
            "page_number": page_num,
            "chunk_level": 3,
            "chunk_idx": 0,
            "parent_chunk_id": f"{filename}::p{page_num}::l1::0",
            "root_chunk_id": f"{filename}::p{page_num}::l1::0",
            "is_media": True,
        })
    return chunks
