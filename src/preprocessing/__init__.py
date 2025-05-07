"""
Module xử lý dữ liệu cho hệ thống RAG UIT@PubHealthQA.

Bao gồm:
- Xử lý văn bản và tài liệu
- Phân đoạn (chunking) nội dung
- Các chức năng tiền xử lý khác nhau
"""

from .document_processor import preprocess_text_for_embedding
from .text_splitter import initialize_text_splitter

__all__ = [
    "document_processor",
    "text_splitter",
    "chunking",
    "preprocess_text_for_embedding",
    "initialize_text_splitter"
]