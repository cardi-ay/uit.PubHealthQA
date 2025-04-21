"""
Module tiền xử lý dữ liệu cho hệ thống RAG UIT@PubHealthQA.

Các chức năng:
- Làm sạch nội dung văn bản
- Phân đoạn (chunking) văn bản
- Trích xuất thông tin cấu trúc (Chương, Mục, Điều...)
- Chuẩn hóa định dạng và metadata
"""

from .document_processor import (
    clean_content,
    parse_document_id,
    parse_effective_date,
    find_structural_elements,
    get_contextual_structure,
    preprocess_text_for_embedding
)

from .text_splitter import initialize_text_splitter