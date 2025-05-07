"""
Module chứa các thành phần cốt lõi của hệ thống RAG UIT@PubHealthQA.

Các module con:
- data_acquisition: Thu thập dữ liệu từ nguồn (bao gồm crawling và ingest)
- preprocessing: Tiền xử lý và phân đoạn (chunking) dữ liệu
- vector_store: Xử lý embedding và truy xuất thông tin từ vector database
- generation: Tạo câu hỏi và câu trả lời dựa trên RAG
- utils: Các tiện ích chung
"""

__version__ = "1.0.0"