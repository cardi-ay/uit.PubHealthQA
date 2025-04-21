"""
Module truy xuất thông tin cho hệ thống RAG UIT@PubHealthQA.

Các chức năng:
- Truy vấn thông tin từ vector database
- Xử lý kết quả truy vấn
- Lọc và xếp hạng các kết quả theo độ phù hợp
"""

from .faiss_retriever import query_documents