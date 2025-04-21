"""
Module xử lý embedding và vector database cho hệ thống RAG UIT@PubHealthQA.

Các chức năng:
- Tạo embeddings từ văn bản
- Xây dựng và quản lý vector database (FAISS)
- Truy vấn vector database
"""

from .faiss_manager import (
    initialize_embedding_model,
    initialize_vector_db,
    save_vector_db,
    load_vector_db,
    create_or_update_vector_db,
    query_vector_db
)