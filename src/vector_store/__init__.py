"""
Module xử lý vector và truy xuất thông tin cho hệ thống RAG UIT@PubHealthQA.

Bao gồm:
- Quản lý vector database (FAISS)
- Truy xuất thông tin (retriever)
- Tìm kiếm ngữ nghĩa
"""

from .faiss_manager import (
    initialize_embedding_model,
    initialize_vector_db,
    create_faiss_vectordb,
    load_vector_db
)

from .faiss_retriever import (
    query_documents, 
    preprocess_text_for_embedding
)

__all__ = [
    "initialize_embedding_model",
    "initialize_vector_db",
    "create_faiss_vectordb",
    "load_vector_db",
    "query_documents",
    "preprocess_text_for_embedding"
]