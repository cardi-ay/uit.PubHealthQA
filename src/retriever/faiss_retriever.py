"""
Module cung cấp các hàm truy vấn và truy xuất từ FAISS vector database.
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from langchain_core.documents import Document

def query_documents(
    vector_db,
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    use_mmr: bool = False,
    with_score: bool = False,
    preprocess_query: bool = True,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Union[Document, Tuple[Document, float]]]:
    """
    Truy vấn văn bản từ vector database.
    
    Args:
        vector_db: Vector database FAISS
        query: Câu truy vấn
        k: Số lượng kết quả trả về
        fetch_k: Số lượng kết quả để lấy trước khi lọc (chỉ dùng với MMR)
        use_mmr: Sử dụng MMR để lấy kết quả đa dạng
        with_score: Trả về điểm số cùng với kết quả
        preprocess_query: Tiền xử lý câu truy vấn không
        filter_metadata: Bộ lọc metadata
        
    Returns:
        Danh sách các kết quả truy vấn
    """
    if not vector_db:
        logging.error("Vector database không tồn tại hoặc không được khởi tạo.")
        return []
    
    try:
        start_time = time.time()
        
        # Tiền xử lý query nếu cần
        if preprocess_query:
            # Import function xử lý từ module preprocess
            try:
                from ..preprocess.document_processor import preprocess_text_for_embedding
                query = preprocess_text_for_embedding(query)
                logging.info(f"Đã tiền xử lý query: '{query}'")
            except ImportError:
                logging.warning("Không thể import hàm tiền xử lý từ module preprocess, sử dụng query gốc.")
        
        # Thực hiện truy vấn phù hợp
        if use_mmr:
            results = vector_db.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, filter=filter_metadata
            )
        elif with_score:
            results = vector_db.similarity_search_with_score(
                query, k=k, filter=filter_metadata
            )
        else:
            results = vector_db.similarity_search(
                query, k=k, filter=filter_metadata
            )
        
        end_time = time.time()
        search_time = end_time - start_time
        
        result_count = len(results)
        logging.info(f"Truy vấn '{query}' hoàn tất sau {search_time:.3f} giây. Tìm thấy {result_count} kết quả.")
        
        return results
    
    except Exception as e:
        logging.error(f"Lỗi khi truy vấn vector database: {e}", exc_info=True)
        return []