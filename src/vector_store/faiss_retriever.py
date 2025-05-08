"""
Module cung cấp các hàm truy vấn và truy xuất từ FAISS vector database.
Hỗ trợ phương pháp Ensemble retrieval và reranking.
"""

import logging
import time
import heapq
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from langchain_core.documents import Document
import numpy as np

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
            # Import function xử lý từ module preprocessing
            try:
                from ..preprocessing.document_processor import preprocess_text_for_embedding
                query = preprocess_text_for_embedding(query)
                logging.info(f"Đã tiền xử lý query: '{query}'")
            except ImportError:
                logging.warning("Không thể import hàm tiền xử lý từ module preprocessing, sử dụng query gốc.")
        
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

def calculate_ensemble_scores(doc_scores_list: List[Dict[str, float]], 
                             weights: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Tính toán điểm số tổng hợp từ nhiều nguồn retriever.
    
    Args:
        doc_scores_list: Danh sách các dict, mỗi dict chứa điểm số từ một retriever
        weights: Trọng số cho mỗi retriever, mặc định là bằng nhau
        
    Returns:
        Dict chứa điểm số tổng hợp cho mỗi document ID
    """
    if not doc_scores_list:
        return {}
    
    # Số lượng retriever
    n_retrievers = len(doc_scores_list)
    
    # Nếu không có trọng số, mặc định mỗi retriever có trọng số bằng nhau
    if weights is None:
        weights = [1.0 / n_retrievers] * n_retrievers
    elif len(weights) != n_retrievers:
        logging.warning(f"Số lượng trọng số ({len(weights)}) không khớp với số lượng retriever ({n_retrievers}). Sử dụng trọng số bằng nhau.")
        weights = [1.0 / n_retrievers] * n_retrievers
    
    # Chuẩn hóa trọng số
    sum_weights = sum(weights)
    weights = [w / sum_weights for w in weights]
    
    ensemble_scores = defaultdict(float)
    
    # Kết hợp điểm số từ các retriever
    for i, doc_scores in enumerate(doc_scores_list):
        for doc_id, score in doc_scores.items():
            ensemble_scores[doc_id] += score * weights[i]
    
    return dict(ensemble_scores)

def normalize_scores(doc_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Chuẩn hóa điểm số về dải [0, 1].
    
    Args:
        doc_scores: Dict chứa điểm số cho mỗi document ID
        
    Returns:
        Dict chứa điểm số đã chuẩn hóa
    """
    if not doc_scores:
        return {}
    
    # Tìm min và max
    min_score = min(doc_scores.values())
    max_score = max(doc_scores.values())
    
    # Tránh chia cho 0
    if max_score == min_score:
        return {doc_id: 1.0 for doc_id in doc_scores}
    
    # Chuẩn hóa về dải [0, 1]
    normalized_scores = {
        doc_id: (score - min_score) / (max_score - min_score)
        for doc_id, score in doc_scores.items()
    }
    
    return normalized_scores

def ensemble_retrieval(
    query: str,
    retrievers: List[Tuple[Any, Callable]],
    k: int = 5,
    retriever_weights: Optional[List[float]] = None,
    preprocess_query: bool = True,
    normalize: bool = True
) -> List[Tuple[Document, float]]:
    """
    Thực hiện truy vấn ensemble từ nhiều retriever.
    
    Args:
        query: Câu truy vấn
        retrievers: Danh sách các tuple (retriever, retrieval_function)
        k: Số lượng kết quả trả về
        retriever_weights: Trọng số cho mỗi retriever
        preprocess_query: Có tiền xử lý query không
        normalize: Có chuẩn hóa điểm số trước khi kết hợp không
        
    Returns:
        Danh sách các document với điểm số tổng hợp, được sắp xếp theo thứ tự giảm dần
    """
    start_time = time.time()
    
    # Tiền xử lý query nếu cần
    processed_query = query
    if preprocess_query:
        try:
            from ..preprocessing.document_processor import preprocess_text_for_embedding
            processed_query = preprocess_text_for_embedding(query)
        except ImportError:
            logging.warning("Không thể import hàm tiền xử lý. Sử dụng query gốc.")
    
    # Thu thập kết quả từ mỗi retriever
    all_docs = {}  # Lưu trữ tất cả các document theo ID
    doc_scores_list = []  # Lưu trữ điểm số từ mỗi retriever
    
    for i, (retriever, retrieval_func) in enumerate(retrievers):
        try:
            # Gọi hàm retrieval tương ứng
            retriever_start_time = time.time()
            results = retrieval_func(retriever, processed_query)
            retriever_time = time.time() - retriever_start_time
            
            # Xử lý kết quả
            doc_scores = {}
            for result in results:
                if isinstance(result, tuple):  # (Document, score)
                    doc, score = result
                    doc_id = f"{doc.metadata.get('document_id', 'unknown')}_{hash(doc.page_content)}"
                    all_docs[doc_id] = doc
                    doc_scores[doc_id] = score
                else:  # Document only
                    doc = result
                    doc_id = f"{doc.metadata.get('document_id', 'unknown')}_{hash(doc.page_content)}"
                    all_docs[doc_id] = doc
                    doc_scores[doc_id] = 1.0  # Default score
            
            # Chuẩn hóa điểm số nếu cần
            if normalize:
                doc_scores = normalize_scores(doc_scores)
                
            doc_scores_list.append(doc_scores)
            logging.info(f"Retriever {i+1} hoàn tất sau {retriever_time:.3f} giây với {len(results)} kết quả.")
            
        except Exception as e:
            logging.error(f"Lỗi với retriever {i+1}: {e}", exc_info=True)
            doc_scores_list.append({})
    
    # Tính toán điểm số tổng hợp
    ensemble_scores = calculate_ensemble_scores(doc_scores_list, retriever_weights)
    
    # Sắp xếp kết quả theo điểm số
    sorted_results = []
    for doc_id, score in sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True):
        if doc_id in all_docs:
            sorted_results.append((all_docs[doc_id], score))
    
    # Giới hạn số lượng kết quả
    top_results = sorted_results[:k]
    
    end_time = time.time()
    logging.info(f"Ensemble retrieval hoàn tất sau {end_time - start_time:.3f} giây. Tìm thấy {len(top_results)} kết quả.")
    
    return top_results

def rerank_documents(
    query: str,
    documents: List[Document],
    reranker_model=None,
    top_k: int = 5
) -> List[Tuple[Document, float]]:
    """
    Reranking documents sử dụng cross-encoder model.
    
    Args:
        query: Câu truy vấn
        documents: Danh sách documents cần rerank
        reranker_model: Model reranker (cross-encoder)
        top_k: Số lượng kết quả trả về
        
    Returns:
        Danh sách documents sau khi rerank với điểm số mới
    """
    # Nếu không có documents hoặc không có reranker
    if not documents:
        return []
    
    if reranker_model is None:
        try:
            # Import thư viện sentence-transformers nếu có
            from sentence_transformers import CrossEncoder
            reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logging.info("Đã tải model reranker mặc định: cross-encoder/ms-marco-MiniLM-L-6-v2")
        except ImportError:
            logging.error("Không thể import CrossEncoder. Hãy cài đặt sentence-transformers hoặc cung cấp model reranker")
            # Trả về documents với điểm số giả
            return [(doc, 1.0) for doc in documents[:top_k]]
    
    start_time = time.time()
    
    try:
        # Chuẩn bị dữ liệu đầu vào cho reranker
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Thực hiện reranking
        scores = reranker_model.predict(pairs)
        
        # Kết hợp documents với scores mới
        doc_scores = list(zip(documents, scores))
        
        # Sắp xếp theo điểm số
        reranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        end_time = time.time()
        logging.info(f"Reranking hoàn tất sau {end_time - start_time:.3f} giây.")
        
        # Trả về top_k kết quả
        return reranked_docs[:top_k]
    
    except Exception as e:
        logging.error(f"Lỗi trong quá trình reranking: {e}", exc_info=True)
        # Trả về documents với điểm số giả nếu có lỗi
        return [(doc, 1.0) for doc in documents[:top_k]]

def ensemble_retrieval_with_rerank(
    query: str,
    retrievers_config: List[Tuple[Any, Callable, float]],
    reranker_model=None,
    k: int = 5,
    fetch_k: int = 20,
    preprocess_query: bool = True
) -> List[Tuple[Document, float]]:
    """
    Thực hiện ensemble retrieval sau đó reranking.
    
    Args:
        query: Câu truy vấn
        retrievers_config: Danh sách các tuple (retriever, retrieval_function, weight)
        reranker_model: Model dùng để rerank
        k: Số lượng kết quả cuối cùng
        fetch_k: Số lượng kết quả lấy từ ensemble trước khi rerank
        preprocess_query: Có tiền xử lý query không
        
    Returns:
        Danh sách các document đã được rerank với điểm số
    """
    # Tách retrievers và weights
    retrievers = [(r, func) for r, func, _ in retrievers_config]
    weights = [w for _, _, w in retrievers_config]
    
    # Thực hiện ensemble retrieval
    ensemble_results = ensemble_retrieval(
        query=query,
        retrievers=retrievers,
        k=fetch_k,  # Lấy nhiều hơn để rerank
        retriever_weights=weights,
        preprocess_query=preprocess_query
    )
    
    # Extract documents từ kết quả
    docs = [doc for doc, _ in ensemble_results]
    
    # Rerank documents
    reranked_results = rerank_documents(
        query=query,
        documents=docs,
        reranker_model=reranker_model,
        top_k=k
    )
    
    return reranked_results

def create_retrieval_function(
    retrieval_type: str,
    k: int = 10,
    fetch_k: int = 30,
    with_score: bool = True,
    **kwargs
) -> Callable:
    """
    Tạo một hàm retrieval phù hợp với loại retriever.
    
    Args:
        retrieval_type: Loại retriever ('faiss', 'custom', etc.)
        k: Số lượng kết quả
        fetch_k: Số lượng kết quả lấy trước khi lọc (cho MMR)
        with_score: Trả về điểm số cùng với kết quả
        **kwargs: Các tham số khác
        
    Returns:
        Hàm retrieval
    """
    if retrieval_type == 'faiss':
        # Sử dụng faiss retriever
        def faiss_retrieval(retriever, query):
            return query_documents(
                retriever, query, k=k, fetch_k=fetch_k, 
                use_mmr=kwargs.get('use_mmr', False),
                with_score=with_score,
                preprocess_query=kwargs.get('preprocess_query', True),
                filter_metadata=kwargs.get('filter_metadata', None)
            )
        return faiss_retrieval
    
    elif retrieval_type == 'custom':
        # Sử dụng một retriever tùy chỉnh
        custom_func = kwargs.get('custom_function', None)
        if custom_func and callable(custom_func):
            return custom_func
        else:
            logging.error("Không tìm thấy hàm custom_function hợp lệ.")
            return lambda r, q: []
    
    else:
        logging.error(f"Loại retriever không hỗ trợ: {retrieval_type}")
        return lambda r, q: []

def optimize_retrieval(
    vector_db,
    query: str,
    k: int = 10,
    preprocess_query: bool = True,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Union[Document, Tuple[Document, float]]]:
    """
    Phương pháp truy vấn tối ưu dựa trên kết quả đánh giá.
    Sử dụng MMR với k=10 để đảm bảo cân bằng giữa recall và precision.
    
    Args:
        vector_db: Vector database FAISS
        query: Câu truy vấn
        k: Số lượng kết quả trả về, mặc định là 10 (giá trị tối ưu từ đánh giá)
        preprocess_query: Tiền xử lý câu truy vấn không
        filter_metadata: Bộ lọc metadata
        
    Returns:
        Danh sách các kết quả truy vấn tối ưu
    """
    # Sử dụng MMR để đảm bảo đa dạng kết quả và tránh trùng lặp thông tin
    fetch_k = min(30, 3*k)  # Lấy nhiều hơn để MMR có thể lọc hiệu quả
    
    return query_documents(
        vector_db=vector_db,
        query=query,
        k=k,
        fetch_k=fetch_k,
        use_mmr=True,  # MMR là phương pháp tối ưu ở k=10
        with_score=True,
        preprocess_query=preprocess_query,
        filter_metadata=filter_metadata
    )