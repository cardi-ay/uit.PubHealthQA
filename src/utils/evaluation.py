"""
Module cung cấp các hàm đánh giá hiệu suất cho hệ thống RAG UIT@PubHealthQA.
"""

import logging
from typing import List, Dict, Any, Set, Tuple, Optional, Union
import numpy as np
from langchain_core.documents import Document

def calc_recall_at_k(
    retrieved_docs: List[Union[Document, Tuple[Document, float]]],
    relevant_doc_ids: List[str],
    k: Optional[int] = None,
    use_metadata_field: str = "document_id"
) -> float:
    """
    Tính toán Recall@k cho các kết quả truy vấn.
    
    Args:
        retrieved_docs: Danh sách các document đã truy xuất hoặc các tuple (document, score)
        relevant_doc_ids: Danh sách các document ID được coi là liên quan
        k: Số lượng kết quả đầu tiên để đánh giá (mặc định là số lượng tất cả tài liệu được truy xuất)
        use_metadata_field: Trường metadata để sử dụng làm ID (mặc định là "document_id")
        
    Returns:
        Giá trị Recall@k từ 0 đến 1
    """
    if not retrieved_docs or not relevant_doc_ids:
        logging.warning("Danh sách documents truy xuất hoặc danh sách document ID liên quan rỗng")
        return 0.0
        
    # Nếu k không được chỉ định, sử dụng tất cả các kết quả
    if k is None:
        k = len(retrieved_docs)
    else:
        k = min(k, len(retrieved_docs))
        
    # Tạo tập hợp các document ID liên quan
    relevant_ids_set = set(relevant_doc_ids)
    
    # Trích xuất document ID từ top-k kết quả
    retrieved_ids = []
    for i, doc_item in enumerate(retrieved_docs):
        if i >= k:
            break
            
        if isinstance(doc_item, tuple):
            doc, _ = doc_item  # Nếu là tuple (doc, score)
        else:
            doc = doc_item  # Nếu chỉ là doc
            
        # Lấy ID từ metadata
        doc_id = doc.metadata.get(use_metadata_field, None)
        if doc_id is not None:
            retrieved_ids.append(doc_id)
    
    # Đếm số document liên quan được tìm thấy trong top-k
    found_relevant = set(retrieved_ids).intersection(relevant_ids_set)
    num_found = len(found_relevant)
    
    # Tính toán Recall@k
    recall_k = num_found / len(relevant_ids_set) if relevant_ids_set else 0.0
    
    return recall_k

def calc_precision_at_k(
    retrieved_docs: List[Union[Document, Tuple[Document, float]]],
    relevant_doc_ids: List[str],
    k: Optional[int] = None,
    use_metadata_field: str = "document_id"
) -> float:
    """
    Tính toán Precision@k cho các kết quả truy vấn.
    
    Args:
        retrieved_docs: Danh sách các document đã truy xuất hoặc các tuple (document, score)
        relevant_doc_ids: Danh sách các document ID được coi là liên quan
        k: Số lượng kết quả đầu tiên để đánh giá (mặc định là số lượng tất cả tài liệu được truy xuất)
        use_metadata_field: Trường metadata để sử dụng làm ID (mặc định là "document_id")
        
    Returns:
        Giá trị Precision@k từ 0 đến 1
    """
    if not retrieved_docs:
        logging.warning("Danh sách documents truy xuất rỗng")
        return 0.0
        
    # Nếu k không được chỉ định, sử dụng tất cả các kết quả
    if k is None:
        k = len(retrieved_docs)
    else:
        k = min(k, len(retrieved_docs))
        
    # Tạo tập hợp các document ID liên quan
    relevant_ids_set = set(relevant_doc_ids)
    
    # Trích xuất document ID từ top-k kết quả
    retrieved_ids = []
    for i, doc_item in enumerate(retrieved_docs):
        if i >= k:
            break
            
        if isinstance(doc_item, tuple):
            doc, _ = doc_item  # Nếu là tuple (doc, score)
        else:
            doc = doc_item  # Nếu chỉ là doc
            
        # Lấy ID từ metadata
        doc_id = doc.metadata.get(use_metadata_field, None)
        if doc_id is not None:
            retrieved_ids.append(doc_id)
    
    # Đếm số document liên quan được tìm thấy trong top-k
    found_relevant = set(retrieved_ids).intersection(relevant_ids_set)
    num_found = len(found_relevant)
    
    # Tính toán Precision@k
    precision_k = num_found / k if k > 0 else 0.0
    
    return precision_k

def calc_ndcg_at_k(
    retrieved_docs: List[Tuple[Document, float]],
    relevant_docs_with_scores: Dict[str, float],
    k: Optional[int] = None,
    use_metadata_field: str = "document_id"
) -> float:
    """
    Tính toán Normalized Discounted Cumulative Gain (NDCG) tại k.
    
    Args:
        retrieved_docs: Danh sách các tuple (document, score) đã truy xuất
        relevant_docs_with_scores: Dict với khóa là document ID và giá trị là điểm liên quan (relevance score)
        k: Số lượng kết quả đầu tiên để đánh giá
        use_metadata_field: Trường metadata để sử dụng làm ID
        
    Returns:
        Giá trị NDCG@k từ 0 đến 1
    """
    if not retrieved_docs or not relevant_docs_with_scores:
        logging.warning("Danh sách documents truy xuất hoặc danh sách điểm liên quan rỗng")
        return 0.0
    
    # Nếu k không được chỉ định, sử dụng tất cả các kết quả
    if k is None:
        k = len(retrieved_docs)
    else:
        k = min(k, len(retrieved_docs))
    
    # Tính DCG
    dcg = 0.0
    for i in range(k):
        if i >= len(retrieved_docs):
            break
            
        doc, _ = retrieved_docs[i]  # retrieved_docs là tuple (doc, score)
        doc_id = doc.metadata.get(use_metadata_field, None)
        
        if doc_id is not None and doc_id in relevant_docs_with_scores:
            # Sử dụng log cơ số 2 theo công thức DCG
            relevance = relevant_docs_with_scores[doc_id]
            dcg += relevance / np.log2(i + 2)  # +2 vì đánh số từ 1 và log_2(1) = 0
    
    # Tính IDCG (Ideal DCG)
    # Sắp xếp giảm dần các điểm liên quan
    sorted_relevance = sorted(relevant_docs_with_scores.values(), reverse=True)
    idcg = 0.0
    for i in range(min(k, len(sorted_relevance))):
        idcg += sorted_relevance[i] / np.log2(i + 2)
    
    # Tính NDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return ndcg

def calc_map(
    retrieved_docs: List[Union[Document, Tuple[Document, float]]],
    relevant_doc_ids: List[str],
    use_metadata_field: str = "document_id"
) -> float:
    """
    Tính toán Mean Average Precision (MAP).
    
    Args:
        retrieved_docs: Danh sách các document đã truy xuất
        relevant_doc_ids: Danh sách các document ID được coi là liên quan
        use_metadata_field: Trường metadata để sử dụng làm ID
        
    Returns:
        Giá trị MAP từ 0 đến 1
    """
    if not retrieved_docs or not relevant_doc_ids:
        logging.warning("Danh sách documents truy xuất hoặc danh sách document ID liên quan rỗng")
        return 0.0
        
    relevant_ids_set = set(relevant_doc_ids)
    precisions = []
    relevant_count = 0
    
    for i, doc_item in enumerate(retrieved_docs):
        if isinstance(doc_item, tuple):
            doc, _ = doc_item  # Nếu là tuple (doc, score)
        else:
            doc = doc_item  # Nếu chỉ là doc
            
        doc_id = doc.metadata.get(use_metadata_field, None)
        
        if doc_id is not None and doc_id in relevant_ids_set:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precisions.append(precision_at_i)
    
    # Tính Average Precision
    if not precisions:
        return 0.0
        
    ap = sum(precisions) / len(relevant_ids_set)
    return ap

def evaluate_retrieval(
    queries_with_results: List[Dict[str, Any]],
    metrics: List[str] = ["recall@k", "precision@k"],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """
    Đánh giá hiệu suất của hệ thống retrieval.
    
    Args:
        queries_with_results: Danh sách các query với kết quả và ground truth
                             Mỗi dict phải có keys: 'query', 'retrieved_docs', 'relevant_doc_ids'
        metrics: Các metric cần tính ["recall@k", "precision@k", "ndcg@k", "map"]
        k_values: Các giá trị k để tính metric
        
    Returns:
        Dict chứa kết quả đánh giá
    """
    results = {}
    
    # Khởi tạo dict kết quả
    for metric in metrics:
        if metric in ["recall@k", "precision@k", "ndcg@k"]:
            results[metric] = {k: [] for k in k_values}
        elif metric == "map":
            results[metric] = []
    
    # Đánh giá từng query
    for query_result in queries_with_results:
        query = query_result.get("query", "")
        retrieved_docs = query_result.get("retrieved_docs", [])
        relevant_doc_ids = query_result.get("relevant_doc_ids", [])
        relevant_docs_with_scores = query_result.get("relevant_docs_with_scores", {})
        
        if not retrieved_docs or not relevant_doc_ids:
            continue
            
        # Tính toán các metric cho query hiện tại
        for metric in metrics:
            if metric == "recall@k":
                for k in k_values:
                    recall = calc_recall_at_k(retrieved_docs, relevant_doc_ids, k)
                    results[metric][k].append(recall)
                    
            elif metric == "precision@k":
                for k in k_values:
                    precision = calc_precision_at_k(retrieved_docs, relevant_doc_ids, k)
                    results[metric][k].append(precision)
                    
            elif metric == "ndcg@k":
                if relevant_docs_with_scores:
                    for k in k_values:
                        ndcg = calc_ndcg_at_k(
                            [(d, s) if isinstance(d, Document) and isinstance(s, float) else (d, 0.0) 
                             for d, s in retrieved_docs] if all(isinstance(x, tuple) for x in retrieved_docs) 
                            else [(d, 0.0) for d in retrieved_docs], 
                            relevant_docs_with_scores, 
                            k
                        )
                        results[metric][k].append(ndcg)
                        
            elif metric == "map":
                map_score = calc_map(retrieved_docs, relevant_doc_ids)
                results[metric].append(map_score)
    
    # Tính giá trị trung bình
    summary = {"num_queries": len(queries_with_results)}
    
    for metric in metrics:
        if metric in ["recall@k", "precision@k", "ndcg@k"]:
            summary[metric] = {k: np.mean(results[metric][k]) if results[metric][k] else 0.0 for k in k_values}
        elif metric == "map":
            summary[metric] = np.mean(results[metric]) if results[metric] else 0.0
    
    return summary