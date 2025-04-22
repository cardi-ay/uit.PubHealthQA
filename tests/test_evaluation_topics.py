import os
import sys
from pathlib import Path
import json
import re

# Add project root to sys.path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.embed.faiss_manager import initialize_embedding_model, load_vector_db
from src.retriever.faiss_retriever import (
    query_documents, 
    ensemble_retrieval, 
    create_retrieval_function, 
    ensemble_retrieval_with_rerank
)
from src.utils.evaluation import evaluate_retrieval, calc_recall_at_k, calc_precision_at_k
from sentence_transformers import CrossEncoder
import logging
import pandas as pd
from tabulate import tabulate

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def get_topics_from_file(file_path):
    """
    Đọc các chủ đề từ file
    
    Args:
        file_path: Đường dẫn đến file chứa các chủ đề
        
    Returns:
        Danh sách các chủ đề
    """
    topics = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                topic = line.strip()
                if topic:
                    topics.append(topic)
        print(f"Đã đọc {len(topics)} chủ đề từ file {file_path}")
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        
    return topics

def get_relevant_doc_ids_for_topic(topic, vector_db, k=30):
    """
    Lấy document IDs cho một chủ đề để sử dụng làm ground truth
    
    Args:
        topic: Chủ đề cần tìm relevant doc IDs
        vector_db: Vector database đã tải
        k: Số lượng kết quả lấy làm "ground truth"
        
    Returns:
        Danh sách document IDs liên quan
    """
    # Phương pháp này sử dụng kết quả truy vấn ban đầu làm ground truth
    # để so sánh giữa các phương pháp khác nhau
    results = query_documents(vector_db, topic, k=k, with_score=True)
    relevant_doc_ids = []
    
    for doc, score in results:
        doc_id = doc.metadata.get('document_id', '')
        if doc_id:
            relevant_doc_ids.append(doc_id)
    
    return relevant_doc_ids

def get_normalized_document_id(doc_id):
    """
    Chuẩn hóa document_id để dễ so sánh
    """
    return re.sub(r'[_\s-]', '', doc_id.lower())

def check_document_id_match(doc_id, relevant_ids):
    """
    Kiểm tra xem document_id có khớp với bất kỳ ID nào trong danh sách không
    """
    if doc_id in relevant_ids:
        return True
        
    # Chuẩn hóa để so sánh
    normalized_doc_id = get_normalized_document_id(doc_id)
    normalized_relevant_ids = [get_normalized_document_id(id) for id in relevant_ids]
    
    # Kiểm tra khớp chính xác sau khi chuẩn hóa
    if normalized_doc_id in normalized_relevant_ids:
        return True
    
    # Kiểm tra khớp một phần
    for norm_rel_id in normalized_relevant_ids:
        if norm_rel_id in normalized_doc_id or normalized_doc_id in norm_rel_id:
            return True
            
    return False

def evaluate_results_for_topic(topic, results, relevant_doc_ids, k_values):
    """
    Đánh giá kết quả truy vấn cho một chủ đề
    
    Args:
        topic: Chủ đề đang đánh giá
        results: Kết quả truy vấn
        relevant_doc_ids: Document IDs liên quan làm ground truth
        k_values: Các giá trị k cần đánh giá
        
    Returns:
        Dict chứa kết quả đánh giá
    """
    evaluation = {
        "topic": topic,
        "recall": {},
        "precision": {}
    }
    
    # Đánh giá cho từng giá trị k
    for k in k_values:
        recall = calc_recall_at_k(results, relevant_doc_ids, k)
        precision = calc_precision_at_k(results, relevant_doc_ids, k)
        
        evaluation["recall"][k] = recall
        evaluation["precision"][k] = precision
    
    return evaluation

def run_evaluation():
    """
    Thực hiện đánh giá các phương pháp retrieval trên các chủ đề từ file
    """
    try:
        # Đọc các chủ đề từ file
        topics_file = project_root / "data" / "sample_topics.txt"
        topics = get_topics_from_file(topics_file)
        
        if not topics:
            print("Không có chủ đề nào để đánh giá. Kiểm tra lại file chủ đề.")
            return
            
        # Khởi tạo embedding model
        print("Đang khởi tạo embedding model...")
        embedding_model = initialize_embedding_model("bkai-foundation-models/vietnamese-bi-encoder")
        
        # Tải vector database
        faiss_path = project_root / "data" / "gold" / "db_faiss_phapluat_yte_full_final"
        if not faiss_path.exists():
            print(f"Không tìm thấy vector database tại: {faiss_path}")
            return
            
        print(f"Đang tải vector database từ {faiss_path}...")
        vector_db = load_vector_db(faiss_path, embedding_model)
        if vector_db is None:
            print("Không thể tải vector database.")
            return
            
        print(f"Đã tải vector database với {vector_db.index.ntotal} vectors.")
        
        # Tạo retrieval functions
        standard_retriever_func = create_retrieval_function(
            'faiss', k=20, with_score=True
        )
        
        mmr_retriever_func = create_retrieval_function(
            'faiss', k=20, use_mmr=True, with_score=True
        )
        
        # Tải Cross-encoder model cho reranking
        print("Đang tải Cross-encoder model cho reranking...")
        try:
            reranker = CrossEncoder('keepitreal/vietnamese-sbert')
        except Exception as e:
            print(f"Lỗi khi tải Cross-encoder: {e}")
            print("Sử dụng model mặc định thay thế.")
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Cấu hình retrievers
        retriever_configs = [
            (vector_db, standard_retriever_func, 0.6),
            (vector_db, mmr_retriever_func, 0.4)
        ]
        
        # Các giá trị k cho đánh giá
        k_values = [1, 3, 5, 10, 20]
        
        # Chuẩn bị kết quả đánh giá
        standard_evaluations = []
        mmr_evaluations = []
        ensemble_evaluations = []
        ensemble_rerank_evaluations = []
        
        # Đánh giá từng chủ đề
        for topic in topics:
            print(f"\n=== Đánh giá cho chủ đề: '{topic}' ===")
            
            # Lấy document IDs liên quan cho chủ đề này làm ground truth
            # Chúng ta dùng top-30 kết quả từ truy vấn tiêu chuẩn làm ground truth
            relevant_doc_ids = get_relevant_doc_ids_for_topic(topic, vector_db, k=30)
            print(f"Đã tìm thấy {len(relevant_doc_ids)} document IDs liên quan cho chủ đề này")
            
            # 1. Standard retrieval
            print("\n> 1. Standard retrieval:")
            std_docs = query_documents(vector_db, topic, k=20, with_score=True)
            std_eval = evaluate_results_for_topic(topic, std_docs, relevant_doc_ids, k_values)
            standard_evaluations.append(std_eval)
            
            # 2. MMR retrieval
            print("\n> 2. MMR retrieval:")
            mmr_docs = query_documents(vector_db, topic, k=20, use_mmr=True, with_score=True)
            mmr_eval = evaluate_results_for_topic(topic, mmr_docs, relevant_doc_ids, k_values)
            mmr_evaluations.append(mmr_eval)
            
            # 3. Ensemble retrieval
            print("\n> 3. Ensemble retrieval:")
            ensemble_docs = ensemble_retrieval(
                query=topic,
                retrievers=[(vector_db, standard_retriever_func), (vector_db, mmr_retriever_func)],
                k=20,
                retriever_weights=[0.6, 0.4]
            )
            ensemble_eval = evaluate_results_for_topic(topic, ensemble_docs, relevant_doc_ids, k_values)
            ensemble_evaluations.append(ensemble_eval)
            
            # 4. Ensemble + Reranking
            print("\n> 4. Ensemble + Reranking:")
            ensemble_rerank_docs = ensemble_retrieval_with_rerank(
                query=topic,
                retrievers_config=retriever_configs,
                reranker_model=reranker,
                k=20,
                fetch_k=30
            )
            ensemble_rerank_eval = evaluate_results_for_topic(topic, ensemble_rerank_docs, relevant_doc_ids, k_values)
            ensemble_rerank_evaluations.append(ensemble_rerank_eval)
        
        # Tổng hợp kết quả đánh giá
        print("\n--- Tổng hợp kết quả đánh giá ---")
        
        # 1. Recall@k
        recall_data = []
        for k in k_values:
            recall_data.append([
                k,
                sum(eval["recall"][k] for eval in standard_evaluations) / len(standard_evaluations),
                sum(eval["recall"][k] for eval in mmr_evaluations) / len(mmr_evaluations),
                sum(eval["recall"][k] for eval in ensemble_evaluations) / len(ensemble_evaluations),
                sum(eval["recall"][k] for eval in ensemble_rerank_evaluations) / len(ensemble_rerank_evaluations)
            ])
        
        recall_df = pd.DataFrame(
            recall_data, 
            columns=['k', 'Standard', 'MMR', 'Ensemble', 'Ensemble+Rerank']
        )
        
        print("\n=== Recall@k ===")
        print(tabulate(recall_df, headers='keys', tablefmt='pretty', floatfmt='.4f'))
        
        # 2. Precision@k
        precision_data = []
        for k in k_values:
            precision_data.append([
                k,
                sum(eval["precision"][k] for eval in standard_evaluations) / len(standard_evaluations),
                sum(eval["precision"][k] for eval in mmr_evaluations) / len(mmr_evaluations),
                sum(eval["precision"][k] for eval in ensemble_evaluations) / len(ensemble_evaluations),
                sum(eval["precision"][k] for eval in ensemble_rerank_evaluations) / len(ensemble_rerank_evaluations)
            ])
        
        precision_df = pd.DataFrame(
            precision_data, 
            columns=['k', 'Standard', 'MMR', 'Ensemble', 'Ensemble+Rerank']
        )
        
        print("\n=== Precision@k ===")
        print(tabulate(precision_df, headers='keys', tablefmt='pretty', floatfmt='.4f'))
        
        # Lưu kết quả chi tiết theo từng chủ đề
        detailed_results = {
            "topics": topics,
            "standard": standard_evaluations,
            "mmr": mmr_evaluations,
            "ensemble": ensemble_evaluations,
            "ensemble_rerank": ensemble_rerank_evaluations
        }
        
        # Lưu kết quả vào file
        results_dir = project_root / "outputs" / "evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Lưu kết quả tổng hợp
        recall_df.to_csv(results_dir / "topic_recall_comparison.csv", index=False)
        precision_df.to_csv(results_dir / "topic_precision_comparison.csv", index=False)
        
        # Lưu kết quả chi tiết
        with open(results_dir / "topic_detailed_results.json", 'w', encoding='utf-8') as f:
            # Chuyển đổi các đối tượng Python thành JSON
            json_results = {
                "topics": topics,
                "standard": [
                    {
                        "topic": eval["topic"],
                        "recall": {str(k): v for k, v in eval["recall"].items()},
                        "precision": {str(k): v for k, v in eval["precision"].items()}
                    } for eval in standard_evaluations
                ],
                "mmr": [
                    {
                        "topic": eval["topic"],
                        "recall": {str(k): v for k, v in eval["recall"].items()},
                        "precision": {str(k): v for k, v in eval["precision"].items()}
                    } for eval in mmr_evaluations
                ],
                "ensemble": [
                    {
                        "topic": eval["topic"],
                        "recall": {str(k): v for k, v in eval["recall"].items()},
                        "precision": {str(k): v for k, v in eval["precision"].items()}
                    } for eval in ensemble_evaluations
                ],
                "ensemble_rerank": [
                    {
                        "topic": eval["topic"],
                        "recall": {str(k): v for k, v in eval["recall"].items()},
                        "precision": {str(k): v for k, v in eval["precision"].items()}
                    } for eval in ensemble_rerank_evaluations
                ]
            }
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        # Lưu kết quả theo từng chủ đề
        topic_results = {}
        
        for i, topic in enumerate(topics):
            topic_results[topic] = {
                "standard": {
                    "recall": standard_evaluations[i]["recall"],
                    "precision": standard_evaluations[i]["precision"]
                },
                "mmr": {
                    "recall": mmr_evaluations[i]["recall"],
                    "precision": mmr_evaluations[i]["precision"]
                },
                "ensemble": {
                    "recall": ensemble_evaluations[i]["recall"],
                    "precision": ensemble_evaluations[i]["precision"]
                },
                "ensemble_rerank": {
                    "recall": ensemble_rerank_evaluations[i]["recall"],
                    "precision": ensemble_rerank_evaluations[i]["precision"]
                }
            }
            
            # Tạo DataFrame cho topic này
            recall_topic_data = []
            precision_topic_data = []
            
            for k in k_values:
                recall_topic_data.append([
                    k,
                    standard_evaluations[i]["recall"][k],
                    mmr_evaluations[i]["recall"][k],
                    ensemble_evaluations[i]["recall"][k],
                    ensemble_rerank_evaluations[i]["recall"][k]
                ])
                
                precision_topic_data.append([
                    k,
                    standard_evaluations[i]["precision"][k],
                    mmr_evaluations[i]["precision"][k],
                    ensemble_evaluations[i]["precision"][k],
                    ensemble_rerank_evaluations[i]["precision"][k]
                ])
            
            # Tạo DataFrames
            recall_topic_df = pd.DataFrame(
                recall_topic_data, 
                columns=['k', 'Standard', 'MMR', 'Ensemble', 'Ensemble+Rerank']
            )
            
            precision_topic_df = pd.DataFrame(
                precision_topic_data, 
                columns=['k', 'Standard', 'MMR', 'Ensemble', 'Ensemble+Rerank']
            )
            
            # Tạo thư mục cho mỗi chủ đề
            topic_dir = results_dir / "topics" / re.sub(r'[\\/*?:"<>|]', "_", topic)
            topic_dir.mkdir(parents=True, exist_ok=True)
            
            # Lưu kết quả cho chủ đề này
            recall_topic_df.to_csv(topic_dir / "recall.csv", index=False)
            precision_topic_df.to_csv(topic_dir / "precision.csv", index=False)
        
        print(f"\nĐã lưu kết quả đánh giá vào thư mục {results_dir}")
        
    except Exception as e:
        print(f"Lỗi trong quá trình đánh giá: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_evaluation()