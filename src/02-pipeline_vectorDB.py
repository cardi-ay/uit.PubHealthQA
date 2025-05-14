"""
Pipeline phân đoạn và tạo vector database cho hệ thống UIT@PubHealthQA

Script này lấy dữ liệu từ thư mục silver (Policy.json), phân đoạn và tạo vector database lưu vào thư mục gold.
Sử dụng các module text_splitter.py và chunking.py để xử lý văn bản thành các đoạn nhỏ tối ưu.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union, Optional

# Thêm thư mục gốc vào đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import các module cần thiết
from preprocessing.text_splitter import split_documents
from preprocessing.chunking import (
    chunk_by_article,
    chunk_by_clause,
    chunk_by_paragraph,
    chunk_with_overlap
)
from vector_store.faiss_manager import initialize_embedding_model, create_vector_db
from utils.logging_utils import setup_logger

# Thiết lập logging
logger = setup_logger("vectordb")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pipeline phân đoạn và tạo vector database pháp luật y tế.')
    parser.add_argument('--input_file', default='../data/silver/Policy.json', 
                        help='Đường dẫn đến file dữ liệu đã xử lý (mặc định: ../data/silver/Policy.json)')
    parser.add_argument('--output_dir', default='../data/gold', 
                        help='Thư mục lưu vector database (mặc định: ../data/gold)')
    parser.add_argument('--embedding_model', default='bkai-foundation-models/vietnamese-bi-encoder', 
                        help='Mô hình embedding sử dụng (mặc định: bkai-foundation-models/vietnamese-bi-encoder)')
    parser.add_argument('--chunk_size', type=int, default=500, 
                        help='Kích thước tối đa của mỗi đoạn (mặc định: 500 ký tự)')
    parser.add_argument('--chunk_overlap', type=int, default=100, 
                        help='Độ chồng lấp giữa các đoạn (mặc định: 100 ký tự)')
    parser.add_argument('--chunking_strategy', default='smart', 
                        choices=['article', 'paragraph', 'clause', 'sliding', 'smart'],
                        help='Chiến lược phân đoạn văn bản (mặc định: smart)')
    parser.add_argument('--log_level', default='info', 
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Mức độ chi tiết của log (mặc định: info)')
    return parser.parse_args()

def ensure_directories(output_dir: str):
    """
    Đảm bảo thư mục lưu trữ tồn tại.
    
    Args:
        output_dir: Đường dẫn thư mục đầu ra
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        logger.info(f"Đã tạo thư mục đầu ra: {output_path}")

def load_documents(input_file: str) -> List[Dict[str, Any]]:
    """
    Tải các văn bản đã xử lý từ file JSON.
    
    Args:
        input_file: Đường dẫn đến file JSON
        
    Returns:
        Danh sách các văn bản
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"Đã tải {len(documents)} văn bản từ {input_file}")
        return documents
    except Exception as e:
        logger.error(f"Lỗi khi tải văn bản từ {input_file}: {str(e)}")
        return []

def apply_chunking_strategy(
    documents: List[Dict[str, Any]], 
    strategy: str, 
    chunk_size: int, 
    chunk_overlap: int
) -> List[Dict[str, Any]]:
    """
    Áp dụng chiến lược phân đoạn tương ứng với các văn bản.
    
    Args:
        documents: Danh sách các văn bản
        strategy: Chiến lược phân đoạn ('article', 'paragraph', 'clause', 'sliding', 'smart')
        chunk_size: Kích thước tối đa của đoạn
        chunk_overlap: Độ chồng lấp giữa các đoạn
        
    Returns:
        Danh sách các đoạn văn bản (chunks)
    """
    logger.info(f"Áp dụng chiến lược phân đoạn: {strategy}")
    
    chunks = []
    
    for doc in documents:
        doc_chunks = []
        doc_id = doc.get('id', '')
        doc_title = doc.get('title', '')
        content = doc.get('clean_content') or doc.get('content', '')
        
        if not content:
            logger.warning(f"Văn bản '{doc_title}' (ID: {doc_id}) không có nội dung để phân đoạn")
            continue
            
        # Áp dụng chiến lược phân đoạn tương ứng
        if strategy == 'article':
            doc_chunks = chunk_by_article(content, doc)
        elif strategy == 'paragraph':
            doc_chunks = chunk_by_paragraph(content, doc, chunk_size, chunk_overlap)
        elif strategy == 'clause':
            doc_chunks = chunk_by_clause(content, doc)
        elif strategy == 'sliding':
            doc_chunks = chunk_with_overlap(content, doc, chunk_size, chunk_overlap)
        elif strategy == 'smart':
            # Phân tích cấu trúc văn bản và chọn chiến lược phù hợp
            structure_info = doc.get('structure_info', {})
            has_articles = structure_info.get('has_articles', False)
            
            if has_articles:
                doc_chunks = chunk_by_article(content, doc)
            else:
                doc_chunks = chunk_by_paragraph(content, doc, chunk_size, chunk_overlap)
                
                # Nếu sau khi phân đoạn theo đoạn mà số đoạn quá ít, sử dụng phân đoạn theo cách thông thường
                if len(doc_chunks) <= 2:
                    doc_chunks = chunk_with_overlap(content, doc, chunk_size, chunk_overlap)
        
        # Đảm bảo trường metadata được thiết lập đúng
        for i, chunk in enumerate(doc_chunks):
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            
            # Bổ sung thông tin metadata nếu chưa có
            chunk['metadata'].update({
                'doc_id': doc_id,
                'title': doc_title,
                'chunk_id': f"{doc_id}_chunk_{i+1}",
                'document_type': doc.get('document_type', ''),
                'field': doc.get('field', ''),
                'chunk_index': i,
                'total_chunks': len(doc_chunks)
            })
        
        chunks.extend(doc_chunks)
        
    logger.info(f"Đã tạo tổng cộng {len(chunks)} đoạn văn bản từ {len(documents)} văn bản")
    return chunks

def create_vector_database(
    chunks: List[Dict[str, Any]], 
    output_dir: str, 
    embedding_model_name: str
) -> bool:
    """
    Tạo vector database từ các đoạn văn bản.
    
    Args:
        chunks: Danh sách các đoạn văn bản
        output_dir: Thư mục lưu vector database
        embedding_model_name: Tên mô hình embedding
        
    Returns:
        True nếu tạo thành công, False nếu thất bại
    """
    if not chunks:
        logger.error("Không có đoạn văn bản nào để tạo vector database")
        return False
        
    try:
        # Tải mô hình embedding
        embedding_model = initialize_embedding_model(embedding_model_name)
        if not embedding_model:
            logger.error(f"Không thể khởi tạo mô hình embedding: {embedding_model_name}")
            return False
            
        # Tạo tên thư mục cho vector database
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = f"faiss_index_{timestamp}"
        vector_db_path = os.path.join(output_dir, db_name)
        
        # Tạo vector database
        logger.info(f"Bắt đầu tạo vector database với {len(chunks)} đoạn văn bản")
        db_path = create_vector_db(chunks, vector_db_path, embedding_model)
        
        if db_path:
            logger.info(f"Đã tạo thành công vector database tại: {db_path}")
            
            # Ghi thông tin về index
            index_info = {
                "name": db_name,
                "path": db_path,
                "created_at": timestamp,
                "embedding_model": embedding_model_name,
                "num_chunks": len(chunks),
                "metadata": {
                    "index_type": "faiss",
                    "description": "Vector database cho văn bản pháp luật y tế"
                }
            }
            
            info_path = os.path.join(output_dir, f"{db_name}_info.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(index_info, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Đã lưu thông tin về vector database tại: {info_path}")
            return True
        else:
            logger.error("Không thể tạo vector database")
            return False
            
    except Exception as e:
        logger.error(f"Lỗi khi tạo vector database: {str(e)}")
        return False

def export_chunks_to_json(chunks: List[Dict[str, Any]], output_dir: str):
    """
    Xuất các đoạn văn bản ra file JSON.
    
    Args:
        chunks: Danh sách các đoạn văn bản
        output_dir: Thư mục đầu ra
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunks_file = os.path.join(output_dir, f"document_chunks_{timestamp}.json")
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Đã xuất {len(chunks)} đoạn văn bản ra file: {chunks_file}")
    except Exception as e:
        logger.error(f"Lỗi khi xuất đoạn văn bản ra file JSON: {str(e)}")

def run_pipeline(args):
    """
    Chạy pipeline phân đoạn và tạo vector database.
    
    Args:
        args: Tham số dòng lệnh
    """
    logger.info("=== BẮT ĐẦU PIPELINE PHÂN ĐOẠN VÀ TẠO VECTOR DATABASE ===")
    logger.info(f"File input: {args.input_file}")
    logger.info(f"Thư mục output: {args.output_dir}")
    logger.info(f"Mô hình embedding: {args.embedding_model}")
    logger.info(f"Chiến lược phân đoạn: {args.chunking_strategy}")
    logger.info(f"Kích thước đoạn: {args.chunk_size}")
    logger.info(f"Độ chồng lấp: {args.chunk_overlap}")
    
    start_time = time.time()
    
    # Đảm bảo các thư mục tồn tại
    ensure_directories(args.output_dir)
    
    # Tải văn bản đã xử lý
    documents = load_documents(args.input_file)
    if not documents:
        logger.error("=== PIPELINE THẤT BẠI: Không thể tải văn bản ===")
        return
        
    # Phân đoạn văn bản
    chunks = apply_chunking_strategy(
        documents, 
        args.chunking_strategy, 
        args.chunk_size, 
        args.chunk_overlap
    )
    
    if not chunks:
        logger.error("=== PIPELINE THẤT BẠI: Không thể phân đoạn văn bản ===")
        return
        
    # Xuất các đoạn văn bản ra file JSON
    export_chunks_to_json(chunks, args.output_dir)
    
    # Tạo vector database
    success = create_vector_database(chunks, args.output_dir, args.embedding_model)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    if success:
        logger.info("=== HOÀN THÀNH PIPELINE PHÂN ĐOẠN VÀ TẠO VECTOR DATABASE ===")
        logger.info(f"Thời gian xử lý: {processing_time:.2f} giây")
    else:
        logger.error("=== PIPELINE PHÂN ĐOẠN VÀ TẠO VECTOR DATABASE THẤT BẠI ===")

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args) 