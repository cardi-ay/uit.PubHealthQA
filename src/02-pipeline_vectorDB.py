"""
Pipeline phân đoạn và tạo vector database cho hệ thống UIT@PubHealthQA

Script này lấy dữ liệu đã được xử lý từ thư mục silver (Policy.json), phân đoạn và tạo vector database lưu vào thư mục gold.
Dữ liệu trong Policy.json đã được tiền xử lý bởi pipeline preprocessing.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Thêm thư mục gốc vào đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import các module cần thiết
from langchain_core.documents import Document
from preprocessing.text_splitter import initialize_text_splitter
from preprocessing.document_processor import (
    find_structural_elements,
    get_contextual_structure
)
from vector_store.faiss_manager import (
    initialize_embedding_model,
    initialize_vector_db,
    create_faiss_vectordb,
    save_vector_db
)
from utils.logging_utils import setup_logger

# Thiết lập logging
logger = setup_logger("vectordb_pipeline")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pipeline phân đoạn và tạo vector database pháp luật y tế.')
    parser.add_argument('--input_file', default='../data/silver/Policy.json', 
                        help='Đường dẫn đến file dữ liệu đã xử lý (mặc định: ../data/silver/Policy.json)')
    parser.add_argument('--output_dir', default='../data/gold', 
                        help='Thư mục lưu vector database (mặc định: ../data/gold)')
    parser.add_argument('--embedding_model', default='bkai-foundation-models/vietnamese-bi-encoder', 
                        help='Mô hình embedding sử dụng (mặc định: bkai-foundation-models/vietnamese-bi-encoder)')
    parser.add_argument('--chunk_size', type=int, default=1000, 
                        help='Kích thước tối đa của mỗi đoạn (mặc định: 1000 ký tự)')
    parser.add_argument('--chunk_overlap', type=int, default=150, 
                        help='Độ chồng lấp giữa các đoạn (mặc định: 150 ký tự)')
    parser.add_argument('--use_clean_content', action='store_true',
                        help='Sử dụng clean_content thay vì content đã được tiền xử lý')
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
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Không tìm thấy file: {input_file}")
            return []
            
        with open(input_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        if isinstance(documents, list):
            logger.info(f"Đã tải {len(documents)} văn bản từ {input_file}")
            return documents
        else:
            logger.error(f"File {input_file} không chứa danh sách JSON hợp lệ")
            return []
            
    except json.JSONDecodeError as e:
        logger.error(f"Lỗi khi parse JSON từ {input_file}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Lỗi khi tải văn bản từ {input_file}: {str(e)}")
        return []

def process_documents_to_chunks(
    documents: List[Dict[str, Any]], 
    text_splitter,
    use_clean_content: bool = False,
    show_progress: bool = True
) -> List[Document]:
    """
    Xử lý danh sách văn bản và tạo các chunks.
    Dữ liệu đã được tiền xử lý bởi pipeline preprocessing.
    
    Args:
        documents: Danh sách văn bản từ file Policy.json
        text_splitter: Text splitter đã khởi tạo
        use_clean_content: Sử dụng clean_content thay vì content
        show_progress: Hiển thị thanh tiến trình
        
    Returns:
        Danh sách các chunk đã được xử lý
    """
    processed_doc_count = 0
    error_doc_count = 0
    total_chunks_prepared = 0
    all_processed_chunks = []
    
    if not documents:
        logger.error("Dữ liệu văn bản rỗng. Không thể xử lý.")
        return []
    
    # Sử dụng tqdm cho thanh tiến trình nếu được yêu cầu
    doc_iterator = tqdm(documents, desc="Đang phân đoạn văn bản") if show_progress else documents
    
    for doc_data in doc_iterator:
        doc_id = "Unknown"
        try:
            # Trích xuất thông tin từ Policy.json (đã được xử lý)
            doc_id = doc_data.get("id", "Unknown")
            title = doc_data.get("title", "")
            
            # Chọn nội dung để phân đoạn
            # - content: đã được tiền xử lý cho embedding
            # - clean_content: nội dung đã làm sạch nhưng chưa tiền xử lý
            if use_clean_content:
                text_to_chunk = doc_data.get("clean_content", "")
                logger.debug(f"Sử dụng clean_content cho văn bản {doc_id}")
            else:
                text_to_chunk = doc_data.get("content", "")
                logger.debug(f"Sử dụng content (đã tiền xử lý) cho văn bản {doc_id}")
            
            if not text_to_chunk:
                logger.debug(f"Văn bản '{title}' (ID: {doc_id}) không có nội dung. Bỏ qua.")
                error_doc_count += 1
                continue

            # Tạo metadata cơ bản từ dữ liệu đã xử lý
            base_metadata = {
                "document_id": doc_id,
                "title": title,
                "document_type": doc_data.get("document_type", ""),
                "effective_date": doc_data.get("effective_date", ""),
                "issuing_body": doc_data.get("issuing_body", ""),
                "field": doc_data.get("field", ""),
                "url": doc_data.get("url", ""),
                "status": doc_data.get("status", ""),
                "law_id": doc_data.get("law_id", doc_id)
            }

            # Lấy thông tin cấu trúc đã được phân tích
            structure_info = doc_data.get("structure_info", {})
            
            # Nếu cần phân tích lại cấu trúc (chỉ khi dùng clean_content)
            if use_clean_content:
                structural_elements = find_structural_elements(text_to_chunk)
            else:
                # Sử dụng thông tin cấu trúc đã có
                structural_elements = []
                if 'sections' in structure_info:
                    for section in structure_info['sections']:
                        structural_elements.append({
                            'type': section.get('type', ''),
                            'identifier': section.get('identifier', ''),
                            'title': section.get('title', ''),
                            'start': 0,  # Không có thông tin vị trí chi tiết
                            'end': 0
                        })

            # Phân chia thành các đoạn (chunks)
            temp_docs = text_splitter.create_documents([text_to_chunk], metadatas=[base_metadata])

            doc_chunk_count = 0
            # Gán metadata cấu trúc cho từng chunk
            for temp_doc in temp_docs:
                chunk_start_in_main = temp_doc.metadata.get("start_index", 0)
                
                # Chỉ thêm context cấu trúc nếu có thông tin chi tiết
                if structural_elements and any(e.get('start', 0) > 0 for e in structural_elements):
                    structure_context = get_contextual_structure(chunk_start_in_main, structural_elements)
                else:
                    structure_context = {
                        "has_chapters": structure_info.get("has_chapters", False),
                        "has_articles": structure_info.get("has_articles", False)
                    }
                
                final_metadata = base_metadata.copy()
                final_metadata.update(structure_context)
                final_metadata["start_index_in_main"] = chunk_start_in_main
                final_metadata["chunk_index"] = doc_chunk_count
                final_metadata["total_chunks"] = len(temp_docs)
                
                # Loại bỏ start_index tạm thời
                if "start_index" in final_metadata:
                    del final_metadata["start_index"]

                # Tạo Document object
                chunk_content = str(temp_doc.page_content)
                if chunk_content.strip():
                    final_chunk_doc = Document(
                        page_content=chunk_content, 
                        metadata=final_metadata
                    )
                    all_processed_chunks.append(final_chunk_doc)
                    doc_chunk_count += 1
                else:
                    logger.debug(f"Bỏ qua chunk rỗng cho {doc_id}")

            if doc_chunk_count > 0:
                processed_doc_count += 1
                total_chunks_prepared += doc_chunk_count
                logger.debug(f"Đã tạo {doc_chunk_count} chunks cho văn bản {doc_id}")
            else:
                # Chỉ log warning nếu văn bản ban đầu có nội dung đáng kể
                if len(text_to_chunk) > 50:
                    logger.warning(f"Không tạo được chunk hợp lệ nào cho {doc_id}")
                error_doc_count += 1

        except Exception as e:
            error_doc_count += 1
            logger.error(f"Lỗi khi xử lý văn bản (ID: {doc_id}): {e}", exc_info=True)
    
    logger.info("--- Hoàn tất phân đoạn văn bản ---")
    logger.info(f"Số văn bản đã xử lý thành công: {processed_doc_count}")
    logger.info(f"Số văn bản bị bỏ qua/lỗi: {error_doc_count}")
    logger.info(f"Tổng số chunks đã tạo: {total_chunks_prepared}")
    
    return all_processed_chunks

def validate_embeddings(chunks: List[Document], embedding_model) -> List[Document]:
    """
    Kiểm tra từng chunk xem có gây lỗi embedding không.
    
    Args:
        chunks: Danh sách chunks cần kiểm tra
        embedding_model: Mô hình embedding
        
    Returns:
        Danh sách các chunk không gây lỗi embedding
    """
    logger.info(f"--- Bắt đầu kiểm tra embedding cho {len(chunks)} chunks ---")
    problematic_indices = []
    good_chunks = []
    
    for i, chunk_doc in enumerate(tqdm(chunks, desc="Kiểm tra embedding")):
        try:
            page_content_str = str(chunk_doc.page_content)
            if not page_content_str.strip():
                continue
                
            # Thử embed nội dung chunk
            _ = embedding_model.embed_documents([page_content_str])
            good_chunks.append(chunk_doc)
        except Exception as e:
            logger.error(f"Lỗi khi embed chunk #{i} (VB: {chunk_doc.metadata.get('document_id', 'N/A')}): {e}")
            problematic_indices.append(i)
    
    if problematic_indices:
        logger.warning(f"Tìm thấy {len(problematic_indices)} chunks gây lỗi embedding")
        logger.info(f"Số chunks embed thành công: {len(good_chunks)}")
    else:
        logger.info("Tất cả chunks đều embed thành công")
    
    return good_chunks

def create_vector_database(
    chunks: List[Document], 
    output_dir: str, 
    embedding_model_name: str
) -> bool:
    """
    Tạo vector database từ các chunks.
    
    Args:
        chunks: Danh sách các chunks (Document objects)
        output_dir: Thư mục lưu vector database
        embedding_model_name: Tên mô hình embedding
        
    Returns:
        True nếu tạo thành công, False nếu thất bại
    """
    if not chunks:
        logger.error("Không có chunks nào để tạo vector database")
        return False
        
    try:
        # Tải mô hình embedding
        embedding_model = initialize_embedding_model(embedding_model_name)
        if not embedding_model:
            logger.error(f"Không thể khởi tạo mô hình embedding: {embedding_model_name}")
            return False
            
        # Tạo tên thư mục cho vector database
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = f"VectorDB"
        vector_db_path = Path(output_dir) / db_name
        
        # Đảm bảo thư mục tồn tại
        vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Kiểm tra embedding cho các chunks
        validated_chunks = validate_embeddings(chunks, embedding_model)
        
        if not validated_chunks:
            logger.error("Không có chunks hợp lệ sau khi kiểm tra embedding")
            return False
        
        # Tạo vector database
        logger.info(f"Bắt đầu tạo vector database với {len(validated_chunks)} chunks")
        start_time = time.time()
        
        vector_db = create_faiss_vectordb(
            documents=validated_chunks,
            embeddings=embedding_model,
            persist_directory=vector_db_path,
            existing_vector_db=None,
            add_to_existing=False
        )
        
        if vector_db:
            end_time = time.time()
            logger.info(f"Đã tạo thành công vector database sau {end_time - start_time:.2f} giây")
            logger.info(f"Vector database được lưu tại: {vector_db_path}")
            
            # Ghi thông tin về index
            index_info = {
                "name": db_name,
                "path": str(vector_db_path),
                "created_at": timestamp,
                "embedding_model": embedding_model_name,
                "num_chunks": len(validated_chunks),
                "num_vectors": vector_db.index.ntotal if hasattr(vector_db, 'index') else len(validated_chunks),
                "metadata": {
                    "index_type": "faiss",
                    "description": "Vector database cho văn bản pháp luật y tế",
                    "source_file": "Policy.json",
                    "preprocessing_pipeline": "02-pipeline_preprocessing.py"
                }
            }
            
            info_path = Path(output_dir) / f"{db_name}_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(index_info, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Đã lưu thông tin về vector database tại: {info_path}")
            
            # Tạo symlink hoặc file chỉ định database mới nhất
            latest_path = Path(output_dir) / "latest_index_info.json"
            with open(latest_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "latest_index": db_name,
                    "path": str(vector_db_path),
                    "created_at": timestamp
                }, f, ensure_ascii=False, indent=2)
                
            return True
        else:
            logger.error("Không thể tạo vector database")
            return False
            
    except Exception as e:
        logger.error(f"Lỗi khi tạo vector database: {str(e)}", exc_info=True)
        return False

def export_chunks_to_json(chunks: List[Document], output_dir: str):
    """
    Xuất các chunks ra file JSON để debug/review.
    
    Args:
        chunks: Danh sách các chunks (Document objects)
        output_dir: Thư mục đầu ra
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunks_file = Path(output_dir) / f"document_chunks_{timestamp}.json"
        
        # Chuyển đổi Document objects sang dict để serialize
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "content": chunk.page_content,
                "metadata": chunk.metadata
            })
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Đã xuất {len(chunks)} chunks ra file: {chunks_file}")
    except Exception as e:
        logger.error(f"Lỗi khi xuất chunks ra file JSON: {str(e)}")

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
    logger.info(f"Kích thước chunk: {args.chunk_size}")
    logger.info(f"Độ chồng lấp: {args.chunk_overlap}")
    logger.info(f"Sử dụng clean_content: {args.use_clean_content}")
    
    start_time = time.time()
    
    # Đảm bảo các thư mục tồn tại
    ensure_directories(args.output_dir)
    
    # Tải văn bản đã xử lý từ pipeline preprocessing
    documents = load_documents(args.input_file)
    if not documents:
        logger.error("=== PIPELINE THẤT BẠI: Không thể tải văn bản ===")
        return
        
    # Khởi tạo text splitter
    separators = [
        "\nChương ", "\nMục ", "\nĐiều ",
        "\n\n", "\n", ". ", "? ", "! ", " ", ""
    ]
    text_splitter = initialize_text_splitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=separators
    )
    
    # Phân đoạn văn bản thành chunks
    chunks = process_documents_to_chunks(
        documents, 
        text_splitter,
        use_clean_content=args.use_clean_content,
        show_progress=True
    )
    
    if not chunks:
        logger.error("=== PIPELINE THẤT BẠI: Không thể phân đoạn văn bản ===")
        return
        
    # Xuất các chunks ra file JSON (optional, để review)
    export_chunks_to_json(chunks, args.output_dir)
    
    # Tạo vector database
    success = create_vector_database(chunks, args.output_dir, args.embedding_model)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    if success:
        logger.info("=== HOÀN THÀNH PIPELINE PHÂN ĐOẠN VÀ TẠO VECTOR DATABASE ===")
        logger.info(f"Thời gian xử lý: {processing_time:.2f} giây")
        logger.info(f"Số văn bản đã xử lý: {len(documents)}")
        logger.info(f"Số chunks đã tạo: {len(chunks)}")
        logger.info("Dữ liệu đã được xử lý qua các bước:")
        logger.info("1. Pipeline ingesting: Thu thập dữ liệu thô -> bronze/raw_Policy.json")
        logger.info("2. Pipeline preprocessing: Làm sạch và tiền xử lý -> silver/Policy.json")
        logger.info("3. Pipeline vectorDB: Phân đoạn và tạo vector database -> gold/faiss_index_*")
    else:
        logger.error("=== PIPELINE PHÂN ĐOẠN VÀ TẠO VECTOR DATABASE THẤT BẠI ===")

if __name__ == "__main__":
    args = parse_args()
    
    # Setup logging level
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level)
    
    run_pipeline(args) 