#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script xử lý tạo chunks và xây dựng vector database FAISS cho hệ thống RAG UIT@PubHealthQA.
Chuyển đổi từ notebook/chunking.ipynb sang Python script.
"""

import json
import time
import logging
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from src.utils.logging_utils import setup_logging
from src.preprocess.document_processor import (
    parse_document_id,
    parse_effective_date,
    clean_content,
    find_structural_elements,
    get_contextual_structure,
    preprocess_text_for_embedding
)
from src.preprocess.text_splitter import initialize_text_splitter
from src.embed.faiss_manager import (
    initialize_embedding_model,
    initialize_vector_db,
    create_faiss_vectordb,
    query_vector_db
)

def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """Tải dữ liệu từ file JSON."""
    if not file_path.exists():
        logging.error(f"Không tìm thấy file: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            logging.info(f"Đã tải thành công {len(data)} văn bản từ {file_path}")
            return data
        else:
            logging.error(f"Lỗi: File {file_path} không chứa một danh sách JSON.")
            return []
    except json.JSONDecodeError as e:
        logging.error(f"Lỗi: File {file_path} không phải là định dạng JSON hợp lệ. {e}")
        return []
    except Exception as e:
        logging.error(f"Lỗi không xác định khi đọc file {file_path}: {e}")
        return []

def process_documents(
    documents_data: List[Dict[str, Any]],
    text_splitter, 
    show_progress: bool = True
) -> List[Document]:
    """
    Xử lý danh sách văn bản và tạo các chunks.
    
    Args:
        documents_data: Danh sách văn bản từ file JSON
        text_splitter: Text splitter đã khởi tạo
        show_progress: Hiển thị thanh tiến trình
        
    Returns:
        Danh sách các chunk đã được xử lý
    """
    processed_doc_count = 0
    error_doc_count = 0
    total_chunks_prepared = 0
    all_processed_chunks = []
    
    if not documents_data:
        logging.error("Dữ liệu văn bản rỗng. Không thể xử lý.")
        return []
    
    # Sử dụng tqdm cho thanh tiến trình nếu được yêu cầu
    doc_iterator = tqdm(documents_data, desc="Đang xử lý văn bản") if show_progress else documents_data
    
    for doc_data in doc_iterator:
        doc_id = "Unknown"
        try:
            # --- Trích xuất Thông tin Cơ bản & Metadata ---
            ten_van_ban = doc_data.get("TenVanBan", "")
            noi_dung = doc_data.get("NoiDung", "")
            if not noi_dung:
                logging.debug(f"Văn bản '{ten_van_ban}' không có nội dung. Bỏ qua.")
                error_doc_count += 1
                continue

            doc_id = parse_document_id(ten_van_ban, noi_dung)
            parsed_date = parse_effective_date(doc_data.get("NgayHieuLuc"))

            base_metadata = {
                "document_id": doc_id,
                "document_type": doc_data.get("LoaiVanBan_ThuocTinh", "UnknownType"),
                "effective_date": parsed_date if parsed_date else doc_data.get("NgayHieuLuc", "N/A"),
                "source_link": doc_data.get("DuongLink", "N/A"),
                "domain": doc_data.get("LinhVuc", "N/A")
            }

            # --- Làm sạch Nội dung ---
            main_content, content_start_offset, content_end_offset = clean_content(noi_dung)
            if not main_content:
                logging.debug(f"Không thể trích xuất nội dung chính cho {doc_id}. Bỏ qua.")
                error_doc_count += 1
                continue

            # --- Tìm các Yếu tố Cấu trúc trong Nội dung chính ---
            structural_elements = find_structural_elements(main_content)

            # --- Phân chia thành các Đoạn (Chunks) ---
            temp_docs = text_splitter.create_documents([main_content], metadatas=[base_metadata])

            doc_chunk_count = 0
            # --- Gán Metadata Cấu trúc và Tiền xử lý Nội dung Chunk ---
            for temp_doc in temp_docs:
                chunk_start_in_main = temp_doc.metadata.get("start_index", 0)
                structure_context = get_contextual_structure(chunk_start_in_main, structural_elements)
                final_metadata = base_metadata.copy()
                final_metadata.update(structure_context)
                final_metadata["start_index_in_main"] = chunk_start_in_main
                if "start_index" in final_metadata:
                    del final_metadata["start_index"]

                # Áp dụng tiền xử lý cho nội dung chunk
                raw_page_content = str(temp_doc.page_content)
                preprocessed_content = preprocess_text_for_embedding(raw_page_content)

                if preprocessed_content:
                    final_chunk_doc = Document(page_content=preprocessed_content, metadata=final_metadata)
                    all_processed_chunks.append(final_chunk_doc)
                    doc_chunk_count += 1
                else:
                    logging.debug(f"Bỏ qua chunk rỗng sau tiền xử lý cho {doc_id}")

            if doc_chunk_count > 0:
                processed_doc_count += 1
                total_chunks_prepared += doc_chunk_count
            else:
                # Chỉ log warning nếu văn bản ban đầu có nội dung đáng kể
                if len(main_content) > 50:
                    logging.warning(f"Không tạo được chunk hợp lệ nào cho {doc_id} (dù đã có main_content).")
                error_doc_count += 1

        except Exception as e:
            error_doc_count += 1
            logging.error(f"Lỗi không xác định khi xử lý văn bản (ID: {doc_id}): {e}", exc_info=True)
    
    logging.info("--- Vòng lặp xử lý hoàn tất ---")
    logging.info(f"Số văn bản đã xử lý thành công (để tạo chunk): {processed_doc_count}")
    logging.info(f"Số văn bản bị bỏ qua/lỗi: {error_doc_count}")
    logging.info(f"Tổng số đoạn (chunks) đã chuẩn bị: {total_chunks_prepared}")
    
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
    logging.info(f"--- Bắt đầu kiểm tra embedding cho {len(chunks)} chunks đã chuẩn bị ---")
    problematic_indices = []
    good_chunks = []
    
    for i, chunk_doc in enumerate(tqdm(chunks, desc="Kiểm tra embedding sau tiền xử lý")):
        try:
            page_content_str = str(chunk_doc.page_content)
            if not page_content_str.strip():
                continue
                
            # Thử embed nội dung chunk đã tiền xử lý
            _ = embedding_model.embed_documents([page_content_str])
            good_chunks.append(chunk_doc)
        except Exception as e:
            logging.error(f"Lỗi khi embed chunk #{i} (VB: {chunk_doc.metadata.get('document_id', 'N/A')}): {e}")
            doc_id_err = chunk_doc.metadata.get('document_id', 'N/A')
            loc_err = chunk_doc.metadata.get('location_detail', 'N/A')
            content_preview_err = page_content_str[:150] if 'page_content_str' in locals() else "N/A"
            logging.error(f"  Văn bản: {doc_id_err}")
            logging.error(f"  Vị trí: {loc_err}")
            logging.error(f"  Nội dung đã tiền xử lý (bắt đầu): '{content_preview_err}...'")
            problematic_indices.append(i)
    
    if problematic_indices:
        logging.warning(f"Tìm thấy {len(problematic_indices)} chunks gây lỗi embedding sau tiền xử lý tại các vị trí (index): {problematic_indices}")
        logging.info(f"Số chunks embed thành công: {len(good_chunks)}")
    else:
        logging.info("--- Kiểm tra embedding tất cả các chunk sau tiền xử lý thành công ---")
    
    return good_chunks

def main(
    json_file_path: Optional[str] = None,
    persist_directory: Optional[str] = None,
    model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    load_existing_faiss_index: bool = True,
    log_file_path: Optional[str] = None
) -> None:
    """
    Hàm chính thực hiện quy trình tạo chunk và xây dựng vector database.
    
    Args:
        json_file_path: Đường dẫn đến file JSON input
        persist_directory: Đường dẫn thư mục lưu FAISS index
        model_name: Tên mô hình embedding
        chunk_size: Kích thước chunk
        chunk_overlap: Độ chồng lấn giữa các chunk
        load_existing_faiss_index: Có tải FAISS index hiện có không
        log_file_path: Đường dẫn file log
    """
    # Đường dẫn mặc định nếu không được chỉ định
    if json_file_path is None:
        json_file_path = "data/silver/data_vbpl_boyte_full_details.json"
    
    if persist_directory is None:
        persist_directory = "data/gold/db_faiss_phapluat_yte_full_final"
        
    if log_file_path is None:
        log_file_path = "outputs/logs/processing_log_vi_python_faiss.txt"
        
    # Chuyển đổi sang Path
    json_file_path = Path(json_file_path)
    persist_directory = Path(persist_directory)
    log_file_path = Path(log_file_path)
    
    # Thiết lập logging
    setup_logging(log_file_path=log_file_path, log_level=logging.INFO)
    
    # Cấu hình các tham số
    logging.info("--- Thiết lập các tham số cấu hình ---")
    logging.info(f"File dữ liệu JSON: {json_file_path}")
    logging.info(f"Thư mục lưu Index FAISS: {persist_directory}")
    logging.info(f"Model Embedding: {model_name}")
    logging.info(f"Chunk size: {chunk_size}")
    logging.info(f"Chunk overlap: {chunk_overlap}")
    logging.info(f"Load index FAISS nếu tồn tại: {load_existing_faiss_index}")
    
    # Khởi tạo thư mục lưu trữ
    persist_directory.mkdir(parents=True, exist_ok=True)
    
    # Khởi tạo các thành phần dùng chung
    logging.info("--- Khởi tạo model embedding ---")
    embedding_model = initialize_embedding_model(model_name)
    if not embedding_model:
        logging.error("Không thể khởi tạo model embedding. Dừng xử lý.")
        return
    
    # Khởi tạo text splitter
    separators = [
        "\nChương ", "\nMục ", "\nĐiều ",
        "\n\n", "\n", ". ", "? ", "! ", " ", ""
    ]
    text_splitter = initialize_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    # Khởi tạo vector store
    logging.info("--- Khởi tạo vector database ---")
    vector_db, index_exists = initialize_vector_db(
        persist_directory,
        embedding_model,
        load_existing=load_existing_faiss_index
    )
    
    # Tải dữ liệu từ file JSON
    logging.info(f"--- Bắt đầu tải dữ liệu từ {json_file_path} ---")
    all_documents_data = load_json_data(json_file_path)
    
    if not all_documents_data:
        logging.warning("Không tải được dữ liệu hoặc file rỗng. Kiểm tra lại file JSON và đường dẫn.")
        return
    
    # Xử lý và tạo chunks
    logging.info(f"--- Bắt đầu xử lý {len(all_documents_data)} văn bản để tạo chunks ---")
    all_processed_chunks = process_documents(all_documents_data, text_splitter)
    
    # Kiểm tra embedding
    if all_processed_chunks:
        validated_chunks = validate_embeddings(all_processed_chunks, embedding_model)
        
        # Tạo hoặc cập nhật index FAISS
        if validated_chunks:
            logging.info(f"--- Bắt đầu tạo hoặc cập nhật index FAISS từ {len(validated_chunks)} chunks hợp lệ ---")
            start_index_time = time.time()
            
            vector_db = create_faiss_vectordb(
                documents=validated_chunks,
                embeddings=embedding_model,
                persist_directory=persist_directory,
                existing_vector_db=vector_db if index_exists and load_existing_faiss_index else None,
                add_to_existing=index_exists and load_existing_faiss_index
            )
            
            end_index_time = time.time()
            logging.info(f"Hoàn tất quá trình tạo/cập nhật/lưu index FAISS sau {end_index_time - start_index_time:.2f} giây.")
            
            # Kiểm tra cuối cùng
            logging.info("--- Kiểm tra cuối cùng ---")
            
            # Thử truy vấn nếu vector_db tồn tại
            if vector_db and vector_db.index.ntotal > 0:
                logging.info("Thực hiện truy vấn thử nghiệm...")
                query = "Đăng ký kinh doanh thuốc"
                
                start_query_time = time.time()
                docs = query_vector_db(vector_db, query, k=5)
                end_query_time = time.time()
                
                logging.info(f"Truy vấn hoàn tất sau {end_query_time - start_query_time:.3f} giây.")
                
                if docs:
                    logging.info(f"Truy vấn thử '{query}' tìm thấy {len(docs)} kết quả (top 5):")
                    for i, doc in enumerate(docs[:5]):
                        print(f"\n--- Kết quả {i+1} ---")
                        print(f"Metadata: {doc.metadata}")
                        content_preview = str(doc.page_content)[:300]
                        print(f"Nội dung (300 ký tự đầu): {content_preview}...")
                else:
                    logging.info(f"Truy vấn thử '{query}' không tìm thấy kết quả nào.")
        else:
            logging.error("Không có chunks hợp lệ sau khi kiểm tra embedding. Không thể xây dựng vector database.")
    else:
        logging.error("Không có chunks được tạo. Không thể xây dựng vector database.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tạo chunks và xây dựng vector database FAISS")
    parser.add_argument("--json", type=str, help="Đường dẫn đến file JSON input")
    parser.add_argument("--output", type=str, help="Đường dẫn thư mục lưu FAISS index")
    parser.add_argument("--model", type=str, default="bkai-foundation-models/vietnamese-bi-encoder", help="Tên mô hình embedding")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Kích thước chunk")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Độ chồng lấn giữa các chunk")
    parser.add_argument("--no-load-existing", action="store_true", help="Không tải FAISS index hiện có")
    parser.add_argument("--log", type=str, help="Đường dẫn file log")
    
    args = parser.parse_args()
    
    main(
        json_file_path=args.json,
        persist_directory=args.output,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        load_existing_faiss_index=not args.no_load_existing,
        log_file_path=args.log
    )