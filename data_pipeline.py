"""
Pipeline xử lý dữ liệu cho hệ thống UIT@PubHealthQA

Script này tự động hóa các bước xử lý dữ liệu từ thu thập (bronze) 
đến làm sạch (silver) và cuối cùng là tạo vector database (gold).
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Đảm bảo có thể import từ thư mục gốc
sys.path.insert(0, '.')

# Import các module cần thiết
from src.data_acquisition.crawlLinks import crawl_links
from src.data_acquisition.crawlContents import crawl_contents
from src.preprocessing.document_processor import process_documents
from src.preprocessing.text_splitter import split_documents
from src.vector_store.faiss_manager import initialize_embedding_model, create_vector_db
from src.utils.logging_utils import setup_logging

# Thiết lập logging
logger = setup_logging()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pipeline xử lý dữ liệu pháp luật y tế.')
    parser.add_argument('--mode', choices=['full', 'crawl', 'process', 'vectorize'], 
                        default='full', help='Chế độ chạy: full (toàn bộ), crawl (chỉ thu thập), process (chỉ xử lý), vectorize (chỉ tạo vector DB)')
    parser.add_argument('--source_url', default='https://vbpl.vn/boyte', 
                        help='URL nguồn để thu thập dữ liệu')
    parser.add_argument('--max_pages', type=int, default=10, 
                        help='Số trang tối đa để thu thập')
    parser.add_argument('--embedding_model', default='bkai-foundation-models/vietnamese-bi-encoder', 
                        help='Mô hình embedding sử dụng cho vectorize')
    parser.add_argument('--chunk_size', type=int, default=500, 
                        help='Kích thước chunk khi phân đoạn văn bản')
    parser.add_argument('--chunk_overlap', type=int, default=100, 
                        help='Độ chồng lấp giữa các chunk')
    
    return parser.parse_args()

def ensure_directories():
    """Đảm bảo các thư mục dữ liệu tồn tại."""
    base_dir = Path('.')
    for data_dir in ['data/bronze', 'data/silver', 'data/gold']:
        dir_path = base_dir / data_dir
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            logger.info(f"Đã tạo thư mục: {dir_path}")

def crawl_data(source_url, max_pages):
    """Thu thập dữ liệu từ nguồn."""
    logger.info(f"Bắt đầu thu thập liên kết từ {source_url}")
    
    # Thư mục đích cho dữ liệu thô
    links_output = Path('./data/bronze/links.json')
    
    # Thu thập links
    links = crawl_links(source_url, max_pages)
    
    logger.info(f"Đã thu thập {len(links)} liên kết. Lưu vào {links_output}")
    
    # Thu thập nội dung
    contents_output = Path('./data/bronze/contents')
    if not contents_output.exists():
        contents_output.mkdir(parents=True)
    
    logger.info(f"Bắt đầu thu thập nội dung văn bản từ các liên kết")
    crawl_contents(links, str(contents_output))
    
    logger.info(f"Hoàn thành thu thập dữ liệu. Dữ liệu thô được lưu tại ./data/bronze/")
    
    return str(contents_output)

def process_data(raw_data_path):
    """Xử lý và làm sạch dữ liệu thô."""
    logger.info(f"Bắt đầu xử lý dữ liệu từ {raw_data_path}")
    
    # Thư mục đích cho dữ liệu đã xử lý
    processed_output = Path('./data/silver/processed_documents')
    if not processed_output.exists():
        processed_output.mkdir(parents=True)
    
    # Xử lý văn bản
    process_documents(raw_data_path, str(processed_output))
    
    logger.info(f"Hoàn thành xử lý dữ liệu. Dữ liệu đã xử lý được lưu tại {processed_output}")
    
    return str(processed_output)

def split_data(processed_data_path, chunk_size, chunk_overlap):
    """Phân đoạn văn bản đã xử lý."""
    logger.info(f"Bắt đầu phân đoạn văn bản với chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    # Thư mục đích cho dữ liệu đã phân đoạn
    chunked_output = Path('./data/silver/chunked_documents')
    if not chunked_output.exists():
        chunked_output.mkdir(parents=True)
    
    # Phân đoạn văn bản
    split_documents(processed_data_path, str(chunked_output), chunk_size, chunk_overlap)
    
    logger.info(f"Hoàn thành phân đoạn văn bản. Dữ liệu đã phân đoạn được lưu tại {chunked_output}")
    
    return str(chunked_output)

def create_vectors(chunked_data_path, embedding_model_name):
    """Tạo vector database từ dữ liệu đã phân đoạn."""
    logger.info(f"Bắt đầu tạo vector database sử dụng mô hình {embedding_model_name}")
    
    # Thư mục đích cho vector database
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vector_db_path = Path(f'./data/gold/db_faiss_phapluat_yte_{timestamp}')
    
    # Khởi tạo mô hình embedding
    embedding_model = initialize_embedding_model(embedding_model_name)
    
    # Tạo vector database
    create_vector_db(chunked_data_path, str(vector_db_path), embedding_model)
    
    # Tạo symbolic link đến vector database mới nhất
    latest_link = Path('./data/gold/db_faiss_phapluat_yte_full_final')
    if latest_link.exists() and latest_link.is_symlink():
        latest_link.unlink()
    os.symlink(vector_db_path, latest_link, target_is_directory=True)
    
    logger.info(f"Hoàn thành tạo vector database. Dữ liệu vector được lưu tại {vector_db_path}")
    logger.info(f"Đã tạo symbolic link tại {latest_link}")
    
    return str(vector_db_path)

def run_pipeline(args):
    """Chạy toàn bộ pipeline xử lý dữ liệu."""
    ensure_directories()
    
    # Theo dõi thời gian chạy
    start_time = datetime.now()
    logger.info(f"Bắt đầu pipeline xử lý dữ liệu lúc {start_time}")
    
    # Chạy từng bước tùy theo mode
    raw_data_path = None
    processed_data_path = None
    chunked_data_path = None
    
    if args.mode in ['full', 'crawl']:
        raw_data_path = crawl_data(args.source_url, args.max_pages)
    
    if args.mode in ['full', 'process']:
        if args.mode == 'process' and raw_data_path is None:
            raw_data_path = './data/bronze/contents'
        processed_data_path = process_data(raw_data_path)
        chunked_data_path = split_data(processed_data_path, args.chunk_size, args.chunk_overlap)
    
    if args.mode in ['full', 'vectorize']:
        if args.mode == 'vectorize' and chunked_data_path is None:
            chunked_data_path = './data/silver/chunked_documents'
        vector_db_path = create_vectors(chunked_data_path, args.embedding_model)
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Hoàn thành pipeline xử lý dữ liệu lúc {end_time}")
    logger.info(f"Tổng thời gian chạy: {duration}")

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args) 