"""
Pipeline xử lý dữ liệu thô cho hệ thống UIT@PubHealthQA

Script này xử lý dữ liệu thô từ bronze (raw_Policy.json) và tạo ra dữ liệu đã làm sạch (Policy.json) trong thư mục silver.
Sử dụng module document_processor.py để xử lý nội dung văn bản pháp luật.
"""

import sys
import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Thêm thư mục gốc vào đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import các module cần thiết
from utils.logging_utils import setup_logger
from preprocessing.document_processor import (
    parse_document_id, 
    parse_effective_date, 
    clean_content, 
    preprocess_text_for_embedding,
    find_structural_elements
)

# Thiết lập logging
logger = setup_logger("preprocessing")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pipeline xử lý dữ liệu thô pháp luật y tế.')
    parser.add_argument('--input_file', default='../data/bronze/raw_Policy.json', 
                        help='Đường dẫn đến file dữ liệu thô (mặc định: ../data/bronze/raw_Policy.json)')
    parser.add_argument('--output_file', default='../data/silver/Policy.json', 
                        help='Đường dẫn đến file dữ liệu đã xử lý (mặc định: ../data/silver/Policy.json)')
    parser.add_argument('--log_level', default='info', 
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Mức độ chi tiết của log (mặc định: info)')
    
    return parser.parse_args()

def ensure_directories():
    """Đảm bảo các thư mục dữ liệu tồn tại."""
    base_dir = Path(project_root)
    for data_dir in ['data/bronze', 'data/silver', 'data/gold']:
        dir_path = base_dir / data_dir
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            logger.info(f"Đã tạo thư mục: {dir_path}")

def load_raw_data(input_file: str) -> List[Dict[str, Any]]:
    """
    Tải dữ liệu thô từ file JSON
    
    Args:
        input_file: Đường dẫn đến file JSON
        
    Returns:
        Danh sách các văn bản dạng dict
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        logger.info(f"Đã đọc {len(raw_data)} bản ghi từ {input_file}")
        return raw_data
    except Exception as e:
        logger.error(f"Lỗi khi đọc file {input_file}: {str(e)}")
        return []

def extract_issuing_body(content: str) -> str:
    """
    Trích xuất cơ quan ban hành từ nội dung văn bản
    
    Args:
        content: Nội dung văn bản
        
    Returns:
        Tên cơ quan ban hành
    """
    common_issuers = ["BỘ Y TẾ", "CHÍNH PHỦ", "THỦ TƯỚNG CHÍNH PHỦ", 
                       "BỘ TRƯỞNG BỘ Y TẾ", "ỦY BAN THƯỜNG VỤ QUỐC HỘI", 
                       "QUỐC HỘI"]
    
    for issuer in common_issuers:
        if issuer in content[:500].upper():
            return issuer.title()
    
    return ""

def process_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Xử lý một văn bản
    
    Args:
        doc: Văn bản dạng dict với các trường từ raw_Policy.json
        
    Returns:
        Văn bản đã xử lý
    """
    try:
        # Lấy dữ liệu thô
        raw_title = doc.get('TenVanBan', '').strip()
        raw_content = doc.get('NoiDung', '').strip()
        
        if not raw_content or not raw_title:
            logger.warning(f"Văn bản '{raw_title or 'không có tiêu đề'}' thiếu nội dung hoặc tiêu đề")
            return None
            
        # Xử lý nội dung văn bản
        clean_text, start_idx, end_idx = clean_content(raw_content)
        
        # Trích xuất mã định danh
        doc_id = parse_document_id(raw_title, raw_content)
        
        # Xử lý ngày có hiệu lực
        effective_date = parse_effective_date(doc.get('NgayHieuLuc', ''))
        
        # Tiền xử lý văn bản cho embedding
        processed_text = preprocess_text_for_embedding(clean_text)
        
        # Phân tích cấu trúc văn bản
        structural_elements = find_structural_elements(clean_text)
        
        # Trích xuất cơ quan ban hành
        issuing_body = extract_issuing_body(raw_content)
        
        # Tạo đối tượng văn bản đã xử lý
        processed_doc = {
            'id': doc_id if doc_id else f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'title': raw_title,
            'content': processed_text,
            'raw_content': raw_content,
            'clean_content': clean_text,
            'url': doc.get('DuongLink', ''),
            'law_id': doc_id,
            'issue_date': '',  # Chưa có thông tin này từ raw_Policy
            'effective_date': effective_date if effective_date else doc.get('NgayHieuLuc', ''),
            'issuing_body': issuing_body,
            'document_type': doc.get('LoaiVanBan_ThuocTinh', '').strip(),
            'status': '',  # Không có thông tin này từ raw_Policy
            'field': doc.get('LinhVuc', '').strip(),
            'structure_info': {
                'elements_count': len(structural_elements),
                'has_chapters': any(e['type'] == 'Chương' for e in structural_elements),
                'has_articles': any(e['type'] == 'Điều' for e in structural_elements),
                'sections': [{'type': e['type'], 'identifier': e['identifier'], 'title': e.get('title', '')} 
                            for e in structural_elements[:10]]  # Lưu 10 phần tử đầu tiên
            },
            'processing_metadata': {
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'content_length': len(raw_content),
                'clean_content_length': len(clean_text),
                'processed_content_length': len(processed_text)
            }
        }
        
        return processed_doc
    except Exception as e:
        logger.error(f"Lỗi khi xử lý văn bản '{doc.get('TenVanBan', '')}': {str(e)}")
        return None

def process_raw_policy(input_file: str, output_file: str) -> int:
    """
    Xử lý dữ liệu thô từ raw_Policy.json sang Policy.json
    
    Args:
        input_file: Đường dẫn đến file raw_Policy.json
        output_file: Đường dẫn đến file Policy.json đầu ra
        
    Returns:
        Số lượng văn bản đã xử lý thành công
    """
    logger.info(f"Bắt đầu xử lý dữ liệu từ {input_file}")
    start_time = time.time()
    
    try:
        # Tải dữ liệu thô
        raw_data = load_raw_data(input_file)
        if not raw_data:
            logger.error("Không thể tải dữ liệu từ file nguồn hoặc file rỗng")
            return 0
            
        # Xử lý từng văn bản
        processed_data = []
        for idx, doc in enumerate(raw_data):
            logger.debug(f"Đang xử lý văn bản {idx+1}/{len(raw_data)}: {doc.get('TenVanBan', '')}")
            processed_doc = process_document(doc)
            if processed_doc:
                processed_data.append(processed_doc)
                
        # Lưu dữ liệu đã xử lý
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Đã xử lý và lưu {len(processed_data)} bản ghi vào {output_file}")
        logger.info(f"Thời gian xử lý: {processing_time:.2f} giây")
        
        return len(processed_data)
    
    except Exception as e:
        logger.error(f"Lỗi khi xử lý dữ liệu: {str(e)}")
        return 0

def run_pipeline(args):
    """
    Chạy pipeline xử lý dữ liệu thô.
    
    Args:
        args: Tham số dòng lệnh
    """
    logger.info("=== BẮT ĐẦU PIPELINE XỬ LÝ DỮ LIỆU THÔ ===")
    logger.info(f"File input: {args.input_file}")
    logger.info(f"File output: {args.output_file}")
    
    # Đảm bảo các thư mục tồn tại
    ensure_directories()
    
    # Xử lý dữ liệu
    processed_count = process_raw_policy(args.input_file, args.output_file)
    
    if processed_count > 0:
        logger.info(f"=== HOÀN THÀNH PIPELINE XỬ LÝ DỮ LIỆU THÔ ===")
        logger.info(f"Đã xử lý thành công {processed_count} văn bản")
    else:
        logger.error("=== PIPELINE XỬ LÝ DỮ LIỆU THÔ THẤT BẠI ===")

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args) 