"""
Module quản lý việc tải và đọc dữ liệu cho hệ thống RAG UIT@PubHealthQA.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """
    Tải dữ liệu từ file JSON.
    
    Args:
        file_path: Đường dẫn đến file JSON
        
    Returns:
        Danh sách các văn bản đã tải
    """
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