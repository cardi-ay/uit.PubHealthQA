"""
Module xử lý phân đoạn (chunking) văn bản cho hệ thống RAG UIT@PubHealthQA.
"""

import logging
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

def initialize_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None
) -> RecursiveCharacterTextSplitter:
    """
    Khởi tạo text splitter để phân đoạn văn bản.
    
    Args:
        chunk_size: Số ký tự tối đa mỗi đoạn (chunk)
        chunk_overlap: Số ký tự chồng lấn giữa các đoạn
        separators: Danh sách các dấu phân tách, theo thứ tự ưu tiên
        
    Returns:
        Đối tượng RecursiveCharacterTextSplitter đã cấu hình
    """
    if separators is None:
        separators = [
            # Các yếu tố cấu trúc
            "\nChương ", "\nMục ", "\nĐiều ",
            # Đoạn văn, câu, từ
            "\n\n", "\n", ". ", "? ", "! ", " ", ""
        ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=separators
    )
    
    logging.info(f"Đã khởi tạo Text Splitter với chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    return text_splitter