"""
Module quản lý cấu hình logging cho hệ thống RAG UIT@PubHealthQA.
"""

import logging
from pathlib import Path
from typing import Optional, Union

def setup_logging(
    log_file_path: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO,
    console_output: bool = True,
    mode: str = 'w'
) -> None:
    """
    Thiết lập cấu hình logging cho hệ thống.
    
    Args:
        log_file_path: Đường dẫn đến file log
        log_level: Mức độ log (INFO, WARNING, ERROR, etc.)
        console_output: Có hiển thị log trên console không
        mode: Chế độ mở file log ('w' để ghi đè, 'a' để thêm vào)
        
    Returns:
        None
    """
    # Xóa các handler cũ để tránh log trùng lặp
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
    
    # Thiết lập cấu hình cơ bản
    handlers = []
    
    # Thêm handler cho file log nếu được chỉ định
    if log_file_path:
        # Chuyển đổi sang Path nếu là string
        if isinstance(log_file_path, str):
            log_file_path = Path(log_file_path)
            
        # Đảm bảo thư mục chứa file log tồn tại
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tạo handler cho file log
        file_handler = logging.FileHandler(
            log_file_path, mode=mode, encoding='utf-8'
        )
        handlers.append(file_handler)
    
    # Thêm handler cho console nếu được yêu cầu
    if console_output:
        console_handler = logging.StreamHandler()
        handlers.append(console_handler)
    
    # Thiết lập cấu hình logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Log thông báo khởi tạo
    logging.info("Đã thiết lập cấu hình logging.")
    if log_file_path:
        logging.info(f"File log: {log_file_path}")
        
    return None