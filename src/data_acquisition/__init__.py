"""
Module thu thập dữ liệu cho hệ thống RAG UIT@PubHealthQA.

Bao gồm:
- Trình thu thập dữ liệu từ web (crawling từ vbpl.vn/boyte)
- Trình nạp dữ liệu từ các nguồn khác nhau (ingest)
- Xử lý và lưu trữ dữ liệu thô
- Chuyển đổi dữ liệu sang định dạng JSON
"""

from .data_loader import load_json_data

"""
Module chứa các công cụ thu thập dữ liệu.

Bao gồm:
- Trình thu thập dữ liệu từ web (crawling)
- Trình nạp dữ liệu từ các nguồn khác nhau (ingest)
"""

__all__ = [
    "crawlContents",
    "crawlLinks",
    "data_loader",
    "load_json_data"
]