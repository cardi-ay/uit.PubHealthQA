"""
Module thu thập dữ liệu cho hệ thống RAG UIT@PubHealthQA.

Các chức năng:
- Crawl dữ liệu từ các nguồn web (vbpl.vn/boyte)
- Xử lý và lưu trữ dữ liệu thô
- Chuyển đổi dữ liệu sang định dạng JSON
"""

from .data_loader import load_json_data