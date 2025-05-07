#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để trích xuất các chủ đề duy nhất từ file CSV và lưu vào file.
"""

import pandas as pd
import os
from pathlib import Path

# Thiết lập đường dẫn
input_file = Path('data/bronze/qa_pthu2_only.csv')
output_file = Path('data/topics.txt')

# Kiểm tra xem file CSV có tồn tại không
if not input_file.exists():
    print(f"File CSV không tồn tại: {input_file}")
    exit(1)

try:
    # Đọc file CSV
    print(f"Đang đọc file CSV: {input_file}")
    df = pd.read_csv(input_file)
    
    # Kiểm tra xem cột 'topic' có tồn tại không
    if 'topic' not in df.columns:
        print(f"Cột 'topic' không tồn tại. Các cột có sẵn: {df.columns.tolist()}")
        exit(1)
    
    # Lấy các giá trị duy nhất
    unique_topics = df['topic'].unique()
    
    # Lưu vào file topics.txt
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(unique_topics))
    
    print(f"Đã trích xuất {len(unique_topics)} chủ đề duy nhất và lưu vào {output_file}")
    
except Exception as e:
    print(f"Lỗi: {e}")
    exit(1) 