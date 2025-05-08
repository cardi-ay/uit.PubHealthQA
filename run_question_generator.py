"""
Script để chạy module sinh câu hỏi từ các chủ đề trong file sample_topics.txt
"""

import os
import sys
from pathlib import Path

# Đảm bảo có thể import từ src
sys.path.insert(0, '.')

# Sử dụng đường dẫn tuyệt đối thay vì tương đối
from src.generation.question_generator import generate_questions_from_topics

def main():
    # Thiết lập đường dẫn
    current_dir = Path('.')
    topic_file = current_dir / 'data' / 'topics.txt'
    vector_db_path = current_dir / 'data' / 'gold' / 'db_faiss_phapluat_yte_full_final'
    
    # Kiểm tra file topics có tồn tại không
    if not topic_file.exists():
        print(f"Không tìm thấy file topic tại: {topic_file}")
        return
    
    # Kiểm tra vector db có tồn tại không
    if not vector_db_path.exists():
        print(f"Không tìm thấy vector database tại: {vector_db_path}")
        return
    
    print("Bắt đầu sinh câu hỏi cho các chủ đề...")
    
    # Chạy sinh câu hỏi
    generate_questions_from_topics(
        topic_file_path=str(topic_file),
        vector_db_path=str(vector_db_path),
        groq_model_name="llama3-70b-8192",  # Có thể thay đổi model nếu cần
        num_questions_per_level=2,  # Số câu hỏi cho mỗi cấp độ Bloom
        chunks_per_topic=5  # Số chunks sử dụng cho mỗi topic
    )
    
    print("Hoàn thành sinh câu hỏi!")

if __name__ == "__main__":
    main() 