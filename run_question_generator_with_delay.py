"""
Script để chạy module sinh câu hỏi từ các chủ đề với delay giữa các yêu cầu để tránh rate limit
"""

import os
import sys
import time
from pathlib import Path

# Đảm bảo có thể import từ src
sys.path.insert(0, '.')

from src.utils.logging_utils import setup_logger
from src.generation.question_generator import generate_questions_from_topics, GroqQuestionGenerator
from src.vector_store.faiss_manager import initialize_embedding_model, load_vector_db

# Set up logger
logger = setup_logger("question_generator_runner", log_file="outputs/logs/question_generator_runner.log")

def generate_questions_for_topics_with_delay(
    topic_file_path,
    vector_db_path,
    output_dir="outputs",
    groq_model_name="llama3-70b-8192",
    num_questions_per_level=2,
    chunks_per_topic=5,
    delay_between_topics=60  # Thêm 60 giây delay giữa các chủ đề
):
    """
    Sinh câu hỏi cho các chủ đề với delay giữa các chủ đề để tránh rate limit
    """
    # Đọc danh sách chủ đề
    topic_path = Path(topic_file_path)
    if not topic_path.exists():
        logger.error(f"Không tìm thấy file chủ đề: {topic_path}")
        return []
    
    with open(topic_path, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"Đã đọc {len(topics)} chủ đề từ file {topic_path}")
    
    # Khởi tạo model embedding
    embeddings = initialize_embedding_model("bkai-foundation-models/vietnamese-bi-encoder")
    if not embeddings:
        logger.error("Không thể khởi tạo model embedding")
        return []
    
    # Tải vector database
    vector_db = load_vector_db(vector_db_path, embeddings)
    if not vector_db:
        logger.error(f"Không thể tải vector database từ {vector_db_path}")
        return []
    
    logger.info(f"Đã tải vector database thành công")
    
    # Khởi tạo generator
    generator = GroqQuestionGenerator(model_name=groq_model_name)
    
    # Đường dẫn lưu kết quả
    output_path = Path(output_dir) / "question_generation"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Xử lý từng chủ đề với delay
    all_questions = []
    
    for i, topic in enumerate(topics, 1):
        try:
            logger.info(f"Xử lý chủ đề {i}/{len(topics)}: '{topic}'")
            print(f"Xử lý chủ đề {i}/{len(topics)}: '{topic}'")
            
            # Tạo câu hỏi cho một chủ đề
            questions_by_level = generator.generate_questions_for_topic(
                topic=topic,
                vector_db=vector_db,
                num_questions_per_level=num_questions_per_level,
                chunks_per_topic=chunks_per_topic,
                max_tokens=800  # Giảm tokens để tránh rate limit
            )
            
            # Lưu kết quả từng chủ đề vào một file riêng để không mất dữ liệu nếu bị lỗi
            topic_questions = []
            for level, qa_pairs in questions_by_level.items():
                for qa_pair in qa_pairs:
                    qa_item = {
                        "topic": topic,
                        "level": level,
                        "question": qa_pair.get("question", ""),
                        "answer": qa_pair.get("answer", ""),
                        "citations": qa_pair.get("citations", ""),
                        "source_chunks": qa_pair.get("source_chunks", [])
                    }
                    topic_questions.append(qa_item)
                    all_questions.append(qa_item)
            
            # Lưu kết quả riêng cho từng chủ đề
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_file = output_path / f"questions_{i}_{topic.replace(' ', '_')[:30]}_{timestamp}.json"
            
            import json
            with open(topic_file, "w", encoding="utf-8") as f:
                json.dump(topic_questions, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Đã lưu kết quả cho chủ đề '{topic}' vào file {topic_file}")
            print(f"Đã lưu kết quả cho chủ đề '{topic}' vào file {topic_file}")
            
            # Delay để tránh rate limit - chỉ delay nếu còn chủ đề tiếp theo
            if i < len(topics):
                delay_time = delay_between_topics
                logger.info(f"Đang chờ {delay_time} giây trước khi xử lý chủ đề tiếp theo...")
                print(f"Đang chờ {delay_time} giây trước khi xử lý chủ đề tiếp theo...")
                time.sleep(delay_time)
        
        except Exception as e:
            logger.error(f"Lỗi khi xử lý chủ đề '{topic}': {e}")
            print(f"Lỗi khi xử lý chủ đề '{topic}': {e}")
            # Vẫn delay trước khi thử chủ đề tiếp theo
            if i < len(topics):
                time.sleep(delay_between_topics)
    
    # Lưu tất cả kết quả vào một file cuối cùng
    if all_questions:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = output_path / f"all_questions_{timestamp}.json"
        
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Đã lưu tất cả {len(all_questions)} câu hỏi vào file {final_file}")
        print(f"Đã lưu tất cả {len(all_questions)} câu hỏi vào file {final_file}")
    
    return all_questions

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
    
    print("Bắt đầu sinh câu hỏi cho các chủ đề với delay để tránh rate limit...")
    
    # Chạy sinh câu hỏi với delay
    generate_questions_for_topics_with_delay(
        topic_file_path=str(topic_file),
        vector_db_path=str(vector_db_path),
        groq_model_name="llama3-70b-8192",  # Giữ nguyên model theo yêu cầu
        num_questions_per_level=2,  # Giảm số câu hỏi để tránh rate limit
        chunks_per_topic=5,  # Giảm số chunks để giảm kích thước context
        delay_between_topics=60  # Đợi 60 giây giữa các chủ đề
    )
    
    print("Hoàn thành sinh câu hỏi!")

if __name__ == "__main__":
    main() 