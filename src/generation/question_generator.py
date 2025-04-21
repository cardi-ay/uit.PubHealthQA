"""
Module sinh câu hỏi và câu trả lời từ chunks trong vector database sử dụng Groq API.
Sinh câu hỏi theo 3 thang đo của Bloom: Remember, Understand, Apply.
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import json
from datetime import datetime
from dotenv import load_dotenv

import groq
from langchain_core.documents import Document

from ..utils.logging_utils import setup_logger
from ..retriever.faiss_retriever import query_documents
from ..embed.faiss_manager import initialize_embedding_model, load_vector_db

# Load environment variables from .env file
load_dotenv()

# Set up logger
logger = setup_logger(
    "question_generator",
    log_file=Path("outputs/logs/question_generation_groq.log")
)

class GroqQuestionGenerator:
    """
    Lớp sinh câu hỏi và câu trả lời từ chunks trong vector database sử dụng Groq API.
    """
    
    def __init__(self, model_name: str = "llama3-70b-8192", api_key: Optional[str] = None):
        """
        Khởi tạo generator.
        
        Args:
            model_name: Tên model Groq API muốn sử dụng
            api_key: Khóa API của Groq (nếu không cung cấp, sẽ lấy từ biến môi trường GROQ_API_KEY)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Không tìm thấy Groq API key. Hãy đặt biến môi trường GROQ_API_KEY "
                "hoặc cung cấp api_key khi khởi tạo."
            )
        
        # Khởi tạo client
        self.client = groq.Client(api_key=self.api_key)
        logger.info(f"Đã khởi tạo Groq Client với model '{model_name}'")
        
        # Định nghĩa các cấp độ của Bloom
        self.bloom_levels = {
            "remember": "Nhớ - Remember: Câu hỏi yêu cầu người học nhớ lại thông tin, từ khóa, quy định, v.v.",
            "understand": "Hiểu - Understand: Câu hỏi yêu cầu người học hiểu và diễn giải thông tin đã học.",
            "apply": "Áp dụng - Apply: Câu hỏi yêu cầu người học áp dụng kiến thức đã học vào tình huống cụ thể."
        }
        
    def generate_questions_for_topic(
        self, 
        topic: str, 
        vector_db, 
        num_questions_per_level: int = 2,
        chunks_per_topic: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Sinh câu hỏi và câu trả lời cho một chủ đề.
        
        Args:
            topic: Chủ đề cần sinh câu hỏi
            vector_db: Vector database (FAISS)
            num_questions_per_level: Số câu hỏi cho mỗi cấp độ Bloom
            chunks_per_topic: Số chunks sử dụng để sinh câu hỏi
            temperature: Độ sáng tạo [0-1]
            max_tokens: Số tokens tối đa cho mỗi response
            
        Returns:
            Dict chứa câu hỏi và câu trả lời theo từng cấp độ Bloom
        """
        logger.info(f"Bắt đầu sinh câu hỏi cho chủ đề: '{topic}'")
        
        # Truy vấn chunks từ vector database
        chunks = query_documents(
            vector_db=vector_db, 
            query=topic, 
            k=chunks_per_topic, 
            use_mmr=True  # Sử dụng MMR để đa dạng kết quả
        )
        
        if not chunks:
            logger.warning(f"Không tìm thấy chunks phù hợp cho chủ đề: '{topic}'")
            return {level: [] for level in self.bloom_levels.keys()}
        
        # Ghép nội dung các chunks
        context = "\n\n".join([chunk.page_content for chunk in chunks])
        logger.info(f"Đã trích xuất {len(chunks)} chunks cho chủ đề '{topic}'")
        
        result_questions = {}
        
        # Sinh câu hỏi và câu trả lời cho từng cấp độ Bloom
        for level, description in self.bloom_levels.items():
            try:
                qa_pairs = self._generate_qa_for_level(
                    topic=topic,
                    context=context,
                    level=level,
                    level_description=description,
                    num_questions=num_questions_per_level,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                result_questions[level] = qa_pairs
                logger.info(f"Đã sinh {len(qa_pairs)} cặp câu hỏi-trả lời cấp độ {level} cho chủ đề '{topic}'")
            except Exception as e:
                logger.error(f"Lỗi khi sinh câu hỏi cấp độ {level} cho chủ đề '{topic}': {e}")
                result_questions[level] = []
        
        return result_questions
    
    def _generate_qa_for_level(
        self,
        topic: str,
        context: str,
        level: str,
        level_description: str,
        num_questions: int,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> List[Dict[str, str]]:
        """
        Sinh câu hỏi và câu trả lời cho một cấp độ Bloom cụ thể.
        
        Args:
            topic: Chủ đề cần sinh câu hỏi
            context: Nội dung chunks từ vector database
            level: Cấp độ Bloom ('remember', 'understand', 'apply')
            level_description: Mô tả chi tiết về cấp độ Bloom
            num_questions: Số câu hỏi cần sinh
            temperature: Độ sáng tạo [0-1]
            max_tokens: Số tokens tối đa cho response
            
        Returns:
            List chứa các câu hỏi và câu trả lời ở cấp độ đã chọn
        """
        # Tạo prompt
        prompt = f"""
Bạn là một chuyên gia giáo dục y tế công cộng. Nhiệm vụ của bạn là tạo ra những câu hỏi chất lượng cao và câu trả lời đầy đủ về chủ đề "{topic}" dựa trên thông tin được cung cấp dưới đây. 

Hãy tạo ra {num_questions} cặp câu hỏi và câu trả lời ở cấp độ: **{level_description}**

Thông tin tham khảo:
```
{context}
```

Yêu cầu:
1. Câu hỏi và câu trả lời phải liên quan đến chủ đề "{topic}"
2. Câu hỏi phải phù hợp với cấp độ "{level}" trong thang đo Bloom
3. Câu hỏi và câu trả lời phải dựa trên các thông tin được cung cấp
4. Câu hỏi và câu trả lời nên có tính thực tiễn và áp dụng được trong lĩnh vực y tế công cộng
5. Câu trả lời phải đầy đủ, chính xác và cung cấp đầy đủ thông tin dựa trên nội dung đã cho
6. Định dạng phản hồi phải theo cấu trúc JSON sau:
```json
[
  {{
    "question": "Câu hỏi đầy đủ ở đây?",
    "answer": "Câu trả lời đầy đủ ở đây."
  }},
  {{
    "question": "Câu hỏi đầy đủ thứ hai ở đây?",
    "answer": "Câu trả lời đầy đủ thứ hai ở đây."
  }}
]
```

Đây là ví dụ về các loại câu hỏi ở mỗi cấp độ Bloom:
- Remember: "Nêu định nghĩa về bảo hiểm y tế theo Luật Bảo hiểm y tế?"
- Understand: "Giải thích sự khác biệt giữa bảo hiểm y tế bắt buộc và tự nguyện?"
- Apply: "Một người lao động tự do không có hợp đồng lao động muốn tham gia BHYT. Anh/chị hãy tư vấn các bước và thủ tục cần thiết?"

Chỉ trả lời với cấu trúc JSON theo định dạng trên, không cần thêm bất kỳ nội dung giải thích nào khác.
"""

        try:
            # Gọi Groq API
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý trí tuệ nhân tạo giúp tạo câu hỏi và câu trả lời giáo dục chất lượng cao bằng Tiếng Việt. Phản hồi với đúng định dạng JSON được yêu cầu."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            end_time = time.time()
            
            # Lấy nội dung trả về
            response_content = response.choices[0].message.content
            
            logger.info(f"Thời gian sinh cặp câu hỏi-trả lời cấp độ {level}: {end_time - start_time:.2f} giây")
            
            # Xử lý response để trích xuất JSON
            # Loại bỏ các dòng không phải JSON (như markdown code block)
            json_content = response_content
            if "```json" in json_content:
                json_content = json_content.split("```json")[1]
            if "```" in json_content:
                json_content = json_content.split("```")[0]
                
            json_content = json_content.strip()
            
            # Parse JSON
            qa_pairs = json.loads(json_content)
            
            # Đảm bảo số lượng câu hỏi đúng như mong muốn
            return qa_pairs[:num_questions]
            
        except Exception as e:
            logger.error(f"Lỗi khi gọi Groq API hoặc xử lý kết quả: {e}")
            return []

def generate_questions_from_topics(
    topic_file_path: Union[str, Path],
    vector_db_path: Union[str, Path],
    output_dir: Union[str, Path] = "outputs",
    embedding_model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
    groq_model_name: str = "llama3-70b-8192",
    num_questions_per_level: int = 2,
    chunks_per_topic: int = 5
) -> List[Dict[str, Any]]:
    """
    Đọc danh sách chủ đề và sinh câu hỏi và câu trả lời cho từng chủ đề.
    
    Args:
        topic_file_path: Đường dẫn đến file chủ đề
        vector_db_path: Đường dẫn đến vector database
        output_dir: Thư mục lưu kết quả
        embedding_model_name: Tên model embedding
        groq_model_name: Tên model Groq
        num_questions_per_level: Số câu hỏi cho mỗi cấp độ Bloom
        chunks_per_topic: Số chunks sử dụng để sinh câu hỏi
        
    Returns:
        Danh sách các câu hỏi-câu trả lời với metadata
    """
    logger.info("Bắt đầu quá trình sinh câu hỏi từ các chủ đề")
    
    # Đọc danh sách chủ đề
    topic_path = Path(topic_file_path)
    if not topic_path.exists():
        logger.error(f"Không tìm thấy file chủ đề: {topic_path}")
        return []
    
    with open(topic_path, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"Đã đọc {len(topics)} chủ đề từ file {topic_path}")
    
    # Khởi tạo model embedding
    embeddings = initialize_embedding_model(embedding_model_name)
    if not embeddings:
        logger.error(f"Không thể khởi tạo model embedding '{embedding_model_name}'")
        return []
    
    # Tải vector database
    vector_db = load_vector_db(vector_db_path, embeddings)
    if not vector_db:
        logger.error(f"Không thể tải vector database từ {vector_db_path}")
        return []
    
    logger.info(f"Đã tải vector database thành công. Số lượng vectors: {vector_db.index.ntotal}")
    
    # Khởi tạo generator
    generator = GroqQuestionGenerator(model_name=groq_model_name)
    
    # Sinh câu hỏi cho từng chủ đề
    all_questions = []
    
    for i, topic in enumerate(topics, 1):
        logger.info(f"Xử lý chủ đề {i}/{len(topics)}: '{topic}'")
        
        questions_by_level = generator.generate_questions_for_topic(
            topic=topic,
            vector_db=vector_db,
            num_questions_per_level=num_questions_per_level,
            chunks_per_topic=chunks_per_topic
        )
        
        # Chuyển đổi cấu trúc dữ liệu thành list với metadata
        for level, qa_pairs in questions_by_level.items():
            for qa_pair in qa_pairs:
                all_questions.append({
                    "topic": topic,
                    "level": level,
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"]
                })
    
    # Lưu kết quả theo định dạng mới
    output_path = Path(output_dir) / "question_generation"
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"questions_with_answers_groq_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Đã lưu kết quả sinh câu hỏi và câu trả lời vào file {output_file}")
    
    return all_questions

# Hàm main để chạy từ dòng lệnh
def main():
    """
    Hàm main để chạy module từ dòng lệnh.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Sinh câu hỏi và câu trả lời từ vector database theo thang đo Bloom")
    parser.add_argument("--topics", type=str, default="data/sample_topics.txt", 
                      help="Đường dẫn đến file chứa danh sách chủ đề")
    parser.add_argument("--vector-db", type=str, 
                      default="data/gold/db_faiss_phapluat_yte_full_final",
                      help="Đường dẫn đến thư mục chứa vector database")
    parser.add_argument("--model", type=str, default="llama3-70b-8192",
                      help="Model Groq sử dụng để sinh câu hỏi")
    parser.add_argument("--questions-per-level", type=int, default=2,
                      help="Số câu hỏi mỗi cấp độ Bloom cho mỗi chủ đề")
    parser.add_argument("--chunks-per-topic", type=int, default=5,
                      help="Số chunks sử dụng cho mỗi chủ đề")
    
    args = parser.parse_args()
    
    generate_questions_from_topics(
        topic_file_path=args.topics,
        vector_db_path=args.vector_db,
        groq_model_name=args.model,
        num_questions_per_level=args.questions_per_level,
        chunks_per_topic=args.chunks_per_topic
    )

if __name__ == "__main__":
    main()