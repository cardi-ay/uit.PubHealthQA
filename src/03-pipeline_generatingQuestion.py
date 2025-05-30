"""
Module sinh câu hỏi và câu trả lời từ chunks trong vector database sử dụng Groq API.
Sinh câu hỏi theo 4 thang đo của Bloom: Remember, Understand, Apply, Analyze.
Bao gồm trích dẫn nguồn và thông tin về chunks được sử dụng.
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
from datetime import datetime
from dotenv import load_dotenv

import groq
from langchain_core.documents import Document

from utils.logging_utils import setup_logger
from vector_store.faiss_retriever import query_documents
from vector_store.faiss_manager import initialize_embedding_model, load_vector_db

# Load environment variables from .env file
load_dotenv()

# Setup logger
logger = setup_logger(
    "question_generator",
    log_file=Path("outputs/logs/question_generation_groq.log")
)

class GroqQuestionGenerator:
    def __init__(self, model_name: str = "llama3-70b-8192", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Không tìm thấy Groq API key. Đặt biến môi trường GROQ_API_KEY hoặc truyền api_key.")
        self.client = groq.Client(api_key=self.api_key)
        logger.info(f"Đã khởi tạo Groq Client với model '{model_name}'")
        self.bloom_levels = {
            "remember": "Ghi nhớ (Remembering): Nhớ lại thông tin cơ bản, khái niệm, hoặc sự kiện.",
            "understand": "Thông hiểu (Understanding): Diễn giải, giải thích, và so sánh các khái niệm đã học.",
            "apply": "Áp dụng (Applying): Sử dụng kiến thức vào tình huống thực tế.",
            "analyze": "Phân tích (Analyzing): Phân tích, tìm mối quan hệ giữa các yếu tố."
        }

    def generate_questions_for_topic(
        self,
        topic: str,
        vector_db,
        num_questions_per_level: int = 2,
        chunks_per_topic: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> Dict[str, List[Dict[str, Any]]]:
        logger.info(f"Bắt đầu sinh câu hỏi cho chủ đề: '{topic}'")

        retrieval_results = query_documents(
            vector_db=vector_db,
            query=topic,
            k=chunks_per_topic,
            use_mmr=True,
            with_score=True
        )
        if not retrieval_results:
            logger.warning(f"Không tìm thấy chunks phù hợp cho chủ đề: '{topic}'")
            return {level: [] for level in self.bloom_levels.keys()}

        context = ""
        chunk_sources = []

        first_item = retrieval_results[0]
        if isinstance(first_item, tuple) and len(first_item) == 2:
            for i, (chunk, score) in enumerate(retrieval_results, 1):
                metadata = getattr(chunk, "metadata", {})
                source_info = {
                    "chunk_id": i,
                    "content": chunk.page_content,
                    "score": float(score),
                    "metadata": metadata
                }
                chunk_sources.append(source_info)

                context += f"[Chunk {i}] "
                if "title" in metadata:
                    context += f"Nguồn: {metadata.get('title', 'Không rõ nguồn')}"
                if "law_id" in metadata:
                    context += f", Số hiệu: {metadata.get('law_id', '')}"
                context += "\n"
                context += chunk.page_content + "\n\n"
        else:
            for i, chunk in enumerate(retrieval_results, 1):
                metadata = getattr(chunk, "metadata", {})
                source_info = {
                    "chunk_id": i,
                    "content": chunk.page_content,
                    "score": 1.0,
                    "metadata": metadata
                }
                chunk_sources.append(source_info)

                context += f"[Chunk {i}] "
                if "title" in metadata:
                    context += f"Nguồn: {metadata.get('title', 'Không rõ nguồn')}"
                if "law_id" in metadata:
                    context += f", Số hiệu: {metadata.get('law_id', '')}"
                context += "\n"
                context += chunk.page_content + "\n\n"

        logger.info(f"Đã trích xuất {len(chunk_sources)} chunks cho chủ đề '{topic}'")

        result_questions = {}
        for level, description in self.bloom_levels.items():
            try:
                qa_pairs = self._generate_qa_with_citations_for_level(
                    topic=topic,
                    context=context,
                    chunk_sources=chunk_sources,
                    level=level,
                    level_description=description,
                    num_questions=num_questions_per_level,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                result_questions[level] = qa_pairs
                logger.info(f"Đã sinh {len(qa_pairs)} cặp câu hỏi-trả lời cấp độ {level} cho chủ đề '{topic}'")
            except Exception as e:
                logger.error(f"Lỗi khi sinh câu hỏi cấp độ {level} cho chủ đề '{topic}': {e}", exc_info=True)
                result_questions[level] = []

        return result_questions

    def _generate_qa_with_citations_for_level(
        self,
        topic: str,
        context: str,
        chunk_sources: List[Dict[str, Any]],
        level: str,
        level_description: str,
        num_questions: int,
        temperature: float = 0.7,
        max_tokens: int = 1500
    ) -> List[Dict[str, Any]]:
        level_guidance_map = {
            "remember": """Định nghĩa: Cấp độ này yêu cầu người học nhớ lại hoặc nhận diện thông tin đã học trước đó. Nói đơn giản là câu trả lời copy paste những gì có sẵn trong văn bản.
Dấu hiệu nhận biết:
- Câu hỏi yêu cầu người học liệt kê, nhắc lại hoặc gọi tên các yếu tố cụ thể mà không yêu cầu phân tích hay giải thích.
- Các câu hỏi này có thể yêu cầu các sự kiện, các danh mục hoặc khái niệm cụ thể.
- Câu trả lời chứa các thông tin giống hệt như trong văn bản sẵn có hoặc trích các điều luật văn bản.
Ví dụ câu hỏi:
- "Bảo hiểm y tế là gì?\"""",
            "understand": """Định nghĩa: Cấp độ này yêu cầu người học hiểu và có thể diễn giải lại thông tin, khái niệm, quy trình hoặc mối quan hệ giữa các yếu tố. Đây là cấp độ thể hiện khả năng giải thích hoặc so sánh, tóm tắt.
Dấu hiệu nhận biết:
- Câu hỏi yêu cầu người học diễn giải lại kiến thức, không chỉ đơn thuần là ghi nhớ.
- Câu hỏi có thể yêu cầu giải thích một khái niệm hoặc so sánh giữa các yếu tố.
- Các câu hỏi này không yêu cầu áp dụng kiến thức vào tình huống mới, mà chỉ yêu cầu hiểu các khái niệm cơ bản.
- Câu trả lời không sẵn có trong văn bản mà xoay quanh việc hiểu và diễn giải các quy định có sẵn trong văn bản.
Ví dụ câu hỏi:
- "Tóm tắt điều X thông tư Y.\"""",
            "apply": """Định nghĩa: Cấp độ này yêu cầu người học sử dụng kiến thức và kỹ năng đã học để giải quyết các vấn đề trong các tình huống mới, đặc biệt là trong các bối cảnh thực tế. Mức độ này yêu cầu áp dụng, thường là trong các tình huống thực tế.
Dấu hiệu nhận biết:
- Câu hỏi yêu cầu người học sử dụng kiến thức vào các tình huống thực tế.
- Câu hỏi yêu cầu người học triển khai kiến thức để giải quyết vấn đề thực tế hoặc thực hiện các tác vụ cụ thể.
- Câu hỏi yêu cầu thực hiện các kỹ năng đã học vào tình huống mới.
- Câu trả lời không sẵn có trong văn bản, đòi hỏi ứng dụng các quy định có sẵn để giải quyết các trường hợp thực tế.
Ví dụ câu hỏi:
- "Vận dụng điều A, B vào tình huống thực tế.\"""",
            "analyze": """Định nghĩa: Cấp độ này yêu cầu người học phân tích các yếu tố trong một vấn đề, tách biệt chúng và tìm ra mối quan hệ giữa các yếu tố. Người học sẽ phân tích thông tin để tìm ra các mẫu, sự tương đồng hoặc sự khác biệt. Ở level này, câu trả lời từ các thông tin từ 2 văn bản trở lên, yêu cầu kết hợp thông tin, suy luận từ văn bản.
Dấu hiệu nhận biết:
- Câu hỏi yêu cầu người học phân tích các yếu tố hoặc xác định mối quan hệ giữa các yếu tố.
- Các câu hỏi này thường yêu cầu người học tách biệt các phần, so sánh hoặc xác định nguyên nhân và hệ quả.
- Các câu hỏi có thể yêu cầu người học phân tích dữ liệu, xác định xu hướng hoặc phát hiện vấn đề tiềm ẩn.
- Câu trả lời không sẵn có trong văn bản mà phải suy luận, kết hợp thông tin.
Ví dụ câu hỏi:
- "Điều 8 thông tư Y khác gì điều 8 thông tư X.\""""
        }

        level_guidance = level_guidance_map.get(level, "")

        prompt = f"""
Bạn là một chuyên gia giáo dục y tế công cộng và luật. Nhiệm vụ của bạn là tạo ra những câu hỏi chất lượng cao và câu trả lời đầy đủ về chủ đề "{topic}" dựa trên thông tin được cung cấp dưới đây.

Hãy tạo ra {num_questions} cặp câu hỏi và câu trả lời ở cấp độ: **{level_description}**

{level_guidance}

Thông tin tham khảo được chia thành các chunks, mỗi chunk có định danh riêng [Chunk X]:
\"\"\"
{context}
\"\"\"

Yêu cầu QUAN TRỌNG VỀ ĐỊNH DẠNG ĐẦU RA (JSON):
1. Câu hỏi và câu trả lời phải liên quan đến chủ đề "{topic}".
2. Câu hỏi phải phù hợp với cấp độ "{level}" trong thang đo Bloom.
3. Câu hỏi và câu trả lời phải dựa trên các thông tin được cung cấp trong các chunks.
4. Câu trả lời phải đầy đủ, chính xác và cung cấp đầy đủ thông tin dựa trên nội dung đã cho.
5. **ĐẶC BIỆT QUAN TRỌNG: TRONG TRƯỜNG "answer", BẮT BUỘC PHẢI TRÍCH DẪN CHÍNH XÁC TÊN VĂN BẢN LUẬT/NGHỊ ĐỊNH/THÔNG TƯ LIÊN QUAN (nếu có).**
   - Định dạng trích dẫn BẮT BUỘC phải là: "theo Điều X của [Loại văn bản] số [Số hiệu]" (ví dụ: "theo Điều 8 của Thông tư số 195/2014/TT-BYT").
   - Nếu không có số điều cụ thể nhưng có Loại văn bản và Số hiệu, định dạng BẮT BUỘC phải là: "theo [Loại văn bản] số [Số hiệu]" (ví dụ: "theo Thông tư số 195/2014/TT-BYT").
   - **CẢNH BÁO: TUYỆT ĐỐI KHÔNG BAO GIỜ ĐƯỢC SỬ DỤNG CÁC CỤM TỪ SAU TRONG TRƯỜNG 'answer' HOẶC TRONG CÂU TRẢ LỜI: 'ID tài liệu', 'Chunk X', 'Theo Chunk X', 'Thông tư này', 'Nghị định này', 'Luật này', 'văn bản này' hoặc bất kỳ biến thể nào của chúng. CHỈ SỬ DỤNG TÊN VĂN BẢN PHÁP LUẬT ĐẦY ĐỦ VÀ CỤ THỂ.**
   - Ví dụ trích dẫn đúng trong answer: "Bảo hiểm xã hội Việt Nam tổ chức thực hiện, thanh toán chi phí khám bệnh, chữa bệnh bảo hiểm y tế theo quy định của pháp luật về bảo hiểm y tế và Thông tư số 35/2016/TT-BYT."
6. **ĐẶC BIỆT QUAN TRỌNG: TRƯỜNG "citations" trong JSON BẮT BUỘC phải chứa TÊN ĐẦY ĐỦ VÀ CHÍNH XÁC của văn bản pháp luật được trích dẫn trong câu trả lời** (ví dụ: "Luật Khám bệnh, chữa bệnh số 15/2023/QH15" hoặc "Nghị định số 63/2010/NĐ-CP").
   - **CẢNH BÁO: TUYỆT ĐỐI KHÔNG BAO GIỜ ĐƯỢC SỬ DỤNG CÁC CỤM TỪ SAU TRONG TRƯỜNG 'citations': 'ID tài liệu', '[SỐ HIỆU]/[NĂM]/[LOẠI VĂN BẢN]', 'Theo Điều X của...', 'Thông tư này', 'Nghị định này', 'Luật này', 'văn bản này', hoặc bất kỳ phần nào của câu trích dẫn trong trường 'answer'. CHỈ CUNG CẤP TÊN ĐẦY ĐỦ VÀ CHÍNH XÁC CỦA VĂN BẢN PHÁP LUẬT.**
   - Nếu không có thông tin cụ thể, hãy sử dụng "Không rõ nguồn".
7. QUAN TRỌNG: Trường "source_chunks" trong JSON phải là một danh sách các dictionary, mỗi dictionary chứa `chunk_id`, `metadata` và `score`.
8. Định dạng phản hồi phải theo cấu trúc JSON sau:
\"\"\"
[
  {{
    "question": "Câu hỏi đầy đủ ở đây?",
    "answer": "Câu trả lời đầy đủ ở đây. Ví dụ: Theo Điều 5 của Luật Khám bệnh, chữa bệnh số 15/2023/QH15...",
    "citations": "Tên đầy đủ của văn bản pháp luật được trích dẫn trong câu trả lời (ví dụ: Luật Khám bệnh, chữa bệnh số 15/2023/QH15)",
    "source_chunks": [1, 3, 5]
  }},
  {{
    "question": "Câu hỏi đầy đủ thứ hai ở đây?",
    "answer": "Câu trả lời đầy đủ thứ hai ở đây. Ví dụ: Theo Điều 10 của Nghị định 125/2023/NĐ-CP...",
    "citations": "Tên đầy đủ của văn bản pháp luật được trích dẫn trong câu trả lời",
    "source_chunks": [2, 4]
  }}
]
\"\"\"

Chỉ trả lời với cấu trúc JSON theo định dạng trên, không cần thêm bất kỳ nội dung giải thích nào khác.
"""

        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý trí tuệ nhân tạo giúp tạo câu hỏi và câu trả lời giáo dục chất lượng cao về luật y tế bằng Tiếng Việt. Phản hồi với đúng định dạng JSON được yêu cầu, đảm bảo trích dẫn đầy đủ các nguồn pháp luật liên quan."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            end_time = time.time()

            response_content = response.choices[0].message.content
            logger.info(f"Thời gian sinh cặp câu hỏi-trả lời cấp độ {level}: {end_time - start_time:.2f} giây")

            # Xử lý response để trích xuất JSON
            json_content = response_content
            if "```json" in json_content:
                json_content = json_content.split("```json")[1]
            if "```" in json_content:
                json_content = json_content.split("```")[0]
            json_content = json_content.strip()

            qa_pairs = json.loads(json_content)

            # Thêm thông tin đầy đủ về các chunks sử dụng
            for qa_pair in qa_pairs:
                if "source_chunks" in qa_pair:
                    source_chunk_ids = qa_pair["source_chunks"]
                    source_details = []

                    for chunk_id in source_chunk_ids:
                        if isinstance(chunk_id, str) and chunk_id.isdigit():
                            chunk_id = int(chunk_id)

                        chunk_index = chunk_id - 1 if isinstance(chunk_id, int) else None

                        if chunk_index is not None and 0 <= chunk_index < len(chunk_sources):
                            chunk_info = chunk_sources[chunk_index]
                            filtered_info = {
                                "chunk_id": chunk_id,
                                "metadata": chunk_info["metadata"],
                                "score": chunk_info["score"]
                            }
                            source_details.append(filtered_info)

                    qa_pair["source_chunks"] = source_details

            return qa_pairs[:num_questions]

        except Exception as e:
            logger.error(f"Lỗi khi gọi Groq API hoặc xử lý kết quả: {e}", exc_info=True)
            return []


def generate_questions_from_topics(
    topic_file_path: Union[str, Path],
    vector_db_path: Union[str, Path],
    output_dir: Union[str, Path] = "data",
    embedding_model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
    groq_model_name: str = "llama3-70b-8192",
    num_questions_per_level: int = 2,
    chunks_per_topic: int = 5
) -> List[Dict[str, Any]]:
    logger.info("Bắt đầu quá trình sinh câu hỏi từ các chủ đề")

    topic_path = Path(topic_file_path)
    if not topic_path.exists():
        logger.error(f"Không tìm thấy file chủ đề: {topic_path}")
        return []

    with open(topic_path, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f.readlines() if line.strip()]

    logger.info(f"Đã đọc {len(topics)} chủ đề từ file {topic_path}")

    embeddings = initialize_embedding_model(embedding_model_name)
    if not embeddings:
        logger.error(f"Không thể khởi tạo model embedding '{embedding_model_name}'")
        return []

    vector_db = load_vector_db(vector_db_path, embeddings)
    if not vector_db:
        logger.error(f"Không thể tải vector database từ {vector_db_path}")
        return []

    logger.info(f"Đã tải vector database thành công. Số lượng vectors: {vector_db.index.ntotal}")

    generator = GroqQuestionGenerator(model_name=groq_model_name)

    all_questions = []

    for i, topic in enumerate(topics, 1):
        logger.info(f"Xử lý chủ đề {i}/{len(topics)}: '{topic}'")

        questions_by_level = generator.generate_questions_for_topic(
            topic=topic,
            vector_db=vector_db,
            num_questions_per_level=num_questions_per_level,
            chunks_per_topic=chunks_per_topic
        )

        for level, qa_pairs in questions_by_level.items():
            for qa_pair in qa_pairs:
                qa_item = {
                    "topic": topic,
                    "level": level,
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "citations": qa_pair.get("citations", ""),
                    "source_chunks": qa_pair.get("source_chunks", []),
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "model": groq_model_name,
                        "question_length": len(qa_pair["question"]),
                        "answer_length": len(qa_pair["answer"])
                    }
                }
                all_questions.append(qa_item)

    output_path = Path(output_dir) / "silver"
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"LLM_generated_question_answer_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    logger.info(f"Đã lưu kết quả sinh câu hỏi và câu trả lời có trích dẫn vào file {output_file}")

    return all_questions


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sinh câu hỏi và câu trả lời có trích dẫn từ vector database theo thang đo Bloom")
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
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Thư mục để lưu kết quả")

    args = parser.parse_args()

    logger.info(f"Using topic file: {args.topics}")
    logger.info(f"Using vector DB path: {args.vector_db}")
    logger.info(f"Using Groq model: {args.model}")
    logger.info(f"Questions per level: {args.questions_per_level}")
    logger.info(f"Chunks per topic: {args.chunks_per_topic}")
    logger.info(f"Output directory: {args.output_dir}")

    generate_questions_from_topics(
        topic_file_path=args.topics,
        vector_db_path=args.vector_db,
        # output_dir=args.output_dir,
        groq_model_name=args.model,
        num_questions_per_level=args.questions_per_level,
        chunks_per_topic=args.chunks_per_topic
    )


if __name__ == "__main__":
    main()
