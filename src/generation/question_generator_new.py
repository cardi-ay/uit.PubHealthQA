
import os
import logging
import time
from pathlib import Path
import re # Import regex module
from typing import List, Dict, Any, Optional, Union, Tuple
import json
from datetime import datetime
from dotenv import load_dotenv

import groq
from langchain_core.documents import Document

# Các module cục bộ (giả định đã tồn tại trong cấu trúc thư mục của bạn)
# Nếu các module này không tồn tại hoặc không thể import,
# bạn sẽ cần cung cấp các triển khai giả hoặc đảm bảo cấu trúc thư mục đúng.
try:
    from src.utils.logging_utils import setup_logger
    from src.vector_store.faiss_retriever import query_documents
    from src.vector_store.faiss_manager import initialize_embedding_model, load_vector_db
except ImportError:
    # Fallback cho trường hợp không tìm thấy các module src
    print("WARNING: Không thể import các module từ 'src'. Vui lòng đảm bảo các tệp và cấu trúc thư mục đúng.")
    print("WARNING: Sử dụng các hàm giả lập cho mục đích phát triển/kiểm tra.")

    # Thiết lập logger cơ bản nếu setup_logger không có sẵn
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_fallback = logging.getLogger("fallback_logger")
    def setup_logger(name, log_file):
        """Hàm giả lập setup_logger."""
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger_fallback.addHandler(file_handler)
        logger_fallback.info(f"Sử dụng logger giả lập cho: {name}. Tệp log: {log_file}")
        return logger_fallback

    def query_documents(vector_db, query, k, use_mmr, with_score):
        """Hàm giả lập query_documents. Trả về rỗng."""
        logger_fallback.warning(f"Hàm giả lập: query_documents được gọi cho truy vấn: '{query}'. Không có tài liệu nào được truy xuất.")
        return []

    def initialize_embedding_model(model_name):
        """Hàm giả lập initialize_embedding_model. Trả về None."""
        logger_fallback.warning(f"Hàm giả lập: initialize_embedding_model được gọi cho model: '{model_name}'. Trả về None.")
        return None

    def load_vector_db(path, embeddings):
        """Hàm giả lập load_vector_db. Trả về đối tượng DB giả lập."""
        logger_fallback.warning(f"Hàm giả lập: load_vector_db được gọi cho đường dẫn: '{path}'. Trả về DB giả lập.")
        class MockVectorDB:
            def __init__(self):
                self.index = self.MockIndex()
            class MockIndex:
                def __init__(self):
                    self.ntotal = 0
        return MockVectorDB()


# Load environment variables from .env file
load_dotenv()

# Set up logger
Path("outputs/logs").mkdir(parents=True, exist_ok=True) # Ensure log directory exists
logger = setup_logger(
    "question_generator",
    log_file=Path("outputs/logs/question_generation_groq.log")
)

class GroqQuestionGenerator:
    """
    Lớp sinh câu hỏi và câu trả lời từ chunks trong vector database sử dụng Groq API.
    Bao gồm trích dẫn nguồn và thông tin về chunks được sử dụng.
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
        
        # Định nghĩa các cấp độ của Bloom dựa trên Guideline.docx và yêu cầu của bạn
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
        """
        Sinh câu hỏi và câu trả lời có trích dẫn cho một chủ đề.
        
        Args:
            topic: Chủ đề cần sinh câu hỏi
            vector_db: Vector database (FAISS)
            num_questions_per_level: Số câu hỏi cho mỗi cấp độ Bloom
            chunks_per_topic: Số chunks sử dụng để sinh câu hỏi
            temperature: Độ sáng tạo [0-1]
            max_tokens: Số tokens tối đa cho mỗi response
            
        Returns:
            Dict chứa câu hỏi và câu trả lời theo từng cấp độ Bloom, kèm thông tin chunks
        """
        logger.info(f"Bắt đầu sinh câu hỏi cho chủ đề: '{topic}'")
        
        # Truy vấn chunks từ vector database
        retrieval_results = query_documents(
            vector_db=vector_db, 
            query=topic, 
            k=chunks_per_topic, 
            use_mmr=True,  # Sử dụng MMR để đa dạng kết quả
            with_score=True  # Lấy điểm tương đồng để đánh giá độ liên quan
        )
        
        if not retrieval_results:
            logger.warning(f"Không tìm thấy chunks phù hợp cho chủ đề: '{topic}'")
            return {level: [] for level in self.bloom_levels.keys()}
        
        # Ghép nội dung các chunks và lưu thông tin về nguồn
        context = ""
        chunk_sources = []
        
        # Xác định cấu trúc của kết quả trả về và xử lý phù hợp
        if isinstance(retrieval_results, list) and len(retrieval_results) > 0:
            # Kiểm tra cấu trúc của từng phần tử trong danh sách
            if len(retrieval_results) > 0:
                first_item = retrieval_results[0]
                
                # Trường hợp 1: Mỗi item là tuple (chunk, score)
                if isinstance(first_item, tuple) and len(first_item) == 2:
                    for i, (chunk, score) in enumerate(retrieval_results, 1):
                        # Lấy thông tin metadata từ chunk
                        metadata = chunk.metadata if hasattr(chunk, "metadata") else {}
                        source_info = {
                            "chunk_id": i,
                            "content": chunk.page_content,
                            "score": float(score),
                            "metadata": metadata,
                            "citation_text": "" # Thêm trường này để lưu thông tin trích dẫn dạng text
                        }

                        # Xây dựng chuỗi trích dẫn đầy đủ cho chunk dựa trên metadata
                        source_description = "Không rõ nguồn"
                        doc_type = metadata.get('document_type')
                        title = metadata.get('title')
                        law_id = metadata.get('law_id')
                        document_id = metadata.get('document_id')
                        
                        if doc_type and law_id:
                            source_description = f"{doc_type} số {law_id}"
                            # Thêm title nếu nó khác với doc_type và có ý nghĩa
                            if title and title.lower().replace('thông tư', '').strip() != doc_type.lower().strip():
                                source_description = f"{title} số {law_id}"
                        elif title: # Nếu chỉ có title
                            source_description = title
                        elif law_id: # Nếu chỉ có law_id (thường đã bao gồm loại và số)
                            source_description = law_id
                        elif document_id: # Fallback nếu không có gì khác
                            source_description = document_id
                        
                        source_info["citation_text"] = source_description
                        chunk_sources.append(source_info)
                        
                        # Context cho LLM chỉ bao gồm nội dung và ID chunk, không cần tên nguồn ở đây
                        context += f"[Chunk {i}]\n"
                        context += chunk.page_content + "\n\n"
                
                # Trường hợp 2: Danh sách các chunk không kèm score
                elif hasattr(first_item, 'page_content'):
                    for i, chunk in enumerate(retrieval_results, 1):
                        metadata = chunk.metadata if hasattr(chunk, "metadata") else {}
                        source_info = {
                            "chunk_id": i,
                            "content": chunk.page_content,
                            "score": 1.0,  # Không có điểm tương đồng, gán giá trị mặc định
                            "metadata": metadata,
                            "citation_text": "" # Thêm trường này để lưu thông tin trích dẫn dạng text
                        }

                        source_description = "Không rõ nguồn"
                        doc_type = metadata.get('document_type')
                        title = metadata.get('title')
                        law_id = metadata.get('law_id')
                        document_id = metadata.get('document_id')
                        
                        if doc_type and law_id:
                            source_description = f"{doc_type} số {law_id}"
                            if title and title.lower().replace('thông tư', '').strip() != doc_type.lower().strip():
                                source_description = f"{title} số {law_id}"
                        elif title:
                            source_description = title
                        elif law_id:
                            source_description = law_id
                        elif document_id:
                            source_description = document_id

                        source_info["citation_text"] = source_description
                        chunk_sources.append(source_info)
                        
                        context += f"[Chunk {i}]\n"
                        context += chunk.page_content + "\n\n"
        
        logger.info(f"Đã trích xuất {len(chunk_sources)} chunks cho chủ đề '{topic}'")
        
        result_questions = {}
        
        # Sinh câu hỏi và câu trả lời cho từng cấp độ Bloom
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
        """
        Sinh câu hỏi và câu trả lời có trích dẫn cho một cấp độ Bloom cụ thể.
        
        Args:
            topic: Chủ đề cần sinh câu hỏi
            context: Nội dung chunks từ vector database
            chunk_sources: Thông tin về các chunks được sử dụng
            level: Cấp độ Bloom ('remember', 'understand', 'apply', 'analyze')
            level_description: Mô tả chi tiết về cấp độ Bloom
            num_questions: Số câu hỏi cần sinh
            temperature: Độ sáng tạo [0-1]
            max_tokens: Số tokens tối đa cho response
            
        Returns:
            List chứa các câu hỏi, câu trả lời kèm trích dẫn và thông tin chunks ở cấp độ đã chọn
        """
        # Tạo hướng dẫn cụ thể cho từng cấp độ Bloom dựa trên Guideline.docx và yêu cầu của bạn
        level_guidance = ""
        if level == "remember":
            level_guidance = """
Định nghĩa: Cấp độ này yêu cầu người học nhớ lại hoặc nhận diện thông tin đã học trước đó. Nói đơn giản là câu trả lời copy paste những gì có sẵn trong văn bản.
Dấu hiệu nhận biết:
- Câu hỏi yêu cầu người học liệt kê, nhắc lại hoặc gọi tên các yếu tố cụ thể mà không yêu cầu phân tích hay giải thích.
- Các câu hỏi này có thể yêu cầu các sự kiện, các danh mục hoặc khái niệm cụ thể.
- Câu trả lời chứa các thông tin giống hệt như trong văn bản sẵn có hoặc trích các điều luật văn bản.
Ví dụ câu hỏi:
- "Bảo hiểm y tế là gì?"
"""
        elif level == "understand":
            level_guidance = """
Định nghĩa: Cấp độ này yêu cầu người học hiểu và có thể diễn giải lại thông tin, khái niệm, quy trình hoặc mối quan hệ giữa các yếu tố. Đây là cấp độ thể hiện khả năng giải thích hoặc so sánh, tóm tắt.
Dấu hiệu nhận biết:
- Câu hỏi yêu cầu người học diễn giải lại kiến thức, không chỉ đơn thuần là ghi nhớ.
- Câu hỏi có thể yêu cầu giải thích một khái niệm hoặc so sánh giữa các yếu tố.
- Các câu hỏi này không yêu cầu áp dụng kiến thức vào tình huống mới, mà chỉ yêu cầu hiểu các khái niệm cơ bản.
- Câu trả lời không sẵn có trong văn bản mà xoay quanh việc hiểu và diễn giải các quy định có sẵn trong văn bản.
Ví dụ câu hỏi:
- "Tóm tắt điều X thông tư Y."
"""
        elif level == "apply":
            level_guidance = """
Định nghĩa: Cấp độ này yêu cầu người học sử dụng kiến thức và kỹ năng đã học để giải quyết các vấn đề trong các tình huống mới, đặc biệt là trong các bối cảnh thực tế. Mức độ này yêu cầu áp dụng, thường là trong các tình huống thực tế.
Dấu hiệu nhận biết:
- Câu hỏi yêu cầu người học sử dụng kiến thức vào các tình huống thực tế.
- Câu hỏi yêu cầu người học triển khai kiến thức để giải quyết vấn đề thực tế hoặc thực hiện các tác vụ cụ thể.
- Câu hỏi yêu cầu thực hiện các kỹ năng đã học vào tình huống mới.
- Câu trả lời không sẵn có trong văn bản, đòi hỏi ứng dụng các quy định có sẵn để giải quyết các trường hợp thực tế.
Ví dụ câu hỏi:
- "Vận dụng điều A, B vào tình huống thực tế."
"""
        elif level == "analyze":
            level_guidance = """
Định nghĩa: Cấp độ này yêu cầu người học phân tích các yếu tố trong một vấn đề, tách biệt chúng và tìm ra mối quan hệ giữa các yếu tố. Người học sẽ phân tích thông tin để tìm ra các mẫu, sự tương đồng hoặc sự khác biệt. Ở level này, câu trả lời từ các thông tin từ 2 văn bản trở lên, yêu cầu kết hợp thông tin, suy luận từ văn bản.
Dấu hiệu nhận biết:
- Câu hỏi yêu cầu người học phân tích các yếu tố hoặc xác định mối quan hệ giữa các yếu tố.
- Các câu hỏi này thường yêu cầu người học tách biệt các phần, so sánh hoặc xác định nguyên nhân và hệ quả.
- Các câu hỏi có thể yêu cầu người học phân tích dữ liệu, xác định xu hướng hoặc phát hiện vấn đề tiềm ẩn.
- Câu trả lời không sẵn có trong văn bản mà phải suy luận, kết hợp thông tin.
Ví dụ câu hỏi:
- "Điều 8 thông tư Y khác gì điều 8 thông tư X?"
"""

        # Tạo prompt
        prompt = f"""
Bạn là một chuyên gia giáo dục y tế công cộng và luật. Nhiệm vụ của bạn là tạo ra những câu hỏi chất lượng cao và câu trả lời đầy đủ về chủ đề "{topic}" dựa trên thông tin được cung cấp dưới đây. 

Hãy tạo ra {num_questions} cặp câu hỏi và câu trả lời ở cấp độ: **{level_description}**

{level_guidance.strip()}

Thông tin tham khảo được chia thành các chunks, mỗi chunk có định danh riêng [Chunk X]:
```
{context}
```

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
```json
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
```

Chỉ trả lời với cấu trúc JSON theo định dạng trên, không cần thêm bất kỳ nội dung giải thích nào khác.
"""
        try:
            # Gọi Groq API
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
            
            # Lấy nội dung trả về
            response_content = response.choices[0].message.content
            
            logger.info(f"Thời gian sinh cặp câu hỏi-trả lời cấp độ {level}: {end_time - start_time:.2f} giây")
            
            # Xử lý response để trích xuất JSON
            json_extracted_content = ""
            if "```json" in response_content:
                json_extracted_content = response_content.split("```json", 1)[1]
                if "```" in json_extracted_content:
                    json_extracted_content = json_extracted_content.split("```", 1)[0]
            elif response_content.strip().startswith("[") and response_content.strip().endswith("]"):
                json_extracted_content = response_content.strip()
                
            if not json_extracted_content:
                logger.error(f"Could not find valid JSON content in API response for level {level}. Response: {response_content}")
                return []
                
            json_content_to_load = json_extracted_content.strip()
            qa_pairs = json.loads(json_content_to_load)
            
            processed_qa_pairs = []
            for qa_pair in qa_pairs:
                current_answer = qa_pair.get("answer", "")
                llm_source_chunk_ids = qa_pair.get("source_chunks", []) # These are the IDs from LLM, not the full details yet

                source_details_for_qa_pair = []
                actual_citation_texts = set() # To collect correctly formatted citation texts for the 'citations' field
                
                # Track citations that should be in the answer but might be missing
                citations_to_ensure_in_answer = set()

                # Iterate through the chunks identified by the LLM
                for chunk_id_any_type in llm_source_chunk_ids:
                    try:
                        chunk_id = int(str(chunk_id_any_type).strip())
                        chunk_index = chunk_id - 1
                        if 0 <= chunk_index < len(chunk_sources):
                            chunk_info = chunk_sources[chunk_index]
                            
                            # Add full chunk info to source_details
                            filtered_info = {
                            "chunk_id": chunk_id,
                            "metadata": chunk_info.get("metadata", {}),
                            "score": chunk_info.get("score", 0.0),
                            "citation_text": chunk_info.get("citation_text", "Không rõ nguồn"),
                        }


                            source_details_for_qa_pair.append(filtered_info)
                            
                            # Add the correctly formatted citation text to our set for the 'citations' field
                            actual_citation_texts.add(filtered_info["citation_text"])

                            # --- Post-processing for 'answer' field ---
                            # Construct the ideal citation string for this chunk
                            ideal_citation_for_answer = ""
                            doc_type = chunk_info["metadata"].get("document_type")
                            law_id = chunk_info["metadata"].get("law_id")
                            doc_dieu = chunk_info["metadata"].get("Điều")

                            if doc_type and law_id:
                                if doc_dieu:
                                    ideal_citation_for_answer = f"theo Điều {doc_dieu} của {doc_type} số {law_id}"
                                else:
                                    ideal_citation_for_answer = f"theo {doc_type} số {law_id}"
                            elif chunk_info["citation_text"] != "Không rõ nguồn":
                                ideal_citation_for_answer = f"theo {chunk_info['citation_text']}"

                            # Replace various forms of problematic document IDs/chunk references
                            # Prioritize replacing specific document_id mentions with the ideal citation
                            doc_id = chunk_info["metadata"].get("document_id")
                            if doc_id:
                                current_answer = re.sub(r"ID tài liệu:\s*" + re.escape(doc_id), ideal_citation_for_answer, current_answer, flags=re.IGNORECASE)
                                current_answer = re.sub(r"\b" + re.escape(doc_id) + r"\b", ideal_citation_for_answer, current_answer, flags=re.IGNORECASE) # Catch just the doc_id

                            # Remove "Theo Chunk X" or "Chunk X"
                            current_answer = re.sub(r"Theo Chunk\s*\d+[.,]?", "", current_answer, flags=re.IGNORECASE)
                            current_answer = re.sub(r"Chunk\s*\d+[.,]?", "", current_answer, flags=re.IGNORECASE)

                            # Check if the ideal citation is present in the answer after initial replacements
                            # If not, mark it to be appended
                            if ideal_citation_for_answer and ideal_citation_for_answer not in current_answer:
                                citations_to_ensure_in_answer.add(ideal_citation_for_answer)

                        else:
                            logger.warning(f"Chunk ID {chunk_id} from LLM is invalid or out of range for available chunk_sources (count: {len(chunk_sources)}).")
                    except ValueError:
                        logger.warning(f"Could not convert source_chunk ID '{chunk_id_any_type}' to an integer.")
                
                # Update the source_chunks with detailed info
                qa_pair["source_chunks"] = source_details_for_qa_pair

                # --- ALWAYS regenerate 'citations' field based on collected data ---
                # This ensures the citations field is always correctly formatted,
                # overriding any incorrect output from the LLM.
                qa_pair["citations"] = ", ".join(sorted(list(actual_citation_texts)))
                if not qa_pair["citations"]: # If no citations found, set to "Không rõ nguồn"
                    qa_pair["citations"] = "Không rõ nguồn"
                
                # --- Append missing citations to the answer if they were not naturally included ---
                if citations_to_ensure_in_answer:
                    # Filter out citations that might have been correctly inserted by LLM but not perfectly matched by `in` operator
                    # This is a heuristic to avoid double-appending if LLM did a good job but with slight variations
                    final_citations_to_append = [
                        c for c in sorted(list(citations_to_ensure_in_answer)) 
                        if c not in current_answer and not any(re.search(re.escape(c.split(' của ')[-1].replace('số ', '')), current_answer, re.IGNORECASE) for c in citations_to_ensure_in_answer)
                    ]
                    if final_citations_to_append:
                        current_answer += " (Tham khảo: " + "; ".join(final_citations_to_append) + ")"
                
                # Final cleaning for 'answer' field: remove any remaining general problematic phrases
                problematic_phrases_general = [
                    "ID tài liệu:", "Thông tư này", "Nghị định này", "Luật này", "văn bản này",
                    "Theo Chunk", # This is a general catch-all after specific chunk IDs are removed
                ]
                for phrase in problematic_phrases_general:
                    current_answer = current_answer.replace(phrase, "")
                
                # Clean up multiple spaces and strip
                qa_pair["answer"] = ' '.join(current_answer.split()).strip()

                processed_qa_pairs.append(qa_pair)
            
            # Ensure the correct number of questions are returned
            return processed_qa_pairs[:num_questions]
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error from Groq API: {e}. Content received: '{json_content_to_load if 'json_content_to_load' in locals() else response_content}'", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Lỗi khi gọi Groq API hoặc xử lý kết quả: {e}", exc_info=True)
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
    Đọc danh sách chủ đề và sinh câu hỏi và câu trả lời có trích dẫn cho từng chủ đề.
    
    Args:
        topic_file_path: Đường dẫn đến file chủ đề
        vector_db_path: Đường dẫn đến vector database
        output_dir: Thư mục lưu kết quả
        embedding_model_name: Tên model embedding
        groq_model_name: Tên model Groq
        num_questions_per_level: Số câu hỏi cho mỗi cấp độ Bloom
        chunks_per_topic: Số chunks sử dụng để sinh câu hỏi
        
    Returns:
        Danh sách các câu hỏi-câu trả lời có trích dẫn với metadata
    """
    logger.info("Bắt đầu quá trình sinh câu hỏi từ các chủ đề")
    
    # Đọc danh sách chủ đề
    topic_path = Path(topic_file_path)
    if not topic_path.exists():
        logger.error(f"Không tìm thấy file chủ đề: {topic_path}")
        return []
    
    try:
        with open(topic_path, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Đã đọc {len(topics)} chủ đề từ file {topic_path}")
    except Exception as e:
        logger.error(f"Lỗi khi đọc file chủ đề {topic_path}: {e}", exc_info=True)
        return []
    
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
    
    if hasattr(vector_db, 'index') and hasattr(vector_db.index, 'ntotal'):
        logger.info(f"Đã tải vector database thành công. Số lượng vectors: {vector_db.index.ntotal}")
    else:
        logger.info(f"Đã tải vector database thành công. Không thể xác định số lượng vectors.")
    
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
                if not isinstance(qa_pair, dict): # Add check for valid qa_pair type
                    logger.warning(f"Skipping invalid qa_pair (not a dict): {qa_pair} for topic '{topic}', level '{level}'")
                    continue
                qa_item = {
                    "topic": topic,
                    "level": level,
                    "question": qa_pair.get("question", "N/A"), # Use .get() for safety
                    "answer": qa_pair.get("answer", "N/A"), # Use .get() for safety
                    "citations": qa_pair.get("citations", ""),
                    "source_chunks": qa_pair.get("source_chunks", []),
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "model": groq_model_name,
                        "question_length": len(qa_pair.get("question", "")),
                        "answer_length": len(qa_pair.get("answer", ""))
                    }
                }
                all_questions.append(qa_item)
    
    # Lưu kết quả theo định dạng mới
    output_path = Path(output_dir) / "question_generation"
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"questions_with_citations_groq_{timestamp}.json"
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=2)
        logger.info(f"Đã lưu kết quả sinh câu hỏi và câu trả lời có trích dẫn vào file {output_file}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu kết quả vào {output_file}: {e}", exc_info=True)
    
    return all_questions

# Hàm main để chạy từ dòng lệnh
def main():
    """
    Hàm main để chạy module từ dòng lệnh.
    """
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
        output_dir=args.output_dir,
        groq_model_name=args.model,
        num_questions_per_level=args.questions_per_level,
        chunks_per_topic=args.chunks_per_topic
    )

if __name__ == "__main__":
    main()
