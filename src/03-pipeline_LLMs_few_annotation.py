import os
import logging
import json
import argparse
from dotenv import load_dotenv
import groq # Ensure this is 'groq' for the client, not 'openai'
import pandas as pd
from typing import Optional, Dict, Any, List # Đã thêm List ở đây nếu cần cho type hints
import re # Import regex for robust JSON extraction
from pathlib import Path # <-- Đảm bảo dòng này có ở đây

# Tải các biến môi trường từ file .env
load_dotenv()

def setup_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Cấu hình và trả về một logger để ghi log ra console và file.
    
    Args:
        name (str): Tên của logger.
        log_file (Path): Đường dẫn đầy đủ đến file log.
        level (int): Mức độ log tối thiểu để ghi (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: Đối tượng logger đã được cấu hình.
    """
    # Tạo thư mục chứa file log nếu nó chưa tồn tại
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Tránh thêm các handler trùng lặp
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler để ghi log vào file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler để in log ra console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

# --- Cấu hình Logger của sử dụng hàm mới ---
log_file_path = Path(__file__).parent.parent / "logs" / "bloom_annotation.log"

# Khởi tạo logger bằng hàm tùy chỉnh
logger = setup_logger("bloom_annotation", log_file=log_file_path)
# --- Kết thúc Cấu hình Logger ---

class BloomLabeler:
    def __init__(self, model_name: str = "llama3-70b-8192", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Không tìm thấy Groq API key. Đặt biến môi trường GROQ_API_KEY hoặc truyền api_key.")
        self.client = groq.Client(api_key=self.api_key)
        logger.info(f"Đã khởi tạo Groq Client với model '{model_name}'")
        
        # Các mức độ Bloom
        self.bloom_levels = {
            "remember": "Ghi nhớ (Remembering): Nhớ lại thông tin cơ bản, khái niệm, hoặc sự kiện.",
            "understand": "Thông hiểu (Understanding): Diễn giải, giải thích, và so sánh các khái niệm đã học.",
            "apply": "Áp dụng (Applying): Sử dụng kiến thức vào tình huống thực tế.",
            "analyze": "Phân tích (Analyzing): Phân tích, tìm mối quan hệ giữa các yếu tố."
        }

    def label_bloom_taxonomy(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Gửi câu hỏi và câu trả lời vào Groq API để tự động gắn nhãn Bloom.
        Trả về kết quả dưới dạng JSON với câu hỏi, câu trả lời, và nhãn Bloom.
        """
        
        # Cấu trúc prompt yêu cầu LLM gắn nhãn Bloom cho câu hỏi và câu trả lời
        # Emphasize ONLY JSON output
        prompt = f"""
        Bạn là một chuyên gia giáo dục. Nhiệm vụ của bạn là gắn nhãn Bloom cho câu hỏi và câu trả lời dưới đây, dựa trên các cấp độ sau:
        - remember:
        Định nghĩa: Cấp độ này yêu cầu người học nhớ lại hoặc nhận diện thông tin đã học trước đó. Nói đơn giản là câu trả lời copy paste những gì có sẵn trong văn bản.
Dấu hiệu nhận biết:
- Câu hỏi yêu cầu người học liệt kê, nhắc lại hoặc gọi tên các yếu tố cụ thể mà không yêu cầu phân tích hay giải thích.
- Các câu hỏi này có thể yêu cầu các sự kiện, các danh mục hoặc khái niệm cụ thể.
- Câu trả lời chứa các thông tin giống hệt như trong văn bản sẵn có hoặc trích các điều luật văn bản.
Ví dụ câu hỏi:
- "Bảo hiểm y tế là gì?\
        - Understand (Thông hiểu)
        Định nghĩa: Cấp độ này yêu cầu người học hiểu và có thể diễn giải lại thông tin, khái niệm, quy trình hoặc mối quan hệ giữa các yếu tố. Đây là cấp độ thể hiện khả năng giải thích hoặc so sánh, tóm tắt.
Dấu hiệu nhận biết:
- Câu hỏi yêu cầu người học diễn giải lại kiến thức, không chỉ đơn thuần là ghi nhớ.
- Câu hỏi có thể yêu cầu giải thích một khái niệm hoặc so sánh giữa các yếu tố.
- Các câu hỏi này không yêu cầu áp dụng kiến thức vào tình huống mới, mà chỉ yêu cầu hiểu các khái niệm cơ bản.
- Câu trả lời không sẵn có trong văn bản mà xoay quanh việc hiểu và diễn giải các quy định có sẵn trong văn bản.
Ví dụ câu hỏi:
- "Tóm tắt điều X thông tư Y.\
        - Apply (Áp dụng): 
        Định nghĩa: Cấp độ này yêu cầu người học sử dụng kiến thức và kỹ năng đã học để giải quyết các vấn đề trong các tình huống mới, đặc biệt là trong các bối cảnh thực tế. Mức độ này yêu cầu áp dụng, thường là trong các tình huống thực tế.
Dấu hiệu nhận biết:
- Câu hỏi yêu cầu người học sử dụng kiến thức vào các tình huống thực tế.
- Câu hỏi yêu cầu người học triển khai kiến thức để giải quyết vấn đề thực tế hoặc thực hiện các tác vụ cụ thể.
- Câu hỏi yêu cầu thực hiện các kỹ năng đã học vào tình huống mới.
- Câu trả lời không sẵn có trong văn bản, đòi hỏi ứng dụng các quy định có sẵn để giải quyết các trường hợp thực tế.
Ví dụ câu hỏi:
- "Vận dụng điều A, B vào tình huống thực tế.\
        - Analyze (Phân tích):
        Định nghĩa: Cấp độ này yêu cầu người học phân tích các yếu tố trong một vấn đề, tách biệt chúng và tìm ra mối quan hệ giữa các yếu tố. Người học sẽ phân tích thông tin để tìm ra các mẫu, sự tương đồng hoặc sự khác biệt. Ở level này, câu trả lời từ các thông tin từ 2 văn bản trở lên, yêu cầu kết hợp thông tin, suy luận từ văn bản.
Dấu hiệu nhận biết:
- Câu Trả lời có chứa dẫn chứng từ 2 văn bản trở lên
- Câu hỏi yêu cầu người học phân tích các yếu tố hoặc xác định mối quan hệ giữa các yếu tố.
- Các câu hỏi này thường yêu cầu người học tách biệt các phần, so sánh hoặc xác định nguyên nhân và hệ quả.
- Các câu hỏi có thể yêu cầu người học phân tích dữ liệu, xác định xu hướng hoặc phát hiện vấn đề tiềm ẩn.
- Câu trả lời không sẵn có trong văn bản mà phải suy luận, kết hợp thông tin.
Ví dụ câu hỏi:
- "Điều 8 thông tư Y khác gì điều 8 thông tư X.\

        Dưới đây là câu hỏi và câu trả lời cần gắn nhãn Bloom:

        Câu hỏi: "{question}"
        Câu trả lời: "{answer}"

        Gắn nhãn Bloom cho câu hỏi và câu trả lời này theo các mức độ Bloom.
        Cung cấp kết quả gắn nhãn Bloom cho câu hỏi và câu trả lời dưới dạng JSON. Đảm bảo rằng nhãn Bloom được đặt trong khóa 'bloom_label'.

        Chỉ trả về JSON hợp lệ và không có bất kỳ văn bản giải thích nào khác.

        Ví dụ JSON đầu ra:
        ```json
        {{
          "question": "Bảo hiểm y tế là gì?",
          "answer": "Bảo hiểm y tế là một hình thức bảo hiểm giúp chi trả chi phí khám chữa bệnh.",
          "bloom_label": "Remember (Ghi nhớ): Nhớ lại thông tin cơ bản, khái niệm, hoặc sự kiện."
        }}
        ```
        """

        try:
            # Gửi prompt đến Groq API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": "Bạn là trợ lý trí tuệ nhân tạo giúp gắn nhãn Bloom cho câu hỏi và câu trả lời. Chỉ trả về JSON hợp lệ, không thêm bất kỳ văn bản nào khác."}, # Even stronger instruction
                                 {"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500,
                response_format={"type": "json_object"} # Explicitly request JSON object
            )

            response_content = response.choices[0].message.content
            logger.info(f"Nhận kết quả từ Groq API.")

            # Phân tích kết quả trả về và trả về nhãn Bloom
            return self._parse_bloom_label_response(response_content, question, answer) 

        except groq.APIError as e: # Corrected typo here
            logger.error(f"Lỗi API Groq: {e}", exc_info=True)
            return {"question": question, "answer": answer, "bloom_label": "Error (API Error)"}
        except Exception as e:
            logger.error(f"Lỗi khi gọi Groq API hoặc xử lý kết quả: {e}", exc_info=True)
            return {"question": question, "answer": answer, "bloom_label": "Error (Processing Error)"}


    def _parse_bloom_label_response(self, response_content: str, original_question: str, original_answer: str) -> Dict[str, Any]:
        """
        Phân tích kết quả trả về từ Groq API để trích xuất nhãn Bloom.
        Thêm khả năng trích xuất JSON từ chuỗi có chứa văn bản phụ.
        """
        try:
            # Attempt to find a JSON block within the response content
            # This regex looks for a block starting with `{` and ending with `}`
            # It's flexible enough to handle leading/trailing text.
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            
            if json_match:
                json_string = json_match.group(0)
                # Attempt to parse the extracted JSON string
                bloom_data = json.loads(json_string)
            else:
                # If no JSON block is found, try direct load (less robust)
                bloom_data = json.loads(response_content)
            
            # Extract the bloom_label, defaulting if not found or if the LLM used "Câu hỏi" or "Nhãn Bloom"
            bloom_label = bloom_data.get("bloom_label", "")
            if not bloom_label: # Fallback for common LLM mistakes
                bloom_label = bloom_data.get("Nhãn Bloom", "")
            if not bloom_label:
                bloom_label = bloom_data.get("Bloom's Taxonomy", "Unknown (Label Not Found)")
            
            # Return the extracted data along with the original question and answer for consistency
            return {
                "question": original_question, 
                "answer": original_answer,    
                "bloom_label": bloom_label
            }
        except json.JSONDecodeError:
            logger.error(f"Không thể phân tích phản hồi JSON từ Groq API. Nội dung: {response_content}")
            return {
                "question": original_question,
                "answer": original_answer,
                "bloom_label": "Unknown (Invalid JSON Response)"
            }
        except Exception as e:
            logger.error(f"Lỗi không xác định khi phân tích phản hồi: {e} - Nội dung: {response_content}")
            return {
                "question": original_question,
                "answer": original_answer,
                "bloom_label": "Unknown (General Parsing Error)"
            }

    def process_and_label_data(self, input_file: str, output_file: str) -> None:
        """
        Tiến hành gắn nhãn Bloom cho tất cả câu hỏi và câu trả lời trong file đầu vào và lưu kết quả ra file đầu ra.
        """
        
        # Đọc dữ liệu từ file CSV sử dụng pandas
        try:
            df = pd.read_csv(input_file)
        except FileNotFoundError:
            logger.error(f"Không tìm thấy file đầu vào: {input_file}")
            return
        except Exception as e:
            logger.error(f"Lỗi khi đọc file CSV: {e}")
            return
        
        # Giả sử dữ liệu CSV có 2 cột 'question' và 'answer'
        if 'question' not in df.columns or 'answer' not in df.columns:
            logger.error("File CSV phải có cột 'question' và 'answer'.")
            return

        qa_pairs = df[['question', 'answer']].to_dict(orient='records')

        # Gắn nhãn Bloom cho các cặp câu hỏi, câu trả lời
        labeled_qa_pairs = []
        for i, qa_pair in enumerate(qa_pairs):
            question = str(qa_pair.get("question", "")).strip() 
            answer = str(qa_pair.get("answer", "")).strip()     

            if question and answer:
                logger.info(f"Đang xử lý cặp Q&A {i+1}/{len(qa_pairs)}: '{question}'")
                labeled_qa = self.label_bloom_taxonomy(question, answer)
                
                # Only append if a result is returned (even if label is 'Unknown')
                if labeled_qa:
                    labeled_qa_pairs.append(labeled_qa)
                else:
                    logger.warning(f"Không nhận được kết quả từ API cho cặp Q&A: Câu hỏi: '{question}'")
            else:
                logger.warning(f"Bỏ qua cặp Q&A không đầy đủ ở dòng {i+1}: Câu hỏi: '{question}', Trả lời: '{answer}'")

        # Lưu kết quả vào file đầu ra
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(labeled_qa_pairs, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu kết quả vào {output_file}. Tổng số cặp được gắn nhãn: {len(labeled_qa_pairs)}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu kết quả vào file: {e}")


def main():
    # Sử dụng argparse để truyền tham số từ lệnh
    parser = argparse.ArgumentParser(description="Gắn nhãn Bloom cho câu hỏi và câu trả lời từ file CSV")
    parser.add_argument("--input-file", type=str, required=True, help="Đường dẫn đến file CSV chứa câu hỏi và câu trả lời")
    parser.add_argument("--output-file", type=str, required=True, help="Đường dẫn đến file JSON lưu kết quả gắn nhãn")

    args = parser.parse_args()

    # Khởi tạo đối tượng BloomLabeler
    labeler = BloomLabeler(model_name="llama3-70b-8192")

    # Gắn nhãn cho các câu hỏi và câu trả lời
    labeler.process_and_label_data(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
