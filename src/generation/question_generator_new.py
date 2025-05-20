import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import json
from datetime import datetime
from dotenv import load_dotenv
import argparse # Moved argparse import to the top

# Attempt to import necessary libraries, especially groq and langchain_core
try:
    import groq
    from langchain_core.documents import Document
except ImportError:
    print("ERROR: Please install necessary libraries: pip install groq langchain-core python-dotenv requests argparse")
    # Define mock classes to avoid errors if not imported
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata if metadata is not None else {}

    class GroqClientMock:
        def __init__(self, api_key=None):
            self.api_key = api_key
            if not self.api_key:
                print("WARNING: Groq API key not provided for GroqClientMock.")

        class ChatCompletionsMock:
            class ChoiceMock:
                def __init__(self, content):
                    self.message = self.MessageMock(content)

                class MessageMock:
                    def __init__(self, content):
                        self.content = content

            def create(self, model, messages, temperature, max_tokens):
                print(f"WARNING: Using GroqClientMock. Model: {model}, Temperature: {temperature}")
                # Return a sample JSON structure for question/answer
                sample_response_content = """
[
  {
    "question": "Đây là câu hỏi mẫu vì Groq không được cấu hình đúng?",
    "answer": "Đây là câu trả lời mẫu. Vui lòng kiểm tra cài đặt Groq và API key.",
    "citations": "Không có văn bản pháp luật nào được trích dẫn (mẫu)",
    "source_chunks": []
  }
]
                """
                return self.ChoiceMock(sample_response_content)

        chat = ChatCompletionsMock() # Assign instance of ChatCompletionsMock to chat

    # Assign mock client to groq if import failed
    # This creates a 'groq' module-like object if 'import groq' failed
    class GroqModuleMock:
        Client = GroqClientMock
    groq = GroqModuleMock()


# Assume these files exist in the user's environment
# or provide mock functions if not found
try:
    from src.utils.logging_utils import setup_logger
    from src.vector_store.faiss_retriever import query_documents
    from src.vector_store.faiss_manager import initialize_embedding_model, load_vector_db
except ImportError:
    print("WARNING: Could not import modules from 'src'. Using mock functions.")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_fallback = logging.getLogger("fallback_logger")

    def setup_logger(name, log_file):
        logger_fallback.info(f"Using fallback logger for: {name}. Log file (if any): {log_file}")
        # Add a file handler to the fallback logger if a log file is specified
        if log_file:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger_fallback.addHandler(file_handler)
        return logger_fallback

    def query_documents(vector_db, query, k, use_mmr, with_score):
        logger_fallback.warning(f"Mock function: query_documents called for query: '{query}'. No documents retrieved.")
        # To simulate document structure for testing downstream code:
        # return [
        #     (Document(page_content=f"Sample content for {query} - chunk 1", metadata={"title": "Sample Doc 1", "law_id": "S1"}), 0.9),
        #     (Document(page_content=f"Sample content for {query} - chunk 2", metadata={"title": "Sample Doc 2", "law_id": "S2"}), 0.8)
        # ]
        return [] # Returns an empty list

    def initialize_embedding_model(model_name):
        logger_fallback.warning(f"Mock function: initialize_embedding_model called for model: '{model_name}'. Returning None.")
        return None # Returns None

    def load_vector_db(path, embeddings):
        logger_fallback.warning(f"Mock function: load_vector_db called for path: '{path}'. Returning mock DB.")
        # Create a mock object with index.ntotal attribute to avoid errors later if possible
        class MockVectorDB:
            def __init__(self):
                self.index = self.MockIndex()
            class MockIndex:
                def __init__(self):
                    self.ntotal = 0 # Simulate an empty DB
        return MockVectorDB() # Returns a mock DB object


# Load environment variables from .env file
load_dotenv()

# Set up logger
# Check if setup_logger was imported successfully
if 'setup_logger' in globals() and callable(setup_logger) and not (hasattr(setup_logger, '__doc__') and setup_logger.__doc__ and "fallback logger" in setup_logger.__doc__):
    # Create logs directory first if it doesn't exist
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        "question_generator",
        log_file=Path("outputs/logs/question_generation_groq.log")
    )
else: # Fallback if setup_logger is not defined/imported or is the mock
    if 'logger_fallback' not in globals(): # ensure fallback logger is defined
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger_fallback = logging.getLogger("fallback_logger_main")
    logger = logger_fallback # Use the fallback logger defined above
    logger.warning("setup_logger not available or is a mock, using basic/fallback logger.")


class GroqQuestionGenerator:
    def __init__(self, model_name: str = "llama3-70b-8192", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            logger.error(
                "Groq API key not found. Please set the GROQ_API_KEY environment variable "
                "or provide api_key during initialization. Attempting to use mock client if groq import failed."
            )
            # The mock client will be used automatically if 'import groq' above assigned the mock
            if not isinstance(groq.Client, type(GroqClientMock)): # If real groq was imported but key is missing
                 raise ValueError("Groq API key is missing and Groq library was loaded successfully.")

        self.client = groq.Client(api_key=self.api_key)
        logger.info(f"Initialized Groq Client with model '{model_name}' (or mock client if previous warning).")


        self.bloom_levels = {
            "understand": "Hiểu - Understand: Câu hỏi yêu cầu người học hiểu và diễn giải thông tin đã học.",
            "apply": "Áp dụng - Apply: Câu hỏi yêu cầu người học áp dụng kiến thức đã học vào tình huống cụ thể.",
            "analyze": "Phân tích - Analyze: Câu hỏi yêu cầu người học kết hợp thông tin từ nhiều nguồn để phân tích hoặc đưa ra quyết định."
        }

    def generate_questions_for_topic(self, topic: str, vector_db, num_questions_per_level: int = 2, chunks_per_topic: int = 5, temperature: float = 0.7, max_tokens: int = 1500) -> Dict[str, List[Dict[str, Any]]]:
        logger.info(f"Starting question generation for topic: '{topic}'")

        # Check if query_documents is a mock function
        is_mock_query_documents = 'query_documents' in globals() and hasattr(query_documents, '__doc__') and query_documents.__doc__ and "Mock function" in query_documents.__doc__
        if is_mock_query_documents:
             logger.warning("Using mock query_documents function.")

        retrieval_results = query_documents(
            vector_db=vector_db,
            query=topic,
            k=chunks_per_topic,
            use_mmr=True, # Assuming this is a valid param for your actual query_documents
            with_score=True
        )

        if not retrieval_results:
            logger.warning(f"No suitable chunks found for topic: '{topic}' (possibly due to mock query_documents).")
            return {level: [] for level in self.bloom_levels.keys()}

        context = ""
        chunk_sources = []

        if isinstance(retrieval_results, list) and len(retrieval_results) > 0:
            first_item = retrieval_results[0]
            is_doc_score_tuple = isinstance(first_item, tuple) and len(first_item) == 2 and hasattr(first_item[0], 'page_content')
            is_doc_only = hasattr(first_item, 'page_content') # Covers Document instances

            if is_doc_score_tuple:
                for i, (chunk_doc, score) in enumerate(retrieval_results, 1):
                    if not hasattr(chunk_doc, 'page_content'): continue # Skip if not expected format
                    metadata = chunk_doc.metadata if hasattr(chunk_doc, "metadata") else {}
                    source_info = {
                        "chunk_id": i,
                        "content": chunk_doc.page_content,
                        "score": float(score),
                        "metadata": metadata
                    }
                    chunk_sources.append(source_info)
                    context += f"[Chunk {i}] Nguồn: {metadata.get('title', 'Không rõ nguồn')}"
                    if metadata.get('law_id'): context += f", Số hiệu: {metadata.get('law_id')}"
                    context += f"\n{chunk_doc.page_content}\n\n"
            elif is_doc_only: # Handles list of Document objects
                for i, chunk_doc in enumerate(retrieval_results, 1):
                    if not hasattr(chunk_doc, 'page_content'): continue
                    metadata = chunk_doc.metadata if hasattr(chunk_doc, "metadata") else {}
                    source_info = {
                        "chunk_id": i,
                        "content": chunk_doc.page_content,
                        "score": 1.0, # Default score if not provided
                        "metadata": metadata
                    }
                    chunk_sources.append(source_info)
                    context += f"[Chunk {i}] Nguồn: {metadata.get('title', 'Không rõ nguồn')}"
                    if metadata.get('law_id'): context += f", Số hiệu: {metadata.get('law_id')}"
                    context += f"\n{chunk_doc.page_content}\n\n"
            else:
                logger.warning(f"Unsupported retrieval_results format: {type(first_item)}")


        logger.info(f"Extracted {len(chunk_sources)} chunks for topic '{topic}'")
        if not context and not is_mock_query_documents: # If context is empty and we weren't using a mock that returns empty
             logger.warning(f"Context is empty for topic '{topic}'. Generated questions might be generic.")
        elif not context and is_mock_query_documents:
             logger.info(f"Context is empty for topic '{topic}' due to mock query_documents.")


        result_questions = {}
        for level, description in self.bloom_levels.items():
            try:
                qa_pairs = self._generate_qa_with_citations_for_level(
                    topic=topic,
                    context=context if context else "Không có nội dung tham khảo được cung cấp.", # Ensure context is not empty for prompt
                    chunk_sources=chunk_sources,
                    level=level,
                    level_description=description,
                    num_questions=num_questions_per_level,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                result_questions[level] = qa_pairs
                logger.info(f"Generated {len(qa_pairs)} Q&A pairs for level {level}, topic '{topic}'")
            except Exception as e:
                logger.error(f"Error generating questions for level {level}, topic '{topic}': {e}", exc_info=True)
                result_questions[level] = []
        return result_questions

    def _generate_qa_with_citations_for_level(self, topic: str, context: str, chunk_sources: List[Dict[str, Any]], level: str, level_description: str, num_questions: int, temperature: float = 0.7, max_tokens: int = 1500) -> List[Dict[str, Any]]:
        prompt = f"""
Hãy tạo ra {num_questions} cặp câu hỏi và câu trả lời ở cấp độ: **{level_description}**

Thông tin tham khảo được chia thành các chunks, mỗi chunk có định danh riêng [Chunk X]:
```
{context}
```

Yêu cầu:
1. Câu hỏi và câu trả lời phải liên quan đến chủ đề "{topic}"
2. Câu hỏi phải phù hợp với cấp độ "{level}" trong thang đo Bloom.
3. Câu hỏi và câu trả lời phải dựa trên các thông tin được cung cấp trong các chunks (nếu context có nội dung) hoặc là câu hỏi chung về chủ đề nếu context rỗng/không liên quan.
4. Câu trả lời phải đầy đủ, chính xác và cung cấp đầy đủ thông tin dựa trên nội dung đã cho (nếu có).
5. QUAN TRỌNG: Câu trả lời phải trích dẫn CHÍNH XÁC tên văn bản luật/nghị định/thông tư liên quan (nếu có trong context).
6. QUAN TRỌNG: Trả về còn phải chỉ rõ những chunks nào đã được sử dụng làm nguồn để tạo ra câu trả lời (dưới dạng danh sách các ID, ví dụ: [1, 3, 5]), nếu context được sử dụng.
7. Định dạng phản hồi phải theo cấu trúc JSON sau:
```json
[
  {{
    "question": "Câu hỏi đầy đủ ở đây?",
    "answer": "Câu trả lời đầy đủ ở đây. Theo [tên văn bản pháp luật/luật/nghị định]...",
    "citations": "Tên đầy đủ của văn bản pháp luật được trích dẫn trong câu trả lời",
    "source_chunks": [1, 3, 5]
  }},
  {{
    "question": "Câu hỏi đầy đủ thứ hai ở đây?",
    "answer": "Câu trả lời đầy đủ thứ hai ở đây. Theo Điều X của [tên văn bản pháp luật]...",
    "citations": "Tên đầy đủ của văn bản pháp luật được trích dẫn trong câu trả lời",
    "source_chunks": [2, 4]
  }}
]
```
Nếu không có context hoặc context không đủ để tạo câu hỏi theo yêu cầu, hãy tạo câu hỏi chung về chủ đề và ghi rõ trong câu trả lời là không dựa trên context cụ thể. Nếu context là "Không có nội dung tham khảo được cung cấp.", hãy tạo câu hỏi chung.
"""
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý AI giúp tạo câu hỏi và câu trả lời giáo dục chất lượng cao về luật y tế bằng Tiếng Việt. Phản hồi với đúng định dạng JSON được yêu cầu, đảm bảo trích dẫn đầy đủ các nguồn pháp luật liên quan nếu có trong context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            end_time = time.time()
            response_content = response.choices[0].message.content
            logger.info(f"Time to generate Q&A for level {level}: {end_time - start_time:.2f}s. Model response (first 200 chars): {response_content[:200]}...")

            # Enhanced JSON extraction
            json_extracted_content = ""
            if "```json" in response_content:
                # Take content after the first ```json
                json_extracted_content = response_content.split("```json", 1)[1]
                # If there's another ```, take content before it
                if "```" in json_extracted_content:
                    json_extracted_content = json_extracted_content.split("```", 1)[0]
            elif response_content.strip().startswith("[") and response_content.strip().endswith("]"):
                 json_extracted_content = response_content.strip() # Assume the whole response is the JSON array

            if not json_extracted_content:
                logger.error(f"Could not find valid JSON content in API response for level {level}. Response: {response_content}")
                return []

            json_content_to_load = json_extracted_content.strip()
            qa_pairs = json.loads(json_content_to_load)

            processed_qa_pairs = []
            for qa_pair in qa_pairs: # Process source_chunks if present
                if "source_chunks" in qa_pair and isinstance(qa_pair["source_chunks"], list):
                    source_chunk_ids = qa_pair["source_chunks"]
                    source_details = []
                    for chunk_id_any_type in source_chunk_ids:
                        try:
                            # Attempt to convert ID to int, handling potential strings like "1", " 1 ", etc.
                            chunk_id = int(str(chunk_id_any_type).strip())
                            chunk_index = chunk_id - 1 # Assuming 1-based indexing from LLM
                            if 0 <= chunk_index < len(chunk_sources):
                                chunk_info = chunk_sources[chunk_index]
                                source_details.append({
                                    "chunk_id": chunk_id, # Store the original (now integer) ID
                                    "metadata": chunk_info.get("metadata", {}),
                                    "score": chunk_info.get("score", 0.0)
                                })
                            else:
                                logger.warning(f"Chunk ID {chunk_id} from LLM is invalid or out of range for available chunk_sources (count: {len(chunk_sources)}).")
                        except ValueError:
                            logger.warning(f"Could not convert source_chunk ID '{chunk_id_any_type}' to an integer.")
                    qa_pair["source_chunks"] = source_details
                processed_qa_pairs.append(qa_pair)

            return processed_qa_pairs[:num_questions] # Return only the requested number of questions

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error from Groq API: {e}. Content received: '{json_content_to_load if 'json_content_to_load' in locals() else response_content}'", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Error calling Groq API or processing result: {e}", exc_info=True)
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
    logger.info("Starting process to generate questions from topics.")
    topic_path = Path(topic_file_path)
    if not topic_path.exists():
        logger.error(f"Topic file not found: {topic_path}")
        return []
    try:
        with open(topic_path, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Read {len(topics)} topics from file {topic_path}")
    except Exception as e:
        logger.error(f"Error reading topic file {topic_path}: {e}", exc_info=True)
        return []


    is_mock_initialize_embedding_model = 'initialize_embedding_model' in globals() and hasattr(initialize_embedding_model, '__doc__') and initialize_embedding_model.__doc__ and "Mock function" in initialize_embedding_model.__doc__
    embeddings = initialize_embedding_model(embedding_model_name)
    if not embeddings and not is_mock_initialize_embedding_model:
        logger.error(f"Failed to initialize embedding model '{embedding_model_name}' and not using mock.")
        return []
    elif not embeddings and is_mock_initialize_embedding_model:
         logger.warning(f"Could not initialize embedding model '{embedding_model_name}' (using mock function or error).")


    is_mock_load_vector_db = 'load_vector_db' in globals() and hasattr(load_vector_db, '__doc__') and load_vector_db.__doc__ and "Mock function" in load_vector_db.__doc__
    vector_db = load_vector_db(vector_db_path, embeddings)
    if not vector_db and not is_mock_load_vector_db:
        logger.error(f"Failed to load vector database from {vector_db_path} and not using mock.")
        return []
    elif not vector_db and is_mock_load_vector_db:
        logger.warning(f"Could not load vector database from {vector_db_path} (using mock function or error).")


    if vector_db and hasattr(vector_db, 'index') and hasattr(vector_db.index, 'ntotal'):
        logger.info(f"Vector database loaded successfully (or mock DB). Number of vectors: {vector_db.index.ntotal}")
    elif vector_db:
        logger.info(f"Vector database loaded (or mock DB). Cannot determine number of vectors (attribute 'index.ntotal' not found).")
    else: # vector_db is None
        logger.error(f"Vector database is None after attempting to load from {vector_db_path}.")
        return []


    generator = GroqQuestionGenerator(model_name=groq_model_name)
    all_questions = []
    for i, topic in enumerate(topics, 1):
        logger.info(f"Processing topic {i}/{len(topics)}: '{topic}'")
        questions_by_level = generator.generate_questions_for_topic(
            topic=topic, vector_db=vector_db,
            num_questions_per_level=num_questions_per_level,
            chunks_per_topic=chunks_per_topic
        )
        for level, qa_pairs in questions_by_level.items():
            for qa_pair in qa_pairs: # qa_pair should be a dict
                if not isinstance(qa_pair, dict):
                    logger.warning(f"Skipping invalid qa_pair (not a dict): {qa_pair} for topic '{topic}', level '{level}'")
                    continue
                qa_item = {
                    "topic": topic, "level": level,
                    "question": qa_pair.get("question", "N/A - Câu hỏi không có sẵn"),
                    "answer": qa_pair.get("answer", "N/A - Câu trả lời không có sẵn"),
                    "citations": qa_pair.get("citations", ""),
                    "source_chunks": qa_pair.get("source_chunks", []), # Expecting list of dicts
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "model": groq_model_name,
                        "question_length": len(qa_pair.get("question", "")),
                        "answer_length": len(qa_pair.get("answer", ""))
                    }
                }
                all_questions.append(qa_item)

    # Ensure output directories exist
    output_main_path = Path(output_dir)
    output_main_path.mkdir(parents=True, exist_ok=True)
    output_gen_path = output_main_path / "question_generation"
    output_gen_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_gen_path / f"questions_with_citations_groq_{timestamp}.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved generated questions and answers with citations to file {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}", exc_info=True)

    return all_questions

def main():
    parser = argparse.ArgumentParser(description="Sinh câu hỏi và câu trả lời có trích dẫn từ vector database theo thang đo Bloom")
    parser.add_argument("--topics", type=str, default="data/sample_topics.txt", help="Đường dẫn đến file chứa danh sách chủ đề (mặc định: data/sample_topics.txt)")
    parser.add_argument("--vector-db", type=str, default="data/gold/db_faiss_phapluat_yte_full_final", help="Đường dẫn đến thư mục chứa vector database (mặc định: data/gold/db_faiss_phapluat_yte_full_final)")
    parser.add_argument("--model", type=str, default="llama3-70b-8192", help="Model Groq sử dụng để sinh câu hỏi (mặc định: llama3-70b-8192)")
    parser.add_argument("--questions-per-level", type=int, default=1, help="Số câu hỏi mỗi cấp độ Bloom cho mỗi chủ đề (mặc định: 1 để test nhanh)")
    parser.add_argument("--chunks-per-topic", type=int, default=3, help="Số chunks sử dụng cho mỗi chủ đề (mặc định: 3 để test nhanh)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Thư mục để lưu kết quả (mặc định: outputs)")

    args = parser.parse_args()

    # Create data directory and sample topics file if they don't exist
    Path("data").mkdir(parents=True, exist_ok=True)
    sample_topic_file = Path(args.topics)
    if str(sample_topic_file) == "data/sample_topics.txt" and not sample_topic_file.exists():
        try:
            with open(sample_topic_file, "w", encoding="utf-8") as f:
                f.write("Luật khám bệnh, chữa bệnh năm 2023\n")
                f.write("Bảo hiểm y tế cho người lao động\n")
                f.write("Quy định về quảng cáo thuốc\n")
            logger.info(f"Created sample topic file at: {sample_topic_file}")
        except Exception as e:
            logger.error(f"Could not create sample topic file at {sample_topic_file}: {e}", exc_info=True)
    
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
