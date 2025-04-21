"""
Module quản lý vector database FAISS cho hệ thống RAG UIT@PubHealthQA.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

# Thử import FAISS từ các vị trí phổ biến
try:
    # Thử import từ vị trí mới hơn trước
    from langchain_faiss import FAISS
    logging.info("Đã import FAISS từ langchain_faiss")
except ImportError:
    try:
        # Nếu không được, thử import từ community (phiên bản cũ hơn)
        from langchain_community.vectorstores import FAISS
        logging.info("Đã import FAISS từ langchain_community.vectorstores")
    except ImportError:
        logging.error("Không thể import FAISS. Hãy cài đặt 'faiss-cpu' hoặc 'faiss-gpu' và 'langchain-faiss'.")
        FAISS = None

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def initialize_embedding_model(model_name: str = "bkai-foundation-models/vietnamese-bi-encoder") -> Optional[HuggingFaceEmbeddings]:
    """
    Khởi tạo mô hình embedding.
    
    Args:
        model_name: Tên mô hình Hugging Face để sử dụng
        
    Returns:
        Đối tượng HuggingFaceEmbeddings nếu thành công, None nếu thất bại
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        logging.info(f"Model Embedding '{model_name}' đã sẵn sàng.")
        return embeddings
    except Exception as e:
        logging.error(f"Lỗi khi khởi tạo model embedding: {e}", exc_info=True)
        return None

def initialize_vector_db(persist_directory: Union[str, Path], embeddings: HuggingFaceEmbeddings, 
                        load_existing: bool = True) -> Tuple[Optional[Any], bool]:
    """
    Khởi tạo vector database FAISS, tải index hiện có nếu có.
    
    Args:
        persist_directory: Đường dẫn thư mục lưu trữ FAISS
        embeddings: Model embedding đã khởi tạo
        load_existing: Có tải index hiện có không
        
    Returns:
        Tuple gồm (vector_db, index_exists)
    """
    if not FAISS or not embeddings:
        return None, False
    
    # Chuyển đổi đường dẫn sang Path nếu là string
    if isinstance(persist_directory, str):
        persist_directory = Path(persist_directory)
    
    # Đảm bảo thư mục tồn tại
    persist_directory.mkdir(parents=True, exist_ok=True)
    
    faiss_index_path = persist_directory
    faiss_file = Path(faiss_index_path).joinpath("index.faiss")
    pkl_file = Path(faiss_index_path).joinpath("index.pkl")
    
    index_exists = faiss_file.exists() and pkl_file.exists()
    vector_db = None
    
    if index_exists and load_existing:
        logging.info(f"Phát hiện index FAISS đã tồn tại tại: {faiss_index_path}")
        try:
            vector_db = FAISS.load_local(
                folder_path=str(faiss_index_path), 
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            logging.info(f"Đã tải index FAISS thành công. Số lượng vector hiện tại: {vector_db.index.ntotal}")
        except EOFError as eof:
            logging.error(f"Lỗi EOFError khi tải index FAISS: {eof}. Sẽ tạo index mới.", exc_info=True)
            vector_db = None
            index_exists = False
            try: 
                faiss_file.unlink(missing_ok=True)
                pkl_file.unlink(missing_ok=True)
                logging.info("Đã xóa file index FAISS bị lỗi.")
            except OSError as del_err: 
                logging.error(f"Không thể xóa file index FAISS bị lỗi: {del_err}")
        except Exception as e:
            logging.error(f"Lỗi khác khi tải index FAISS: {e}. Sẽ tạo index mới nếu cần.", exc_info=True)
            vector_db = None
            index_exists = False
    elif not load_existing and index_exists:
        logging.info("Cấu hình không tải index cũ. Index sẽ được tạo mới và ghi đè.")
        try: 
            faiss_file.unlink(missing_ok=True)
            pkl_file.unlink(missing_ok=True)
            logging.info("Đã xóa file index FAISS cũ để ghi đè.")
        except OSError as del_err: 
            logging.error(f"Không thể xóa file index FAISS cũ để ghi đè: {del_err}")
        index_exists = False
    else:
        logging.info(f"Chưa có index FAISS tại: {faiss_index_path}. Index sẽ được tạo mới.")
    
    return vector_db, index_exists

def save_vector_db(vector_db, persist_directory: Union[str, Path]) -> bool:
    """
    Lưu vector database FAISS vào thư mục.
    
    Args:
        vector_db: Đối tượng vector database FAISS
        persist_directory: Đường dẫn thư mục lưu trữ
        
    Returns:
        True nếu lưu thành công, False nếu thất bại
    """
    if not vector_db:
        logging.error("Không thể lưu vì vector_db là None.")
        return False
    
    try:
        if isinstance(persist_directory, Path):
            persist_directory = str(persist_directory)
        
        start_save = time.time()
        vector_db.save_local(folder_path=persist_directory)
        end_save = time.time()
        logging.info(f"Lưu trữ FAISS thành công sau {end_save - start_save:.2f} giây.")
        return True
    except Exception as e:
        logging.error(f"Lỗi khi lưu index FAISS: {e}", exc_info=True)
        return False

def load_vector_db(persist_directory: Union[str, Path], embeddings: HuggingFaceEmbeddings) -> Optional[Any]:
    """
    Tải vector database FAISS từ thư mục.
    
    Args:
        persist_directory: Đường dẫn thư mục lưu trữ
        embeddings: Model embedding đã khởi tạo
        
    Returns:
        Đối tượng vector database FAISS nếu thành công, None nếu thất bại
    """
    if not FAISS or not embeddings:
        logging.error("FAISS hoặc embeddings chưa sẵn sàng.")
        return None
    
    if isinstance(persist_directory, Path):
        persist_directory = str(persist_directory)
    
    faiss_file = Path(persist_directory).joinpath("index.faiss")
    pkl_file = Path(persist_directory).joinpath("index.pkl")
    
    if not (faiss_file.exists() and pkl_file.exists()):
        logging.error(f"Không tìm thấy file index FAISS tại {persist_directory}")
        return None
    
    try:
        vector_db = FAISS.load_local(
            folder_path=persist_directory, 
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        logging.info(f"Đã tải index FAISS thành công. Số lượng vector hiện tại: {vector_db.index.ntotal}")
        return vector_db
    except Exception as e:
        logging.error(f"Lỗi khi tải index FAISS: {e}", exc_info=True)
        return None

def create_or_update_vector_db(documents: List[Document], 
                               embeddings: HuggingFaceEmbeddings,
                               persist_directory: Union[str, Path],
                               existing_vector_db=None, 
                               add_to_existing: bool = False) -> Optional[Any]:
    """
    Tạo mới hoặc cập nhật vector database FAISS.
    
    Args:
        documents: Danh sách Document để thêm vào vector database
        embeddings: Model embedding đã khởi tạo
        persist_directory: Đường dẫn thư mục lưu trữ
        existing_vector_db: Vector database hiện có (nếu có)
        add_to_existing: Có thêm vào vector database hiện có không
        
    Returns:
        Vector database FAISS đã cập nhật, None nếu thất bại
    """
    if not FAISS or not embeddings:
        logging.error("FAISS hoặc embeddings chưa sẵn sàng.")
        return None
    
    if not documents:
        logging.warning("Không có documents để thêm vào vector database.")
        return existing_vector_db
    
    vector_db = None
    save_needed = False
    
    try:
        if existing_vector_db is not None and add_to_existing:
            logging.info(f"Đang thêm {len(documents)} chunks mới vào index FAISS hiện có (Tổng cũ: {existing_vector_db.index.ntotal})...")
            existing_vector_db.add_documents(documents=documents)
            logging.info(f"Thêm chunks mới thành công. Tổng số vector mới: {existing_vector_db.index.ntotal}")
            vector_db = existing_vector_db
            save_needed = True
        else:
            logging.info("Đang tạo index FAISS mới từ đầu...")
            vector_db = FAISS.from_documents(documents=documents, embedding=embeddings)
            logging.info(f"Tạo index FAISS mới thành công với {vector_db.index.ntotal} vectors.")
            save_needed = True
        
        if vector_db is not None and save_needed:
            save_vector_db(vector_db, persist_directory)
        
        return vector_db
    
    except Exception as e:
        logging.error(f"Lỗi khi tạo/cập nhật vector database FAISS: {e}", exc_info=True)
        return None

def query_vector_db(vector_db, query: str, k: int = 5, fetch_k: int = 20, use_mmr: bool = False, 
                    with_score: bool = False) -> List[Union[Document, Tuple[Document, float]]]:
    """
    Truy vấn vector database FAISS.
    
    Args:
        vector_db: Vector database FAISS
        query: Câu truy vấn
        k: Số lượng kết quả trả về
        fetch_k: Số lượng kết quả để lấy trước khi lọc (chỉ dùng với MMR)
        use_mmr: Có sử dụng MMR không
        with_score: Có trả về điểm số không
        
    Returns:
        Danh sách kết quả, mỗi kết quả là Document hoặc (Document, score)
    """
    if not vector_db:
        logging.error("Vector database không tồn tại.")
        return []
    
    try:
        start_time = time.time()
        
        if use_mmr:
            results = vector_db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
        elif with_score:
            results = vector_db.similarity_search_with_score(query, k=k)
        else:
            results = vector_db.similarity_search(query, k=k)
        
        end_time = time.time()
        search_time = end_time - start_time
        logging.info(f"Truy vấn hoàn tất sau {search_time:.3f} giây.")
        return results
    
    except Exception as e:
        logging.error(f"Lỗi khi truy vấn vector database: {e}", exc_info=True)
        return []