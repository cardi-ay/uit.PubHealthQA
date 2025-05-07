import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import modules from src
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Now we can import modules from src
from src.vector_store.faiss_manager import initialize_embedding_model, load_vector_db
from src.vector_store.faiss_retriever import create_retrieval_function, ensemble_retrieval_with_rerank
from sentence_transformers import CrossEncoder

# Khởi tạo embedding model
embedding_model = initialize_embedding_model("bkai-foundation-models/vietnamese-bi-encoder")

# Tạo đường dẫn tuyệt đối đến vector database
faiss_path = project_root / "data" / "gold" / "db_faiss_phapluat_yte_full_final"

# Kiểm tra đường dẫn
if not faiss_path.exists():
    print(f"Không tìm thấy thư mục vector database tại: {faiss_path}")
    print(f"Đường dẫn hiện tại: {os.getcwd()}")
else:
    print(f"Tìm thấy thư mục vector database tại: {faiss_path}")

# Tải vector database
vector_db = load_vector_db(faiss_path, embedding_model)

# Tạo hai retriever function khác nhau để kết hợp
standard_retriever_func = create_retrieval_function(
    'faiss', 
    k=10, 
    with_score=True
)

mmr_retriever_func = create_retrieval_function(
    'faiss', 
    k=10, 
    use_mmr=True,
    with_score=True
)

# Cấu hình retrievers (dùng cùng một vector_db nhưng với cách truy vấn khác nhau)
retriever_configs = [
    (vector_db, standard_retriever_func, 0.6),  # Standard retrieval với trọng số 0.6
    (vector_db, mmr_retriever_func, 0.4)        # MMR retrieval với trọng số 0.4
]

# Tải Cross-encoder model cho reranking
reranker = CrossEncoder('keepitreal/vietnamese-sbert')

# Thực hiện ensemble retrieval với reranking
query = "Quy định về đăng ký kinh doanh thuốc"
results = ensemble_retrieval_with_rerank(
    query=query,
    retrievers_config=retriever_configs,
    reranker_model=reranker,
    k=5,
    fetch_k=20
)

# In kết quả
print(f"\nKết quả truy vấn cho: '{query}'\n")
for i, (doc, score) in enumerate(results):
    print(f"Kết quả #{i+1} - Score: {score:.4f}")
    print(f"Metadata: {doc.metadata}")
    content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
    print(f"Nội dung: {content_preview}\n")