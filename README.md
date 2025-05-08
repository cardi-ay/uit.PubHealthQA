# [UIT@PubHealthQA] HCM Public Health Office Procedure Q&A Dataset

## Table of Contents
- [\[UIT@PubHealthQA\] HCM Public Health Office Procedure Q\&A Dataset](#uitpubhealthqa-hcm-public-health-office-procedure-qa-dataset)
  - [Table of Contents](#table-of-contents)
  - [🧾 Overview](#-overview)
  - [🗂️ Project Structure](#️-project-structure)
  - [Acknowledgement](#acknowledgement)
  - [🚀 Installation & Usage](#-installation--usage)

## 🧾 Overview
**UIT@PubHealthQA** là một hệ thống RAG (Retrieval-Augmented Generation) được phát triển để tìm kiếm và trả lời các câu hỏi liên quan đến luật y tế công cộng tại Việt Nam. Dự án này bao gồm:

1. **Thu thập dữ liệu (Data Acquisition)**: Thu thập các văn bản pháp luật về y tế từ các nguồn chính thống như vbpl.vn/boyte.

2. **Tiền xử lý (Preprocessing)**: Làm sạch và phân đoạn các văn bản pháp luật thành các chunks có kích thước phù hợp để tạo vector database.

3. **Vector Store**: Sử dụng FAISS để lưu trữ và truy xuất các vector embeddings từ văn bản đã được phân đoạn.

4. **Generation**: Sinh câu hỏi và câu trả lời dựa trên các văn bản pháp luật, sử dụng các mô hình ngôn ngữ lớn (LLM).

Hệ thống hỗ trợ các chức năng:
- Tìm kiếm ngữ nghĩa (semantic search) trong văn bản pháp luật y tế
- Truy xuất thông tin có trích dẫn nguồn văn bản pháp luật
- Sinh câu hỏi và câu trả lời theo các cấp độ Bloom (Remember, Understand, Apply)
- Đánh giá kết quả tìm kiếm (retrieval) với các phương pháp khác nhau

### Dataset
Dự án này sử dụng bộ dữ liệu văn bản pháp luật về y tế công cộng tại Việt Nam, được thu thập và xử lý để hỗ trợ hệ thống hỏi đáp tự động. Mỗi văn bản bao gồm:
- Thông tin metadata (số hiệu, loại văn bản, ngày hiệu lực, lĩnh vực)
- Nội dung văn bản được cấu trúc (các chương, điều, khoản, điểm)
- Vector embeddings tạo từ các đoạn văn bản

## 🗂️ Project Structure
```tex
uit.PubHealthQA/
│
├── data/                          # Tất cả dữ liệu được tổ chức theo từng giai đoạn xử lý
│   ├── bronze/                    # Dữ liệu thô (crawl, thu thập)
│   ├── silver/                    # Dữ liệu đã được làm sạch và chuẩn hóa
│   └── gold/                      # Dữ liệu cuối cùng sẵn sàng cho ML, phân tích
│       └── db_faiss_phapluat_yte_full_final/     # Vector database FAISS
│
├── notebooks/                     # Jupyter notebooks để khám phá và xử lý
│   ├── RAG.ipynb                  # Demo sử dụng RAG
│   └── question_answer_generation_groq.ipynb     # Sinh câu hỏi-đáp
│
├── src/                           # Mã nguồn Python cho xử lý dữ liệu theo module
│   ├── __init__.py                # File khởi tạo package
│   ├── data_acquisition/          # Thu thập dữ liệu (kết hợp từ crawling và ingest)
│   │   ├── __init__.py
│   │   ├── crawlLinks.py          # Thu thập links từ trang web
│   │   ├── crawlContents.py       # Thu thập nội dung văn bản
│   │   └── data_loader.py         # Tải dữ liệu từ các nguồn
│   │
│   ├── preprocessing/             # Tiền xử lý và phân đoạn dữ liệu (từ preprocess và chunking)
│   │   ├── __init__.py
│   │   ├── document_processor.py  # Xử lý văn bản
│   │   ├── text_splitter.py       # Chia nhỏ văn bản
│   │   └── chunking.py            # Phân đoạn dữ liệu thành chunks
│   │
│   ├── vector_store/              # Quản lý vector database (từ embed và retriever)
│   │   ├── __init__.py
│   │   ├── faiss_manager.py       # Quản lý FAISS vector database
│   │   └── faiss_retriever.py     # Truy xuất thông tin từ vector database
│   │
│   ├── generation/                # Sinh câu hỏi và câu trả lời
│   │   ├── __init__.py
│   │   └── question_generator.py  # Sinh câu hỏi và câu trả lời
│   │
│   └── utils/                     # Các tiện ích dùng chung
│       ├── __init__.py
│       ├── logging_utils.py       # Tiện ích ghi log
│       └── evaluation.py          # Đánh giá kết quả
│
├── outputs/                       # Kết quả đầu ra như báo cáo, log
│   ├── visualizations/
│   └── logs/
│
├── tests/                         # Unit tests cho các script
│   ├── test.py
│   └── test_evaluation_topics.py
│
├── requirements.txt               # Các thư viện Python cần thiết
├── README.md                      # Tài liệu dự án
├── .gitignore                     # Luật bỏ qua cho Git
├── LICENSE                        # Giấy phép (ví dụ: MIT)
└── update_imports.py              # Script cập nhật đường dẫn imports
```

## Acknowledgement
I would like to express my heartfelt gratitude to the following individuals for their invaluable guidance and support throughout this project:
- Ph.D. Nguyen Gia Tuan Anh – University of Information Technology, VNUHCM
- Ph.D. Duong Ngoc Hao - University of Information Technology, VNUHCM
- T.A. Tran Quoc Khanh – University of Information Technology, VNUHCM

Their expertise and encouragement were instrumental in helping us navigate challenges and achieve our objectives.

I would also like to extend my appreciation to my dedicated teammates for their significant contributions to the successful completion of this project:
- Dung Ho Tan, 23520327@gm.uit.edu.vn
- An Pham Dang, 22520027@gm.uit.edu.vn

## 🚀 Installation & Usage

### Yêu cầu
- Python 3.8+
- Thư viện PyTorch
- Thư viện FAISS (Facebook AI Similarity Search)
- Thư viện Sentence Transformers
- Thư viện Langchain

### Cài đặt
1. Clone repository:
```bash
git clone https://github.com/yourusername/uit.PubHealthQA.git
cd uit.PubHealthQA
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Tải vector database:
Đảm bảo thư mục `data/gold/db_faiss_phapluat_yte_full_final` có chứa vector database FAISS hoặc chạy script tạo vector database.

### Sử dụng

#### 1. Tìm kiếm thông tin
Sử dụng notebook `notebooks/RAG.ipynb` để tìm kiếm thông tin từ văn bản pháp luật y tế:
```python
from src.vector_store.faiss_manager import initialize_embedding_model, load_vector_db
from src.vector_store.faiss_retriever import query_documents

# Khởi tạo model embedding
embedding_model = initialize_embedding_model("bkai-foundation-models/vietnamese-bi-encoder")

# Tải vector database
vector_db = load_vector_db("data/gold/db_faiss_phapluat_yte_full_final", embedding_model)

# Tìm kiếm
results = query_documents(vector_db, "Quy định về đăng ký thuốc mới", k=3)
```

#### 2. Sinh câu hỏi và câu trả lời
Sử dụng module `generation` để sinh câu hỏi và câu trả lời từ văn bản:
```python
from src.generation.question_generator import generate_questions_from_topics

# Sinh câu hỏi và câu trả lời từ danh sách chủ đề
generate_questions_from_topics(
    topic_file_path="data/sample_topics.txt",
    vector_db_path="data/gold/db_faiss_phapluat_yte_full_final",
    groq_model_name="llama3-70b-8192"
)
```

#### 3. Đánh giá hệ thống
Sử dụng script test để đánh giá hiệu suất của hệ thống tìm kiếm:
```bash
python tests/test_evaluation_topics.py
```

#### 4. Pipeline xử lý dữ liệu
Sử dụng script `data_pipeline.py` để tự động hóa toàn bộ quy trình xử lý dữ liệu từ thu thập đến vector database:

##### Chạy toàn bộ pipeline
```bash
python data_pipeline.py
```

##### Chỉ thu thập dữ liệu
```bash
python data_pipeline.py --mode crawl --source_url https://vbpl.vn/boyte --max_pages 20
```

##### Chỉ xử lý và phân đoạn dữ liệu
```bash
python data_pipeline.py --mode process --chunk_size 500 --chunk_overlap 100
```

##### Chỉ tạo vector database
```bash
python data_pipeline.py --mode vectorize --embedding_model bkai-foundation-models/vietnamese-bi-encoder
```

##### Các tham số tùy chỉnh
- `--mode`: Chế độ chạy (`full`, `crawl`, `process`, `vectorize`)
- `--source_url`: URL nguồn để thu thập dữ liệu
- `--max_pages`: Số trang tối đa để thu thập
- `--embedding_model`: Mô hình embedding sử dụng
- `--chunk_size`: Kích thước chunk khi phân đoạn văn bản
- `--chunk_overlap`: Độ chồng lấp giữa các chunk
