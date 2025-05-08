# [UIT@PubHealthQA] HCM Public Health Office Procedure Q&A Dataset

## Table of Contents
- [\[UIT@PubHealthQA\] HCM Public Health Office Procedure Q\&A Dataset](#uitpubhealthqa-hcm-public-health-office-procedure-qa-dataset)
  - [Table of Contents](#table-of-contents)
  - [🧾 Tổng quan](#-tổng-quan)
  - [🗂️ Cấu trúc dự án](#️-cấu-trúc-dự-án)
  - [🤝 Lời cảm ơn](#-lời-cảm-ơn)
  - [🚀 Cài đặt & Sử dụng](#-cài-đặt--sử-dụng)
    - [Yêu cầu](#yêu-cầu)
    - [Cài đặt](#cài-đặt)
    - [Quy trình sử dụng](#quy-trình-sử-dụng)
      - [1. Thu thập dữ liệu](#1-thu-thập-dữ-liệu)
      - [2. Tiền xử lý dữ liệu](#2-tiền-xử-lý-dữ-liệu)
      - [3. Tạo vector database](#3-tạo-vector-database)
      - [4. Sinh câu hỏi và câu trả lời](#4-sinh-câu-hỏi-và-câu-trả-lời)
      - [5. Chạy ứng dụng](#5-chạy-ứng-dụng)
    - [Đánh giá hiệu suất](#đánh-giá-hiệu-suất)

## 🧾 Tổng quan
**UIT@PubHealthQA** là một hệ thống RAG (Retrieval-Augmented Generation) được phát triển để tìm kiếm và trả lời các câu hỏi liên quan đến luật y tế công cộng tại Việt Nam, đồng thời xây dựng bộ câu hỏi dựa theo 3 mức độ của thang đo Bloom cho LLMs. Dự án này bao gồm:

1. **Thu thập dữ liệu (Data Acquisition)**: Thu thập các văn bản pháp luật về y tế từ các nguồn chính thống như vbpl.vn/boyte.

2. **Tiền xử lý (Preprocessing)**: Làm sạch và phân đoạn các văn bản pháp luật thành các chunks có kích thước phù hợp để tạo vector database.

3. **Vector Store**: Sử dụng FAISS để lưu trữ và truy xuất các vector embeddings từ văn bản đã được phân đoạn.

4. **Generation**: Sinh câu hỏi và câu trả lời dựa trên các văn bản pháp luật, sử dụng các mô hình ngôn ngữ lớn (LLM) theo 3 cấp độ thang đo Bloom(Apply, Remember, Understand)

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

## 🗂️ Cấu trúc dự án
```tex
uit.PubHealthQA/
│
├── data/                          # Tất cả dữ liệu được tổ chức theo từng giai đoạn xử lý
│   ├── bronze/                    # Dữ liệu thô (crawl, thu thập)
│   │   ├── qa_pthu2_only.csv      # Dữ liệu Q&A thô
│   │   └── links_vbpl_boyte.json  # Danh sách links các văn bản pháp luật
│   │
│   ├── silver/                    # Dữ liệu đã được làm sạch và chuẩn hóa
│   │   ├── data_vbpl_boyte_full_details.json  # Dữ liệu văn bản đã được chuẩn hóa
│   │   └── data_vbpl_boyte_full_details_jupyter.json  # Phiên bản cho jupyter
│   │
│   ├── gold/                      # Dữ liệu cuối cùng sẵn sàng cho ML, phân tích
│   │   ├── db_faiss_phapluat_yte_full_final/  # Vector database FAISS
│   │   └── qa_datasets/           # Bộ dữ liệu câu hỏi đáp được tạo
│   │
│   └── topics.txt                 # Danh sách các chủ đề
│
├── notebooks/                     # Jupyter notebooks để khám phá và xử lý
│   ├── extract.ipynb              # Trích xuất các chủ đề từ bộ câu hỏi thu được trên trang thông tin hỏi đáp của bộ y tế
│   └── silver_data_eda.ipynb      # Phân tích khám phá dữ liệu silver
│
├── src/                           # Mã nguồn Python cho xử lý dữ liệu theo module
│   ├── data_acquisition/          # Thu thập dữ liệu
│   │   ├── __init__.py
│   │   ├── crawlLinks.py          # Thu thập links từ trang web
│   │   ├── crawlContents.py       # Thu thập nội dung văn bản
│   │   └── data_loader.py         # Tải dữ liệu từ các nguồn
│   │
│   ├── preprocessing/             # Tiền xử lý và phân đoạn dữ liệu
│   │   ├── __init__.py
│   │   ├── document_processor.py  # Xử lý văn bản
│   │   ├── text_splitter.py       # Chia nhỏ văn bản
│   │   └── chunking.py            # Phân đoạn dữ liệu thành chunks
│   │
│   ├── vector_store/              # Quản lý vector database
│   │   ├── __init__.py
│   │   ├── faiss_manager.py       # Quản lý FAISS vector database
│   │   └── faiss_retriever.py     # Truy xuất thông tin từ vector database
│   │
│   ├── generation/                # Sinh câu hỏi và câu trả lời
│   │   ├── __init__.py
│   │   └── question_generator.py  # Sinh câu hỏi và câu trả lời
│   │ 
│   │
│   └── utils/                     # Các tiện ích dùng chung
│       ├── __init__.py
│       ├── logging_utils.py       # Tiện ích ghi log
│       └── evaluation.py          # Đánh giá kết quả
│
├── app/                           # Ứng dụng web
│   ├── static/                    # Tài nguyên tĩnh (CSS, JS, hình ảnh)
│   └──── templates/               # Templates HTML
│
├── outputs/                       # Kết quả đầu ra như báo cáo, log
│   ├── visualizations/
│   └── logs/
│
├── tests/                         # Unit tests cho các script
│   ├── test.py
│   └── test_evaluation_topics.py
│
├── app.py                         # Ứng dụng chính
├── data_pipeline.py               # Pipeline tự động hóa quy trình xử lý dữ liệu
├── run_question_generator.py      # Script chạy module sinh câu hỏi
├── requirements.txt               # Các thư viện Python cần thiết
├── README.md                      # Tài liệu dự án
├── .gitignore                     
└── setup_groq_key.py              # Cấu hình cho API key
```

## 🤝 Lời cảm ơn
Xin chân thành cảm ơn các cá nhân sau đã hướng dẫn và hỗ trợ cho dự án này:
- TS. Nguyễn Gia Tuấn Anh – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
- GVHD Trần Quốc Khánh – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM

Chuyên môn và sự khích lệ của họ đã giúp chúng em vượt qua thách thức và đạt được mục tiêu.

Xin gửi lời cảm ơn đến các thành viên trong nhóm đã đóng góp đáng kể vào sự thành công của dự án:
- Hồ Tấn Dũng
- Nguyễn Hoàng Long 

## 🚀 Cài đặt & Sử dụng

### Yêu cầu
- Python 3.8+
- Thư viện PyTorch
- Thư viện FAISS (Facebook AI Similarity Search)
- Thư viện Sentence Transformers
- Thư viện Langchain
- Thư viện Flask (cho web app)

### Cài đặt
1. Clone repository:
```bash
git clone https://github.com/yourusername/uit.PubHealthQA.git
cd uit.PubHealthQA
```

2. Tạo và kích hoạt môi trường ảo (khuyến nghị):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

3. Cài đặt các thư viện cần thiết (Đảm bảo thiết bị đã có cài đặt Python 3.8 trở lên):
```bash
pip install -r requirements.txt
```

### Quy trình sử dụng

#### 1. Thu thập dữ liệu
Thu thập văn bản pháp luật từ nguồn vbpl.vn/boyte:

```bash
# Thu thập links văn bản
python data_pipeline.py --mode crawl --source_url https://vbpl.vn/boyte --max_pages 20

# Thu thập nội dung văn bản
python data_pipeline.py --mode extract --input_file data/bronze/links_vbpl_boyte.json --output_file data/silver/data_vbpl_boyte_full_details.json
```

Hoặc sử dụng trực tiếp module Python:
```python
from src.data_acquisition.crawlLinks import crawl_links
from src.data_acquisition.crawlContents import extract_contents

# Thu thập links
links = crawl_links("https://vbpl.vn/boyte", max_pages=20)

# Thu thập nội dung
documents = extract_contents(links, output_file="data/silver/data_vbpl_boyte_full_details.json")
```

#### 2. Tiền xử lý dữ liệu
Xử lý và phân đoạn văn bản pháp luật:

```bash
python data_pipeline.py --mode process --input_file data/silver/data_vbpl_boyte_full_details.json --chunk_size 500 --chunk_overlap 100
```

Hoặc sử dụng trực tiếp module Python:
```python
from src.preprocessing.document_processor import process_documents
from src.preprocessing.chunking import chunk_documents

# Xử lý văn bản
processed_docs = process_documents("data/silver/data_vbpl_boyte_full_details.json")

# Phân đoạn văn bản
chunks = chunk_documents(processed_docs, chunk_size=500, chunk_overlap=100)
```

#### 3. Tạo vector database
Tạo vector database FAISS từ các đoạn văn bản đã xử lý:

```bash
python data_pipeline.py --mode vectorize --input_chunks data/silver/processed_chunks.json --embedding_model bkai-foundation-models/vietnamese-bi-encoder --output_dir data/gold/db_faiss_phapluat_yte_full_final
```

Hoặc sử dụng trực tiếp module Python:
```python
from src.vector_store.faiss_manager import create_vector_db

# Tạo vector database
vector_db = create_vector_db(
    chunks_file="data/silver/processed_chunks.json",
    output_dir="data/gold/db_faiss_phapluat_yte_full_final",
    embedding_model_name="bkai-foundation-models/vietnamese-bi-encoder"
)
```

#### 4. Sinh câu hỏi và câu trả lời
4.1 Thiết lập API Key:
- Vào trang [Groq](https://console.groq.com/home) đăng ký và tạo API Key
- Chạy lệnh sau đây và nhập API Key:

```bash
python setup_groq_key.py
```

4.2 Tạo câu hỏi và câu trả lời từ các chủ đề:

```bash
python run_question_generator.py --topic_file data/topics.txt --vector_db data/gold/db_faiss_phapluat_yte_full_final --model llama3-70b-8192 --output_dir data/gold/qa_datasets
```

Hoặc sử dụng notebook:
```bash
jupyter notebook notebooks/question_answer_generation_groq.ipynb
```

#### 5. Chạy ứng dụng 
Thiết lập API Key Làm theo bước [4.2](#4-sinh-câu-hỏi-và-câu-trả-lời)
Khởi động web app:

```bash
python app.py
```

Mặc định, ứng dụng sẽ chạy tại địa chỉ: http://localhost:8000
Demo app:
![Demo app](img/demo.png)

Các tham số tùy chỉnh:
```bash
python app.py --port 8080 --vector_db data/gold/db_faiss_phapluat_yte_full_final --embedding_model bkai-foundation-models/vietnamese-bi-encoder --llm_model llama3-8b-8192
```


### Đánh giá hiệu suất
Đánh giá hiệu suất hệ thống:

```bash
python tests/test_evaluation_topics.py --vector_db data/gold/db_faiss_phapluat_yte_full_final --qa_dataset data/gold/qa_datasets/qa_dataset.json --metrics precision recall
```

Kết quả đánh giá sẽ được lưu trong thư mục `outputs/`.
