# 🩺 [UIT@PubHealthQA] HCM Public Health Office Procedure Q&A Dataset

## Table of Contents
- [Overview](#-overview)
- [Dataset Structure](#-dataset-structure)
- [Project Structure](#%EF%B8%8F-project-structure)
- [Acknowledgement](#acknowledgement)

## 🧾 Overview
The **HCM Public Health Office Procedure Q&A Dataset** - `UIT@PubHealthQA` is a multi-tiered dataset project focused on building a high-quality Question Answering (QA) dataset in the public health domain, using a combination of document crawling, structured text extraction, and LLM+RAG-based question generation.

The dataset includes real-world public health inquiries submitted by citizens and corresponding expert responses provided by official sources. Each entry typically contains:
- A user-submitted question (in Vietnamese)
- An official answer provided by the local health department or relevant authority
- Metadata such as category, timestamp, and location (where applicable)

This project aims to promote transparency in public health communication and provide a valuable resource for developing AI models capable of understanding and generating Vietnamese health-related content.

## 📊 Dataset Structure
We adopt a Bronze–Silver–Gold data quality framework:
| Tier    | Description                                                                                             |
| :------ | :------------------------------------------------------------------------------------------------------ |
| 🥉 Bronze | Raw crawled data (PDFs, web pages), extracted with minimal preprocessing                                |
| 🥈 Silver | Cleaned, structured, and partially labeled data (Bloom levels, topics, etc.)                              |
| 🥇 Gold   | High-quality, fully validated dataset with inter-annotator agreement, QA pairs                           |


## 🗂️ Project Structure
```tex
UIT@PubHealthQA/
│
├── data/                          # All dataset files organized by processing stage
│   ├── bronze/                    
│   │   │── raw_QAPair.csv         # Raw data (question-answer pairs from Ministry of Health - Socialist Republic of Vietnam
│   │   └── raw_Policy.json        # Raw data (policies of Office Procedure in Public Health - National Database of Legislation
│   ├── silver/                    # Semi-silver datasets: Cleaned dataset from raw datasets
│   │   │── semi-silver_QAPair.csv
│   │   │── semi-silver_Policy.json
│   │   └── silver_dataset.csv     # Merged dataset from original cleaned dataset (semi-silver Q&A dataset) and LLM-generated dataset.
│   └── gold/                      # Final dataset ready for ML, analytics, or publishing
│       └── gold_dataset.csv
│
├── notebooks/                     # Jupyter notebooks for exploration and processing
│   ├── 01-exploration.ipynb
│   ├── 02-cleaning-transform.ipynb
│   └── 03-feature-engineering.ipynb
│
├── src/                                                      # Python scripts for modular data processing
│   ├── __init__.py
│   ├── 01-pipeline_ingestingPolicy.py                        # For downloading or scraping raw data
│   ├── 01-pipeline_ingestingQAPair.py                        # For downloading or scraping raw data
│   ├── 02-pipeline_cleaning.py                               # For cleaning and transforming data (bronze → silver)
│   ├── 02-pipeline_generatingQAPair.py                       # For generating question-answer pairs, relying on theirs topics and inference levels
│   └── utils.py                                              # Reusable functions/utilities
│
├── outputs/                       # Output artifacts like visualizations or logs
│   ├── visualizations/
│   └── logs/
│
├── tests/                         # Unit tests for your processing scripts
│   └── test_clean.py
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Git ignore rules
└── LICENSE                       # License (e.g., MIT)
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
