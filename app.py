"""
Backend FastAPI cho ứng dụng chatbot sức khỏe công cộng sử dụng hệ thống RAG & LLM.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import groq
from langchain_core.documents import Document

from src.utils.logging_utils import setup_logger
from src.vector_store.faiss_retriever import query_documents, optimize_retrieval
from src.vector_store.faiss_manager import initialize_embedding_model, load_vector_db

load_dotenv()

logger = setup_logger(
    "pubhealth_chatbot",
    log_file=Path("outputs/logs/chatbot.log")
)

app = FastAPI(
    title="PubHealthQA Chatbot",
    description="Chatbot sức khỏe công cộng sử dụng RAG và Groq LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    
class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    retrieval_time: float = 0
    generation_time: float = 0

vector_db = None
embeddings = None
groq_client = None

DEFAULT_VECTOR_DB_PATH = "data/gold/db_faiss_phapluat_yte_full_final"
DEFAULT_EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
DEFAULT_LLM_MODEL = "llama3-70b-8192"

@app.on_event("startup")
async def startup_event():
    """Khởi tạo các tài nguyên khi ứng dụng khởi động"""
    global vector_db, embeddings, groq_client
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY không tìm thấy trong biến môi trường!")
        raise ValueError("GROQ_API_KEY không được đặt. Hãy chạy setup_groq_key.py trước!")
    
    groq_client = groq.Client(api_key=groq_api_key)
    logger.info(f"Đã khởi tạo Groq client")
    
    embeddings = initialize_embedding_model(DEFAULT_EMBEDDING_MODEL)
    if not embeddings:
        logger.error(f"Không thể khởi tạo model embedding '{DEFAULT_EMBEDDING_MODEL}'")
        raise ValueError(f"Không thể khởi tạo model embedding: {DEFAULT_EMBEDDING_MODEL}")
    
    vector_db = load_vector_db(DEFAULT_VECTOR_DB_PATH, embeddings)
    if not vector_db:
        logger.error(f"Không thể tải vector database từ {DEFAULT_VECTOR_DB_PATH}")
        raise ValueError(f"Không thể tải vector database: {DEFAULT_VECTOR_DB_PATH}")
    
    logger.info(f"Ứng dụng đã khởi động thành công. Vector database có {vector_db.index.ntotal} vectors.")

@app.get("/")
async def read_root(request: Request):
    """Endpoint chính trả về trang HTML cho giao diện chatbot"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """API endpoint cho chức năng chat"""
    global vector_db, groq_client
    
    if not vector_db or not groq_client:
        raise HTTPException(
            status_code=503, 
            detail="Hệ thống chưa được khởi tạo đầy đủ. Vui lòng thử lại sau."
        )
    
    query = request.message
    chat_history = request.history
    
    start_time = time.time()
    
    try:
        retrieval_start = time.time()
        docs_with_score = optimize_retrieval(
            vector_db=vector_db,
            query=query,
            k=5,  
            preprocess_query=True
        )
        retrieval_time = time.time() - retrieval_start
        
        context = ""
        sources = []
        
        for i, item in enumerate(docs_with_score):
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                
                context += f"[Tài liệu {i+1}] "
                if "title" in metadata:
                    context += f"Nguồn: {metadata.get('title', 'Không rõ nguồn')}"
                if "law_id" in metadata:
                    context += f", Số hiệu: {metadata.get('law_id', '')}"
                context += "\n"
                context += doc.page_content + "\n\n"
                
                source_info = {
                    "id": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "similarity": float(score),
                    "metadata": metadata
                }
                sources.append(source_info)
        
        llm_start = time.time()
        
        # Tạo prompt
        messages = [
            {"role": "system", "content": """Bạn là trợ lý sức khỏe công cộng và pháp luật y tế thông minh, 
nhiệm vụ của bạn là trả lời các câu hỏi dựa trên thông tin y tế và pháp luật chính xác.
Hãy trả lời bằng tiếng Việt, ngắn gọn, dễ hiểu và chính xác.
Dựa vào thông tin được cung cấp, nếu không có thông tin đầy đủ thì hãy nói rõ.
Luôn trích dẫn nguồn thông tin pháp luật chính xác (nếu có) như tên văn bản, điều khoản.
Trình bày câu trả lời theo đoạn văn có cấu trúc tốt, dễ hiểu."""}
        ]
        
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role and content:
                messages.append({"role": role, "content": content})
        
        prompt = f"""Người dùng hỏi: {query}

Dưới đây là các tài liệu y tế và pháp luật liên quan giúp bạn trả lời:

{context}

Hãy trả lời câu hỏi của người dùng dựa trên thông tin từ các tài liệu trên. Nếu tài liệu không cung cấp đủ thông tin để trả lời, hãy nói rõ. Luôn trích dẫn nguồn thông tin từ các văn bản pháp luật nếu có (ví dụ: Theo Luật X, Điều Y...).
Trả lời súc tích, dễ hiểu nhưng đầy đủ thông tin quan trọng."""

        messages.append({"role": "user", "content": prompt})
        
        response = groq_client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        generation_time = time.time() - llm_start
        
        return ChatResponse(
            answer=answer,
            sources=sources[:3],  
            retrieval_time=retrieval_time,
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý câu hỏi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))  # Render cung cấp PORT qua biến môi trường
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)