import os
import sys
from app import app

if __name__ == "__main__":
    import uvicorn
    
    # Đảm bảo lấy PORT từ biến môi trường của Render
    port = int(os.environ.get("PORT", 10000))
    
    # In ra thông tin port để debug
    print(f"Starting server on port {port}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Chạy ứng dụng với port đã chỉ định
    uvicorn.run(app, host="0.0.0.0", port=port) 