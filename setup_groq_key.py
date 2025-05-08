"""
Script để thiết lập biến môi trường GROQ_API_KEY
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv, set_key

def setup_groq_key():
    # Tải biến môi trường từ file .env nếu tồn tại
    env_file = Path('.') / '.env'
    load_dotenv(env_file)
    
    # Kiểm tra xem GROQ_API_KEY đã tồn tại trong môi trường chưa
    groq_key = os.getenv("GROQ_API_KEY")
    
    if groq_key:
        print(f"GROQ_API_KEY đã được thiết lập trước đó.")
        continue_setup = input("Bạn có muốn thiết lập lại không? (y/n): ")
        if continue_setup.lower() != 'y':
            print("Giữ nguyên GROQ_API_KEY hiện tại.")
            return
    
    # Yêu cầu người dùng nhập API key
    new_key = input("Nhập Groq API key của bạn: ")
    
    if not new_key:
        print("API key không hợp lệ. Hủy thiết lập.")
        return
    
    # Thiết lập biến môi trường cho phiên hiện tại
    os.environ["GROQ_API_KEY"] = new_key
    print("Đã thiết lập GROQ_API_KEY cho phiên hiện tại.")
    
    # Lưu vào file .env
    try:
        if not env_file.exists():
            with open(env_file, 'w') as f:
                f.write(f"GROQ_API_KEY={new_key}\n")
        else:
            set_key(dotenv_path=str(env_file), key_to_set="GROQ_API_KEY", value_to_set=new_key)
        print(f"Đã lưu GROQ_API_KEY vào file {env_file}")
    except Exception as e:
        print(f"Lỗi khi lưu API key vào file .env: {e}")
        print("API key chỉ có hiệu lực trong phiên hiện tại.")

if __name__ == "__main__":
    setup_groq_key() 