# Hướng dẫn Triển khai PubHealthQA trên Render

## Chuẩn bị

1. Đảm bảo bạn đã có tài khoản [Render.com](https://render.com)
2. Đảm bảo bạn đã có GROQ API key (từ [Groq Console](https://console.groq.com))
3. Đảm bảo repository GitHub của bạn chứa các file sau:
   - `app.py` - File chính của ứng dụng
   - `app_render.py` - File khởi động cho Render
   - `Procfile` - Chỉ định lệnh khởi động
   - `render.yaml` - Cấu hình Render
   - `requirements.txt` - Các thư viện cần thiết

## Các bước triển khai

### 1. Fork/Clone repository về tài khoản GitHub của bạn

```bash
git clone https://github.com/YourUsername/uit.PubHealthQA.git
cd uit.PubHealthQA
```

### 2. Tạo Web Service trên Render

1. Truy cập [Render Dashboard](https://dashboard.render.com)
2. Chọn "New" -> "Web Service"
3. Liên kết với repository GitHub của bạn
4. Đặt tên cho dịch vụ: "pubhealthqa" (hoặc tên tùy chọn)
5. Đặt region: Singapore (gần Việt Nam)
6. Branch: main (hoặc branch khác nếu cần)
7. Runtime: Python 3
8. Build Command: `pip install -r requirements.txt`
9. Start Command: `python app_render.py`
10. Instance Type: Free

### 3. Thiết lập biến môi trường

Trong trang cấu hình Web Service, chọn "Environment":

1. Thêm biến `GROQ_API_KEY` với giá trị là API key của bạn
2. Thêm biến `PORT` với giá trị `10000`

### 4. Theo dõi quá trình triển khai

1. Nhấn "Create Web Service"
2. Theo dõi logs để phát hiện và xử lý lỗi (nếu có)
3. Sau khi deploy thành công, bạn sẽ có URL dạng: `https://pubhealthqa.onrender.com`

## Xử lý lỗi phổ biến

### Lỗi "No open ports detected"

Nếu gặp lỗi này, đảm bảo:
1. File `app_render.py` đã được tạo đúng cách
2. Biến môi trường `PORT` đã được thiết lập
3. Start Command được đặt là `python app_render.py`

### Lỗi khi tải vector database

Vector database có thể quá lớn cho tài khoản miễn phí. Giải pháp:
1. Tạo Disk trên Render (500MB miễn phí)
2. Liên kết Disk với Web Service
3. Điều chỉnh đường dẫn trong `app.py` 

### Lỗi thời gian xử lý quá lâu

Render miễn phí có giới hạn về tài nguyên:
1. Ứng dụng sẽ "ngủ" sau 15 phút không hoạt động
2. Có thể mất 30-50 giây để "đánh thức"
3. Cân nhắc nâng cấp lên gói trả phí nếu cần phản hồi nhanh hơn

## Nâng cấp và tùy chỉnh

### Nâng cấp ứng dụng
```bash
git add .
git commit -m "Cập nhật ứng dụng"
git push
```

Render sẽ tự động deploy phiên bản mới khi phát hiện thay đổi trong repository.

### Tùy chỉnh giao diện
Chỉnh sửa các file trong thư mục `templates` và `static` để thay đổi giao diện.

## Hỗ trợ

Nếu bạn gặp vấn đề, vui lòng:
1. Kiểm tra logs trên Render Dashboard
2. Tạo issue trong repository GitHub
3. Liên hệ dịch vụ hỗ trợ của Render 