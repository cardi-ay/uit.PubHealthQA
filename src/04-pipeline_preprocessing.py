import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse

# Hàm đọc dữ liệu từ file CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Hàm lọc các hàng lẻ và loại bỏ các hàng có dấu "..."
def preprocess_data(df):
    # Lọc hàng lẻ, bỏ qua hàng đầu tiên (index 0)
    df_odd_rows = df.iloc[1::2]
    
    # Xóa các hàng chứa dấu "..." trong cột 'question' hoặc 'answer'
    df_cleaned = df_odd_rows[~df_odd_rows['question'].str.contains(r'\.\.\.', na=False)]  # Sửa lỗi escape
    df_cleaned = df_cleaned[~df_cleaned['answer'].str.contains(r'\.\.\.', na=False)]  # Sửa lỗi escape
    
    return df_cleaned

# Hàm chia dữ liệu thành 3 phần: train, dev, test
def split_data(df):
    # Chia ra 40 câu cho test và phần còn lại (df_temp) cho dev + train
    df_test, df_temp = train_test_split(df, test_size=len(df)-40, random_state=42)
    
    # Kiểm tra kích thước của df_temp trước khi chia tiếp
    print(f"Số lượng dữ liệu còn lại (dev + train): {len(df_temp)}")
    
    # Điều chỉnh giá trị test_size cho dev sao cho phù hợp
    test_size_dev = min(20, len(df_temp)-1)  # Đảm bảo test_size <= số mẫu còn lại
    
    # Chia phần còn lại thành 20 câu cho dev và phần còn lại cho train
    df_dev, df_train = train_test_split(df_temp, test_size=len(df_temp)-test_size_dev, random_state=42)  # Đảm bảo test_size hợp lý
    
    return df_train, df_dev, df_test

# Hàm lưu dữ liệu vào các thư mục khác nhau
def save_data(df_train, df_dev, df_test):
    # Đường dẫn đến thư mục gold (đường dẫn tương đối từ thư mục src)
    gold_path = os.path.join('..', 'data', 'gold')  # Sử dụng đường dẫn tương đối từ thư mục src đến data/gold
    
    # Đảm bảo thư mục tồn tại
    if not os.path.exists(gold_path):
        os.makedirs(gold_path)
        print(f"Thư mục '{gold_path}' đã được tạo.")

    # Lưu các tập dữ liệu vào các file CSV
    try:
        df_train.to_csv(os.path.join(gold_path, 'train.csv'), index=False)
        df_dev.to_csv(os.path.join(gold_path, 'dev.csv'), index=False)
        df_test.to_csv(os.path.join(gold_path, 'test.csv'), index=False)
        print(f"Dữ liệu đã được lưu vào thư mục '{gold_path}'.")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi lưu dữ liệu: {e}")

# Hàm main để thực thi toàn bộ quá trình
def main(file_path):
    # 1. Đọc dữ liệu từ file
    df = load_data(file_path)
    
    # 2. Xử lý dữ liệu
    df_cleaned = preprocess_data(df)
    
    # 3. Chia dữ liệu thành 3 phần (train, dev, test)
    df_train, df_dev, df_test = split_data(df_cleaned)
    
    # 4. Lưu dữ liệu vào thư mục gold
    save_data(df_train, df_dev, df_test)
    
    # 5. In ra thông báo
    print(f"Số lượng dữ liệu trong train: {len(df_train)}")
    print(f"Số lượng dữ liệu trong dev: {len(df_dev)}")
    print(f"Số lượng dữ liệu trong test: {len(df_test)}")
    print("Dữ liệu đã được xử lý và chia thành các tập train, dev, test và lưu vào thư mục 'gold'.")

# Kiểm tra xem có phải đang chạy script này không
if __name__ == "__main__":
    # Thiết lập đối số đầu vào
    parser = argparse.ArgumentParser(description="Chia dữ liệu thành các tập train, dev, test.")
    parser.add_argument('file_path', type=str, help="Đường dẫn đến file CSV đầu vào.")
    
    # Phân tích đối số
    args = parser.parse_args()
    
    # Chạy hàm main với đường dẫn file được truyền từ dòng lệnh
    main(args.file_path)
