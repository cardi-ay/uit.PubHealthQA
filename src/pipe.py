import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse
import logging

# Cấu hình logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_qa_csv(
    input_csv_path: str,
    output_train_csv_path: str,
    output_test_csv_path: str,
    test_size: float = 0.3,
    random_state: int = 42
) -> None:
    """
    Chia file CSV chứa các cặp câu hỏi-câu trả lời thành tập huấn luyện và tập kiểm tra.

    Args:
        input_csv_path (str): Đường dẫn đến file CSV đầu vào.
        output_train_csv_path (str): Đường dẫn để lưu file CSV của tập huấn luyện.
        output_test_csv_path (str): Đường dẫn để lưu file CSV của tập kiểm tra.
        test_size (float): Tỷ lệ của tập kiểm tra (mặc định là 0.3 cho 30%).
        random_state (int): Hạt giống ngẫu nhiên để đảm bảo khả năng tái tạo kết quả.
    """
    logger.info(f"Bắt đầu chia dữ liệu từ: {input_csv_path}")

    # Kiểm tra xem file đầu vào có tồn tại không
    if not os.path.exists(input_csv_path):
        logger.error(f"Lỗi: Không tìm thấy file đầu vào '{input_csv_path}'.")
        return

    try:
        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(input_csv_path, encoding='utf-8')
        logger.info(f"Đã đọc thành công {len(df)} dòng từ '{input_csv_path}'.")

        # Kiểm tra xem các cột 'question' và 'answer' có tồn tại không
        if 'question' not in df.columns or 'answer' not in df.columns:
            logger.error("Lỗi: File CSV phải chứa cả cột 'question' và 'answer'.")
            return

        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        # stratify=None vì không có nhãn phân loại cụ thể cần duy trì tỷ lệ
        train_df, test_df = train_test_split(
            df[['question', 'answer']], # Chỉ chọn 2 cột này
            test_size=test_size,
            random_state=random_state
        )

        logger.info(f"Đã chia dữ liệu:")
        logger.info(f"- Tập huấn luyện (Train set): {len(train_df)} dòng ({len(train_df)/len(df):.2%})")
        logger.info(f"- Tập kiểm tra (Test set): {len(test_df)} dòng ({len(test_df)/len(df):.2%})")

        # Tạo thư mục đầu ra nếu chưa tồn tại
        os.makedirs(os.path.dirname(output_train_csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_test_csv_path), exist_ok=True)

        # Lưu tập huấn luyện vào file CSV
        train_df.to_csv(output_train_csv_path, index=False, encoding='utf-8')
        logger.info(f"Đã lưu tập huấn luyện vào: {output_train_csv_path}")

        # Lưu tập kiểm tra vào file CSV
        test_df.to_csv(output_test_csv_path, index=False, encoding='utf-8')
        logger.info(f"Đã lưu tập kiểm tra vào: {output_test_csv_path}")

    except pd.errors.EmptyDataError:
        logger.error(f"Lỗi: File CSV '{input_csv_path}' trống.")
    except pd.errors.ParserError as e:
        logger.error(f"Lỗi khi phân tích file CSV '{input_csv_path}': {e}")
    except Exception as e:
        logger.error(f"Đã xảy ra lỗi không mong muốn: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Chia file CSV Q&A thành tập huấn luyện và tập kiểm tra.")
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Đường dẫn đến file CSV chứa dữ liệu câu hỏi và câu trả lời."
    )
    parser.add_argument(
        "--output-train-csv",
        type=str,
        default="data/train_qa.csv",
        help="Đường dẫn để lưu file CSV của tập huấn luyện (mặc định: data/train_qa.csv)."
    )
    parser.add_argument(
        "--output-test-csv",
        type=str,
        default="data/test_qa.csv",
        help="Đường dẫn để lưu file CSV của tập kiểm tra (mặc định: data/test_qa.csv)."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Tỷ lệ của tập kiểm tra (ví dụ: 0.3 cho 30%%, mặc định: 0.3)."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Hạt giống ngẫu nhiên để đảm bảo khả năng tái tạo kết quả (mặc định: 42)."
    )

    args = parser.parse_args()

    # Gọi hàm chia dữ liệu
    split_qa_csv(
        input_csv_path=args.input_csv,
        output_train_csv_path=args.output_train_csv,
        output_test_csv_path=args.output_test_csv,
        test_size=args.test_size,
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()
