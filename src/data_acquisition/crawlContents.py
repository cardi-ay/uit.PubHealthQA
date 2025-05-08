import time
import pandas as pd
import json # Thêm thư viện json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# --- Cấu hình ---
INPUT_CSV_FILE = "data_vbpl_boyte_complex_pagination_v2.csv" # File CSV đầu vào
OUTPUT_JSON_FILE = "data_vbpl_boyte_full_details.json"     # File JSON đầu ra <<< THAY ĐỔI
WAIT_TIMEOUT = 5
SHORT_PAUSE = 0.5
CONTENT_WAIT_TIMEOUT = 7

# --- Khởi tạo WebDriver ---
print("Đang khởi tạo trình duyệt Chrome...")
options = webdriver.ChromeOptions()
# options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("start-maximized")
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = None
try:
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    print("Đã khởi tạo trình duyệt.")
except Exception as e:
    print(f"Lỗi khi khởi tạo WebDriver: {e}")
    exit()

# --- Đọc file CSV đầu vào ---
print(f"Đang đọc file input: {INPUT_CSV_FILE}")
try:
    df_input = pd.read_csv(INPUT_CSV_FILE)
    if 'DuongLink' not in df_input.columns:
        raise ValueError("File CSV input thiếu cột 'DuongLink'")
    print(f"Đọc thành công {len(df_input)} dòng từ CSV.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {INPUT_CSV_FILE}")
    if driver: driver.quit()
    exit()
except Exception as e_csv:
    print(f"Lỗi khi đọc file CSV: {e_csv}")
    if driver: driver.quit()
    exit()

# --- Chuẩn bị lưu trữ kết quả ---
results_data = [] # List để chứa các dictionary kết quả
errors_encountered = 0

# --- Lặp qua từng dòng trong DataFrame (từng link) ---
print("\n--- Bắt đầu xử lý từng link ---")
for index, row in df_input.iterrows():
    document_url = row.get('DuongLink')
    print(f"\nĐang xử lý dòng {index + 1}/{len(df_input)}: {document_url}")

    # Tạo dictionary để lưu kết quả cho dòng này, copy dữ liệu cũ
    row_data = row.to_dict()
    # Khởi tạo các trường mới với giá trị None (sẽ thành null trong JSON) <<< THAY ĐỔI
    row_data['NoiDung'] = None
    row_data['LinhVuc'] = None
    row_data['LoaiVanBan_ThuocTinh'] = None

    if not document_url or not isinstance(document_url, str) or not document_url.startswith('http'):
        print("  -> Lỗi: URL không hợp lệ hoặc bị thiếu. Bỏ qua.")
        errors_encountered += 1
        results_data.append(row_data)
        continue

    content_extracted = False
    properties_extracted = False
    properties_page_accessed = False

    try:
        # 1. Truy cập link văn bản và lấy nội dung
        print(f"  Truy cập trang văn bản: {document_url}")
        driver.get(document_url)
        content_div_locator = (By.ID, "toanvancontent")
        try:
            content_div = WebDriverWait(driver, CONTENT_WAIT_TIMEOUT).until(
                EC.visibility_of_element_located(content_div_locator)
            )
            full_content_text = content_div.text
            row_data['NoiDung'] = full_content_text # Cập nhật nội dung
            content_extracted = True
            print(f"    -> Lấy nội dung thành công ({len(full_content_text)} ký tự).")
        except TimeoutException:
            print(f"    -> Lỗi: Timeout khi chờ nội dung (toanvancontent) của link {document_url}. Nội dung sẽ là null.")
            # row_data['NoiDung'] đã là None
        except Exception as e_content_inner:
             print(f"    -> Lỗi: Lỗi không xác định khi lấy nội dung (toanvancontent) của link {document_url}: {e_content_inner}. Nội dung sẽ là null.")
             # row_data['NoiDung'] đã là None


        # 2. Tìm và click link "Thuộc tính"
        print("  Tìm và click link 'Thuộc tính'...")
        try:
            properties_link_locator = (By.XPATH, "//a[.//b[contains(@class, 'properties') and normalize-space(.)='Thuộc tính']]")
            properties_link = WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.element_to_be_clickable(properties_link_locator)
            )
            properties_url = properties_link.get_attribute('href')
            print(f"    -> Tìm thấy link Thuộc tính: {properties_url}")
            driver.execute_script("arguments[0].click();", properties_link)
            properties_page_accessed = True # Đánh dấu đã truy cập trang thuộc tính

            # 3. Chờ trang Thuộc tính tải và lấy thông tin
            print("  Chờ trang Thuộc tính tải...")
            linh_vuc_label_locator = (By.XPATH, "//td[contains(@class,'label') and normalize-space(.)='Lĩnh vực']")
            WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.visibility_of_element_located(linh_vuc_label_locator)
            )
            print("    -> Trang Thuộc tính đã tải.")

            # 3.1 Lấy "Lĩnh vực"
            try:
                linh_vuc_label_td = driver.find_element(*linh_vuc_label_locator)
                linh_vuc_value_td = linh_vuc_label_td.find_element(By.XPATH, "./following-sibling::td")
                row_data['LinhVuc'] = linh_vuc_value_td.text.strip()
                print(f"      -> Lấy Lĩnh vực: {row_data['LinhVuc']}")
                properties_extracted = True
            except NoSuchElementException:
                print("      -> Không tìm thấy giá trị Lĩnh vực. Giá trị sẽ là null.")
                row_data['LinhVuc'] = None # Đảm bảo là None

            # 3.2 Lấy "Loại văn bản" (từ trang Thuộc tính)
            try:
                loai_vb_label_locator = (By.XPATH, "//td[contains(@class,'label') and normalize-space(.)='Loại văn bản']")
                loai_vb_label_td = driver.find_element(*loai_vb_label_locator)
                loai_vb_value_td = loai_vb_label_td.find_element(By.XPATH, "./following-sibling::td")
                row_data['LoaiVanBan_ThuocTinh'] = loai_vb_value_td.text.strip()
                print(f"      -> Lấy Loại văn bản (Thuộc tính): {row_data['LoaiVanBan_ThuocTinh']}")
                properties_extracted = True
            except NoSuchElementException:
                print("      -> Không tìm thấy giá trị Loại văn bản (Thuộc tính). Giá trị sẽ là null.")
                row_data['LoaiVanBan_ThuocTinh'] = None # Đảm bảo là None

        except (TimeoutException, NoSuchElementException) as e_prop_link_or_wait:
             print(f"  -> Lỗi: Không tìm thấy link 'Thuộc tính' hoặc trang thuộc tính tải thất bại: {e_prop_link_or_wait}. Các thuộc tính sẽ là null.")
             # Các giá trị thuộc tính đã được khởi tạo là None

    except TimeoutException as e_timeout:
        print(f"  -> Lỗi: Timeout tổng thể khi xử lý link {document_url}. Chi tiết: {e_timeout}")
        errors_encountered += 1
    except Exception as e_general:
        print(f"  -> Lỗi: Lỗi không xác định khi xử lý link {document_url}: {e_general}")
        errors_encountered += 1

    # Thêm dữ liệu (đã cập nhật hoặc còn giá trị None nếu lỗi) vào list kết quả
    results_data.append(row_data)
    time.sleep(SHORT_PAUSE)

# --- Dọn dẹp ---
print("\nĐang đóng trình duyệt...")
if driver:
    driver.quit()

# --- Lưu kết quả cuối cùng ra file JSON --- <<< THAY ĐỔI
print(f"\n--- Hoàn tất xử lý {len(df_input)} link. Gặp lỗi với {errors_encountered} link ---")
if results_data:
    print(f"Tổng số bản ghi thu được: {len(results_data)}")
    # In thử 5 bản ghi đầu (dạng dictionary)
    print("\n5 bản ghi đầu tiên (dạng dict):")
    for i, record in enumerate(results_data[:5]):
        print(f"Record {i+1}:")
        # In các key chính, giới hạn độ dài nội dung
        print(f"  TenVanBan: {record.get('TenVanBan', 'N/A')}")
        print(f"  DuongLink: {record.get('DuongLink', 'N/A')}")
        print(f"  NgayHieuLuc: {record.get('NgayHieuLuc', 'N/A')}")
        print(f"  LinhVuc: {record.get('LinhVuc')}") # Sẽ là None nếu không tìm thấy
        print(f"  LoaiVanBan_ThuocTinh: {record.get('LoaiVanBan_ThuocTinh')}") # Sẽ là None nếu không tìm thấy
        noi_dung_preview = str(record.get('NoiDung'))[:100] + '...' if record.get('NoiDung') else 'None'
        print(f"  NoiDung (preview): {noi_dung_preview}")

    # Lưu vào file JSON
    try:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=4) # indent=4 để dễ đọc
        print(f"\nĐã lưu dữ liệu đầy đủ vào file JSON: {OUTPUT_JSON_FILE}")
    except Exception as e_save:
        print(f"\nLỗi khi lưu file JSON: {e_save}")
else:
    print("Không có dữ liệu nào được xử lý hoặc lưu trữ.")

print("\nKết thúc chương trình.")