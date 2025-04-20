import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://vbpl.vn/boyte/Pages/Home.aspx"
WAIT_TIMEOUT = 30
SHORT_PAUSE = 0.5

KEYWORD = ""
LOAI_VAN_BAN_VALUES = ["17", "18", "22"]
CO_QUAN_BH_VALUES = ["50"]
TINH_TRANG_HL_VALUES = ["4", "2"]
SAP_XEP_THEO_VALUE = "VBPQNgayBanHanh"
THU_TU_SAP_XEP_VALUE = "False"

# --- Hàm trợ giúp (Giữ nguyên) ---
def handle_multi_select_exact(driver, activator_id, checkbox_name, values_to_select):
    retries = 2
    for attempt in range(retries):
        try:
            activator = WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.element_to_be_clickable((By.ID, activator_id))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", activator)
            time.sleep(0.2)
            driver.execute_script("arguments[0].click();", activator)
            time.sleep(SHORT_PAUSE)

            options_div_xpath = f"//a[@id='{activator_id}']/following-sibling::div[contains(@class, 'multiSelectOptions')]"
            options_div = WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.visibility_of_element_located((By.XPATH, options_div_xpath))
            )

            all_checkboxes = options_div.find_elements(By.XPATH, f".//input[@name='{checkbox_name}[]']")
            if not all_checkboxes:
                 # print(f"    Cảnh báo: Không tìm thấy checkbox nào cho {checkbox_name}")
                 driver.execute_script("arguments[0].click();", activator)
                 time.sleep(SHORT_PAUSE)
                 return True

            # print(f"    Tìm thấy {len(all_checkboxes)} checkboxes cho {checkbox_name}. Đang kiểm tra/thiết lập...")

            for cb in all_checkboxes:
                for cb_retry in range(retries):
                    try:
                        cb_value = cb.get_attribute('value')
                        is_selected = cb.is_selected()
                        should_be_selected = cb_value in values_to_select

                        if should_be_selected and not is_selected:
                            # print(f"      -> Chọn {checkbox_name} value: {cb_value}")
                            driver.execute_script("arguments[0].click();", cb)
                            time.sleep(0.1)
                        elif not should_be_selected and is_selected:
                            # print(f"      -> Bỏ chọn {checkbox_name} value: {cb_value}")
                            driver.execute_script("arguments[0].click();", cb)
                            time.sleep(0.1)
                        break
                    except StaleElementReferenceException:
                         # print(f"      Gặp StaleElementReferenceException cho checkbox {cb_value}, thử lại...")
                         if cb_retry == retries - 1:
                             raise
                         time.sleep(0.5)
                         options_div = WebDriverWait(driver, WAIT_TIMEOUT).until(
                            EC.visibility_of_element_located((By.XPATH, options_div_xpath))
                         )
                         all_checkboxes = options_div.find_elements(By.XPATH, f".//input[@name='{checkbox_name}[]']")
                         # print("      Thử lại toàn bộ hàm handle_multi_select_exact...")
                         raise StaleElementReferenceException("Thử lại từ đầu hàm")


            driver.execute_script("arguments[0].click();", activator)
            time.sleep(SHORT_PAUSE)
            # print(f"  Đã xử lý xong multi-select (exact): {activator_id}")
            return True

        except StaleElementReferenceException as e_stale:
             print(f"Gặp StaleElementReferenceException khi xử lý {activator_id} (lần thử {attempt + 1}), thử lại...")
             if attempt == retries - 1:
                 print(f"Lỗi StaleElementReferenceException không thể khắc phục cho {activator_id}")
                 return False
             time.sleep(1)
        except TimeoutException:
            print(f"Lỗi: Timeout khi xử lý multi-select '{activator_id}'")
            return False
        except Exception as e:
            print(f"Lỗi khi xử lý multi-select (exact) '{activator_id}': {e}")
            return False
    return False

def wait_for_page_load(driver, previous_ul_element_ref):
    print("Chờ trang mới tải...")
    try:
        if previous_ul_element_ref:
            WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.staleness_of(previous_ul_element_ref)
            )
            # print("  -> Ul cũ đã stale.")

        results_container_locator = (By.ID, "grid_vanban")
        ul_xpath = ".//div[contains(@class,'box-container')]/div[contains(@class,'box-tab')]/div[@class='content']/ul[contains(@class,'listLaw')]"
        WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located(results_container_locator)
        )
        WebDriverWait(driver, WAIT_TIMEOUT).until(
             EC.visibility_of_element_located((By.XPATH, ul_xpath))
        )
        print("Trang mới đã tải và ul mới hiển thị.")
        return True
    except TimeoutException:
        print("Lỗi: Timeout khi chờ trang mới tải xong.")
        return False
    except Exception as e:
        print(f"Lỗi không xác định khi chờ trang mới: {e}")
        return False


def scrape_current_page_data(driver, page_num, data_list):
    print(f"--- Bắt đầu scrape dữ liệu trang {page_num} ---")
    items_scraped_on_page = 0
    try:
        results_container_locator = (By.ID, "grid_vanban")
        results_container = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located(results_container_locator)
        )
        ul_xpath = ".//div[contains(@class,'box-container')]/div[contains(@class,'box-tab')]/div[@class='content']/ul[contains(@class,'listLaw')]"
        results_ul = WebDriverWait(results_container, WAIT_TIMEOUT).until(
             EC.visibility_of_element_located((By.XPATH, ul_xpath))
        )

        time.sleep(0.8)
        document_items = results_ul.find_elements(By.XPATH, "./li")
        print(f"Tìm thấy {len(document_items)} văn bản trên trang {page_num}.")

        if not document_items:
             print("Không có văn bản nào để scrape trên trang này.")
             return None

        for item_index, item in enumerate(document_items):
            item_data = {}
            try:
                title_link_element = WebDriverWait(item, 5).until(
                     EC.visibility_of_element_located((By.CSS_SELECTOR, "p.title > a"))
                 )
                item_data['TenVanBan'] = title_link_element.text.strip()
                item_data['DuongLink'] = title_link_element.get_attribute('href')
                item_data['NgayHieuLuc'] = "Không tìm thấy"
                try:
                    date_p_element = item.find_element(By.XPATH, ".//div[@class='right']/p[contains(., 'Hiệu lực:')]")
                    full_text = date_p_element.text
                    if ":" in full_text:
                        item_data['NgayHieuLuc'] = full_text.split(":")[-1].strip()
                except NoSuchElementException:
                    pass

                item_data['Trang'] = page_num
                data_list.append(item_data)
                items_scraped_on_page += 1
                # print(f"  - Đã lấy: {item_data['TenVanBan']} | Hiệu lực: {item_data['NgayHieuLuc']}")

            except (NoSuchElementException, TimeoutException, StaleElementReferenceException) as e_item:
                print(f"*** Lỗi khi xử lý mục văn bản {item_index+1} trang {page_num}: {type(e_item).__name__}. Bỏ qua mục này. ***")
                continue

        print(f"--- Hoàn tất scrape trang {page_num}, lấy được {items_scraped_on_page} mục ---")
        return results_ul

    except TimeoutException:
        print(f"Lỗi Timeout khi tìm danh sách kết quả trên trang {page_num}.")
        return None
    except Exception as e:
         print(f"Lỗi không xác định khi scrape trang {page_num}: {e}")
         return None


# --- Khởi tạo WebDriver ---
print("Đang khởi tạo trình duyệt Chrome...")
options = webdriver.ChromeOptions()
# options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("start-maximized")
options.add_experimental_option('excludeSwitches', ['enable-logging'])

try:
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    print("Đã khởi tạo trình duyệt.")
except Exception as e:
    print(f"Lỗi khi khởi tạo WebDriver: {e}")
    exit()

# --- Mở trang web và Tìm kiếm ---
try:
    print(f"Đang truy cập URL: {URL}")
    driver.get(URL)
    print("\n--- Bắt đầu thiết lập tìm kiếm nâng cao ---")
    print("  Click vào 'Tìm kiếm nâng cao'...")
    advanced_search_button = WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.element_to_be_clickable((By.ID, "timkiemnangcao"))
    )
    driver.execute_script("arguments[0].click();", advanced_search_button)
    WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.visibility_of_element_located((By.ID, "LoaiVanBanDiv"))
    )
    print("  Đã mở form tìm kiếm nâng cao.")
    time.sleep(SHORT_PAUSE)
    print("  Chọn tìm trong (đảm bảo là Trích yếu)")
    WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.element_to_be_clickable((By.ID, "rdtrichyeu"))
    ).click()
    print("  Chọn Loại văn bản...")
    if not handle_multi_select_exact(driver, "LoaiVanBan", "LoaiVanBan", LOAI_VAN_BAN_VALUES):
        raise Exception("Không thể xử lý Loại văn bản")
    print("  Chọn Cơ quan ban hành...")
    if not handle_multi_select_exact(driver, "CoQuanBanHanh", "CoQuanBanHanh", CO_QUAN_BH_VALUES):
         raise Exception("Không thể xử lý Cơ quan ban hành")
    print("  Chọn Tình trạng hiệu lực...")
    if not handle_multi_select_exact(driver, "TrangThaiHieuLuc", "TrangThaiHieuLuc", TINH_TRANG_HL_VALUES):
        raise Exception("Không thể xử lý Tình trạng hiệu lực")
    print("  Click nút 'Tìm kiếm'...")
    search_submit_button = WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.element_to_be_clickable((By.ID, "searchSubmit"))
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", search_submit_button)
    time.sleep(0.2)
    driver.execute_script("arguments[0].click();", search_submit_button)

    print("  Chờ kết quả tìm kiếm tải...")
    results_container_locator = (By.ID, "grid_vanban")
    results_count_locator = (By.XPATH, "//div[@id='grid_vanban']//a[contains(@class,'selected')]/span[contains(., 'Kết quả tìm kiếm: Tìm thấy')]")
    first_result_item_locator = (By.CSS_SELECTOR, "#grid_vanban ul.listLaw > li:first-child")

    WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_element_located(results_container_locator))
    WebDriverWait(driver, WAIT_TIMEOUT).until(EC.visibility_of_element_located(results_count_locator))
    WebDriverWait(driver, WAIT_TIMEOUT).until(EC.visibility_of_element_located(first_result_item_locator))
    print("  Kết quả tìm kiếm đã tải xong và ổn định.")
    print("--- Kết thúc thiết lập tìm kiếm ---")

except Exception as e_setup:
    print(f"\n!!! Lỗi trong quá trình truy cập hoặc thiết lập tìm kiếm: {e_setup}")
    driver.save_screenshot("error_screenshot_setup.png")
    print("Đã lưu ảnh chụp màn hình lỗi: error_screenshot_setup.png")
    driver.quit()
    exit()

# --- Logic Scrape và Phân trang Mới (Theo luồng: Sau -> Trước -> Cuối -> Scrape -> Trước -> Scrape -> Trước -> Scrape) ---
scraped_data = []
current_page_num = 1
scraping_mode = "forward" # forward, go_to_last, last_page_reached, backward
pages_left_to_scrape_backwards = 2
last_ul_element_ref = None

while True:
    print(f"\n>>> Chế độ: {scraping_mode}, Trang logic: {current_page_num}, Lùi còn: {pages_left_to_scrape_backwards}")
    page_start_time = time.time()
    action_taken = 'none'
    exception_occurred = False

    try:
        # --- Xử lý dựa trên chế độ ---
        if scraping_mode == "forward":
            # 1. Scrape trang hiện tại
            print(f"Thực hiện: Scrape trang {current_page_num} (forward)")
            last_ul_element_ref = scrape_current_page_data(driver, current_page_num, scraped_data)
            if last_ul_element_ref is None:
                 raise Exception(f"Scrape thất bại ở trang {current_page_num} chế độ forward")

            # 2. Thử tìm và click nút "Sau"
            print("Đang tìm nút 'Sau'...")
            try:
                next_button_locator = (By.XPATH, "//div[@class='paging']//a[text()='Sau']")
                next_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable(next_button_locator))
                print("  -> Tìm thấy 'Sau'. Click và tiếp tục.")
                driver.execute_script("arguments[0].click();", next_button)
                if not wait_for_page_load(driver, last_ul_element_ref):
                    raise Exception("Chờ trang sau thất bại")
                current_page_num += 1
                action_taken = 'next'
            except (TimeoutException, NoSuchElementException):
                print("  -> Không tìm thấy 'Sau'. Chuyển sang chế độ 'go_to_last'.")
                scraping_mode = "go_to_last"
                action_taken = 'none' # Sẽ xử lý ở lần lặp sau

        elif scraping_mode == "go_to_last":
            print("Thực hiện: Click 'Trước' (1 lần, không scrape)")
            try:
                prev_button_locator = (By.XPATH, "//div[@class='paging']//a[text()='Trước']") # Đảm bảo locator này đúng
                prev_button = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.element_to_be_clickable(prev_button_locator))
                driver.execute_script("arguments[0].click();", prev_button)
                if not wait_for_page_load(driver, last_ul_element_ref):
                    raise Exception("Chờ trang trước thất bại (go_to_last)")
                # Cập nhật tham chiếu ul sau khi tải trang
                ul_xpath = ".//div[contains(@class,'box-container')]/div[contains(@class,'box-tab')]/div[@class='content']/ul[contains(@class,'listLaw')]"
                last_ul_element_ref = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.visibility_of_element_located((By.XPATH, ul_xpath)))
                print("  -> Đã click 'Trước'.")
                time.sleep(SHORT_PAUSE)
            except Exception as e_prev:
                 print(f"Lỗi khi click 'Trước' trong go_to_last: {e_prev}")
                 raise

            print("Thực hiện: Click 'Cuối »'")
            try:
                last_button_locator = (By.XPATH, "//div[@class='paging']//a[text()='Cuối »']")
                last_button = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.element_to_be_clickable(last_button_locator))
                driver.execute_script("arguments[0].click();", last_button)
                if not wait_for_page_load(driver, last_ul_element_ref):
                    raise Exception("Chờ trang cuối thất bại (go_to_last)")
                print("  -> Đã click 'Cuối »'.")
                scraping_mode = "last_page_reached" # Sẵn sàng scrape trang cuối ở lần lặp sau
                action_taken = 'last'
                # Xác định số trang cuối
                try:
                    page_links = driver.find_elements(By.XPATH, "//div[@class='paging']/a[number(text())]")
                    last_page_num_detected = max([int(a.text) for a in page_links if a.text.isdigit()])
                    current_page_num = last_page_num_detected
                    print(f"  -> Đã đến trang cuối (Trang {current_page_num}).")
                except Exception as e_get_last_num:
                    print(f"  -> Cảnh báo: Không thể tự động xác định số trang cuối ({e_get_last_num}).")
            except Exception as e_last:
                 print(f"Lỗi khi click 'Cuối »' trong go_to_last: {e_last}")
                 raise

        elif scraping_mode == "last_page_reached":
            # 1. Scrape trang cuối
            print(f"Thực hiện: Scrape trang cuối (Trang {current_page_num})")
            last_ul_element_ref = scrape_current_page_data(driver, current_page_num, scraped_data)
            if last_ul_element_ref is None:
                raise Exception("Scrape thất bại ở trang cuối")

            # 2. Chuẩn bị cho lần click 'Trước' đầu tiên
            print("  -> Đã scrape trang cuối. Chuyển sang chế độ 'backward'.")
            scraping_mode = "backward"
            action_taken = 'none'

        elif scraping_mode == "backward":
            # Logic đi lùi: Click Trước -> Scrape -> Click Trước -> Scrape -> Dừng
            if pages_left_to_scrape_backwards > 0:
                print(f"Thực hiện: Click 'Trước' (còn {pages_left_to_scrape_backwards} lần scrape lùi)")
                try:
                    prev_button_locator = (By.XPATH, "//div[@class='paging']//a[text()='Trước']")
                    prev_button = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.element_to_be_clickable(prev_button_locator))
                    driver.execute_script("arguments[0].click();", prev_button)
                    if not wait_for_page_load(driver, last_ul_element_ref):
                        raise Exception("Chờ trang trước (backward) thất bại")

                    current_page_num -= 1 # Cập nhật số trang logic
                    action_taken = 'prev'

                    # Scrape ngay trang vừa lùi về
                    print(f"Thực hiện: Scrape trang {current_page_num} (backward)")
                    last_ul_element_ref = scrape_current_page_data(driver, current_page_num, scraped_data)
                    if last_ul_element_ref is None:
                         raise Exception(f"Scrape thất bại ở trang {current_page_num} (backward)")

                    pages_left_to_scrape_backwards -= 1 # Giảm số lần cần lùi

                    # Nếu đã scrape đủ số trang lùi, dừng lại
                    if pages_left_to_scrape_backwards == 0:
                        print("Đã scrape đủ số trang backward. Kết thúc.")
                        action_taken = 'break'
                        break

                except Exception as e_prev_back:
                    print(f"Lỗi khi click 'Trước' hoặc scrape trong chế độ backward: {e_prev_back}. Dừng lại.")
                    action_taken = 'break'
                    break
            else:
                # Đã scrape đủ, dừng vòng lặp
                print("Đã scrape đủ số trang backward (trạng thái cuối). Kết thúc.")
                action_taken = 'break'
                break

    except Exception as e_main_loop:
        exception_occurred = True
        print(f"\n!!! Lỗi nghiêm trọng trong vòng lặp chính: Chế độ={scraping_mode}, Trang logic={current_page_num}: {e_main_loop}")
        try:
            driver.save_screenshot(f"error_screenshot_loop_page{current_page_num}.png")
            print(f"Đã lưu ảnh chụp màn hình lỗi: error_screenshot_loop_page{current_page_num}.png")
        except Exception as e_ss:
            print(f"Không thể lưu ảnh chụp màn hình: {e_ss}")
        break
    finally:
        page_end_time = time.time()
        print(f"--- Hoàn tất vòng lặp: Chế độ={scraping_mode}, Trang logic={current_page_num}, Hành động={action_taken}, Thời gian={page_end_time - page_start_time:.2f} giây ---")
        if action_taken == 'break':
             print("Thoát vòng lặp chính.")


# --- Đóng trình duyệt ---
print("\nĐang đóng trình duyệt...")
driver.quit()

# --- Xử lý và Xuất dữ liệu ---
print(f"\n--- Hoàn tất thu thập dữ liệu. Tổng cộng: {len(scraped_data)} mục ---")

if scraped_data:
    df = pd.DataFrame(scraped_data)
    # Loại bỏ các dòng trùng lặp nếu có (do scrape tiến rồi lùi)
    # Giữ lại bản ghi cuối cùng cho mỗi DuongLink (thường là từ lần scrape cuối cùng)
    print(f"Số mục trước khi loại bỏ trùng lặp: {len(df)}")
    df_unique = df.drop_duplicates(subset=['DuongLink'], keep='last')
    print(f"Số mục sau khi loại bỏ trùng lặp (giữ bản ghi cuối): {len(df_unique)}")

    print("\nDữ liệu đã thu thập (5 dòng đầu - duy nhất):")
    print(df_unique.head())
    print("\nDữ liệu đã thu thập (5 dòng cuối - duy nhất):")
    print(df_unique.tail())
    try:
        csv_filename = "data_vbpl_boyte_complex_pagination_v2.csv" # Đổi tên file
        df_unique.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"\nĐã lưu dữ liệu duy nhất vào file: {csv_filename}")
    except Exception as e_save:
        print(f"\nLỗi khi lưu file CSV: {e_save}")
else:
    print("Không có dữ liệu nào được thu thập.")

print("\nKết thúc chương trình.")