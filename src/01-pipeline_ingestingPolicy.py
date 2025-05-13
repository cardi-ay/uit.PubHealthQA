import time
import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException,
    ElementClickInterceptedException, StaleElementReferenceException,
    WebDriverException # Catch broader driver issues
)
from webdriver_manager.chrome import ChromeDriverManager

# --- Cấu hình ---
URL = "https://vbpl.vn/boyte/Pages/Home.aspx"
WAIT_TIMEOUT = 30
SHORT_PAUSE = 0.5
CONTENT_WAIT_TIMEOUT = 7 # Timeout for waiting for full content element

# Search parameters
KEYWORD = ""
LOAI_VAN_BAN_VALUES = ["17", "18", "22"]
CO_QUAN_BH_VALUES = ["50"]
TINH_TRANG_HL_VALUES = ["4", "2"]
# SAP_XEP_THEO_VALUE = "VBPQNgayBanHanh" # Not used in logic
# THU_TU_SAP_XEP_VALUE = "False" # Not used in logic

# Output file
OUTPUT_JSON_FILE = Path("data_vbpl_boyte_full_details_merged_modular.json")

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def handle_multi_select_exact(driver: webdriver.Chrome, activator_id: str, checkbox_name: str, values_to_select: List[str]) -> bool:
    """
    Xử lý multi-select dropdown trên trang tìm kiếm.
    Clicks activator, selects/deselects checkboxes based on exact values_to_select.
    Returns True on success, False on failure.
    """
    retries = 3
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
                logging.warning(f"handle_multi_select_exact: Không tìm thấy checkbox nào cho {checkbox_name} (Activator: {activator_id}).")
                # Try closing the dropdown if it opened empty
                try:
                    driver.execute_script("arguments[0].click();", activator)
                    time.sleep(SHORT_PAUSE)
                except Exception:
                    pass # Ignore if closing fails
                return True # Consider it successful if there are no options to select

            # Process checkboxes
            for cb in all_checkboxes:
                 cb_value = cb.get_attribute('value')
                 is_selected = cb.is_selected()
                 should_be_selected = cb_value in values_to_select

                 if should_be_selected and not is_selected:
                     # logging.debug(f"  -> Chọn {checkbox_name} value: {cb_value}")
                     try:
                         driver.execute_script("arguments[0].click();", cb)
                         time.sleep(0.1)
                     except StaleElementReferenceException:
                          logging.warning(f"handle_multi_select_exact: StaleElementReferenceException clicking checkbox {cb_value}. Retrying function attempt {attempt+1}.")
                          raise # Re-raise to trigger function retry

                 elif not should_be_selected and is_selected:
                     # logging.debug(f"  -> Bỏ chọn {checkbox_name} value: {cb_value}")
                     try:
                         driver.execute_script("arguments[0].click();", cb)
                         time.sleep(0.1)
                     except StaleElementReferenceException:
                          logging.warning(f"handle_multi_select_exact: StaleElementReferenceException clicking checkbox {cb_value}. Retrying function attempt {attempt+1}.")
                          raise # Re-raise to trigger function retry


            # Close the dropdown
            driver.execute_script("arguments[0].click();", activator)
            time.sleep(SHORT_PAUSE)
            logging.info(f"handle_multi_select_exact: Đã xử lý xong multi-select (exact): {activator_id}")
            return True

        except (StaleElementReferenceException, TimeoutException) as e:
            logging.warning(f"handle_multi_select_exact: Gặp {type(e).__name__} khi xử lý '{activator_id}' (lần thử {attempt + 1}). Thử lại...")
            if attempt == retries - 1:
                logging.error(f"handle_multi_select_exact: Lỗi {type(e).__name__} không thể khắc phục cho '{activator_id}'.")
                return False
            time.sleep(1) # Wait before retrying the whole function
        except Exception as e:
            logging.error(f"handle_multi_select_exact: Lỗi không xác định khi xử lý multi-select '{activator_id}': {e}")
            return False # Fail on unexpected errors
    return False # Return False if all retries fail


def wait_for_list_page_load(driver: webdriver.Chrome, previous_ul_element_ref: Optional[webdriver.remote.webelement.WebElement]) -> Tuple[bool, Optional[webdriver.remote.webelement.WebElement]]:
    """
    Chờ cho trang kết quả tìm kiếm mới tải sau khi phân trang.
    Waits for old list to go stale and new list to be visible.
    Returns (success_status, new_ul_element_reference).
    """
    # logging.info("wait_for_list_page_load: Chờ trang mới tải...")
    try:
        # Wait for the old element to become stale if a reference is provided
        if previous_ul_element_ref:
            WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.staleness_of(previous_ul_element_ref)
            )
            # logging.debug("wait_for_list_page_load: Ul cũ đã stale.")

        # Wait for the new results container and the new UL list to appear/be visible
        results_container_locator = (By.ID, "grid_vanban")
        ul_xpath = ".//div[contains(@class,'box-container')]/div[contains(@class,'box-tab')]/div[@class='content']/ul[contains(@class,'listLaw')]"
        WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located(results_container_locator)
        )
        new_ul_element = WebDriverWait(driver, WAIT_TIMEOUT).until(
             EC.visibility_of_element_located((By.XPATH, ul_xpath))
        )
        logging.info("wait_for_list_page_load: Trang mới đã tải và ul mới hiển thị.")
        return True, new_ul_element # Return success and the new ul element reference
    except TimeoutException:
        logging.error("wait_for_list_page_load: Lỗi: Timeout khi chờ trang mới tải xong.")
        return False, None
    except Exception as e:
        logging.error(f"wait_for_list_page_load: Lỗi không xác định khi chờ trang mới: {e}")
        return False, None


def scrape_list_page_data(driver: webdriver.Chrome, page_num: int) -> Tuple[List[Dict[str, Any]], Optional[webdriver.remote.webelement.WebElement]]:
    """
    Scrape basic information (Title, Link, Effective Date) from the current results page.
    Returns a list of dictionaries for the current page and the UL element reference.
    Returns ([], None) on failure.
    """
    logging.info(f"scrape_list_page_data: --- Bắt đầu scrape dữ liệu trang {page_num} ---")
    page_data = []
    results_ul_ref = None

    try:
        results_container_locator = (By.ID, "grid_vanban")
        results_container = WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.presence_of_element_located(results_container_locator)
        )
        ul_xpath = ".//div[contains(@class,'box-container')]/div[contains(@class,'box-tab')]/div[@class='content']/ul[contains(@class,'listLaw')]"
        results_ul = WebDriverWait(results_container, WAIT_TIMEOUT).until(
             EC.visibility_of_element_located((By.XPATH, ul_xpath))
        )
        results_ul_ref = results_ul # Store the reference

        time.sleep(0.8) # Small pause after visibility might help with element stability
        document_items = results_ul.find_elements(By.XPATH, "./li")
        logging.info(f"scrape_list_page_data: Tìm thấy {len(document_items)} văn bản trên trang {page_num}.")

        if not document_items:
            logging.warning("scrape_list_page_data: Không có văn bản nào để scrape trên trang này.")
            return page_data, results_ul_ref # Return empty list but valid UL ref

        for item_index, item in enumerate(document_items):
            item_data = {}
            try:
                # Re-find/verify item visibility to reduce StaleElementReferenceException possibility
                item = WebDriverWait(driver, 5).until(EC.visibility_of(item))

                title_link_element = item.find_element(By.CSS_SELECTOR, "p.title > a")
                item_data['TenVanBan'] = title_link_element.text.strip()
                item_data['DuongLink'] = title_link_element.get_attribute('href')
                item_data['NgayHieuLuc'] = "Không tìm thấy" # Default value

                try:
                    # Find the 'Hiệu lực' paragraph within this specific item
                    date_p_element = item.find_element(By.XPATH, ".//div[@class='right']/p[contains(., 'Hiệu lực:')]")
                    full_text = date_p_element.text
                    if ":" in full_text:
                        item_data['NgayHieuLuc'] = full_text.split(":")[-1].strip()
                except NoSuchElementException:
                    pass # Keep default value if not found

                item_data['Trang'] = page_num # Store the page number where it was found
                # Add placeholders for data to be scraped in the second phase
                item_data['NoiDung'] = None
                item_data['LinhVuc'] = None
                item_data['LoaiVanBan_ThuocTinh'] = None

                page_data.append(item_data)
                # logging.debug(f"  - Đã lấy: {item_data['TenVanBan']} | Hiệu lực: {item_data['NgayHieuLuc']}")

            except (NoSuchElementException, TimeoutException, StaleElementReferenceException) as e_item:
                logging.warning(f"scrape_list_page_data: *** Lỗi khi xử lý mục văn bản {item_index+1} trang {page_num}: {type(e_item).__name__}. Bỏ qua mục này. ***")
                continue # Continue to the next item

        logging.info(f"scrape_list_page_data: --- Hoàn tất scrape trang {page_num}, lấy được {len(page_data)} mục ---")
        return page_data, results_ul_ref # Return collected data and the UL ref

    except TimeoutException:
        logging.error(f"scrape_list_page_data: Lỗi Timeout khi tìm danh sách kết quả trên trang {page_num}.")
        return [], None # Indicate failure to scrape page by returning empty list and None ref
    except Exception as e:
        logging.error(f"scrape_list_page_data: Lỗi không xác định khi scrape trang {page_num}: {e}")
        return [], None # Indicate failure

# --- Hàm tải dữ liệu JSON (Không thay đổi theo yêu cầu) ---
def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """
    Tải dữ liệu từ file JSON.

    Args:
        file_path: Đường dẫn đến file JSON

    Returns:
        Danh sách các văn bản đã tải
    """
    if not file_path.exists():
        logging.error(f"load_json_data: Không tìm thấy file: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            logging.info(f"load_json_data: Đã tải thành công {len(data)} văn bản từ {file_path}")
            return data
        else:
            logging.error(f"load_json_data: Lỗi: File {file_path} không chứa một danh sách JSON.")
            return []
    except json.JSONDecodeError as e:
        logging.error(f"load_json_data: Lỗi: File {file_path} không phải là định dạng JSON hợp lệ. {e}")
        return []
    except Exception as e:
        logging.error(f"load_json_data: Lỗi không xác định khi đọc file {file_path}: {e}")
        return []

# --- Core Scraping Functions (Modularized) ---

def setup_driver() -> Optional[webdriver.Chrome]:
    """Initializes and returns the Chrome WebDriver."""
    logging.info("setup_driver: Đang khởi tạo trình duyệt Chrome...")
    driver = None
    try:
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless") # Uncomment to run without opening browser window
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("start-maximized")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        # Prevent image loading to speed up (optional)
        # prefs = {"profile.managed_default_content_settings.images": 2}
        # options.add_experimental_option("prefs", prefs)


        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        logging.info("setup_driver: Đã khởi tạo trình duyệt.")
        return driver
    except Exception as e:
        logging.critical(f"setup_driver: Lỗi khi khởi tạo WebDriver: {e}")
        return None

def setup_advanced_search(driver: webdriver.Chrome):
    """Navigates to the page and sets up the advanced search filters."""
    logging.info(f"setup_advanced_search: Đang truy cập URL: {URL}")
    driver.get(URL)
    logging.info("setup_advanced_search: --- Bắt đầu thiết lập tìm kiếm nâng cao ---")

    # Click 'Tìm kiếm nâng cao'
    logging.info("setup_advanced_search: Click vào 'Tìm kiếm nâng cao'...")
    advanced_search_button = WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.element_to_be_clickable((By.ID, "timkiemnangcao"))
    )
    driver.execute_script("arguments[0].click();", advanced_search_button)
    WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.visibility_of_element_located((By.ID, "LoaiVanBanDiv")) # Wait for part of the form to appear
    )
    logging.info("setup_advanced_search: Đã mở form tìm kiếm nâng cao.")
    time.sleep(SHORT_PAUSE)

    # Select search scope (Trích yếu)
    logging.info("setup_advanced_search: Chọn tìm trong (đảm bảo là Trích yếu)")
    WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.element_to_be_clickable((By.ID, "rdtrichyeu"))
    ).click()

    # Select multi-select filters
    logging.info("setup_advanced_search: Chọn Loại văn bản...")
    if not handle_multi_select_exact(driver, "LoaiVanBan", "LoaiVanBan", LOAI_VAN_BAN_VALUES):
        raise Exception("setup_advanced_search: Không thể xử lý Loại văn bản filter")

    logging.info("setup_advanced_search: Chọn Cơ quan ban hành...")
    if not handle_multi_select_exact(driver, "CoQuanBanHanh", "CoQuanBanHanh", CO_QUAN_BH_VALUES):
         raise Exception("setup_advanced_search: Không thể xử lý Cơ quan ban hành filter")

    logging.info("setup_advanced_search: Chọn Tình trạng hiệu lực...")
    if not handle_multi_select_exact(driver, "TrangThaiHieuLuc", "TrangThaiHieuLuc", TINH_TRANG_HL_VALUES):
        raise Exception("setup_advanced_search: Không thể xử lý Tình trạng hiệu lực filter")

    # Click search button
    logging.info("setup_advanced_search: Click nút 'Tìm kiếm'...")
    search_submit_button = WebDriverWait(driver, WAIT_TIMEOUT).until(
        EC.element_to_be_clickable((By.ID, "searchSubmit"))
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", search_submit_button)
    time.sleep(0.2)
    driver.execute_script("arguments[0].click();", search_submit_button)

    # Wait for initial search results to load
    logging.info("setup_advanced_search: Chờ kết quả tìm kiếm tải...")
    results_container_locator = (By.ID, "grid_vanban")
    first_result_item_locator = (By.CSS_SELECTOR, "#grid_vanban ul.listLaw > li:first-child")

    WebDriverWait(driver, WAIT_TIMEOUT).until(EC.presence_of_element_located(results_container_locator))
    WebDriverWait(driver, WAIT_TIMEOUT).until(EC.visibility_of_element_located(first_result_item_locator))
    logging.info("setup_advanced_search: Kết quả tìm kiếm đã tải xong và ổn định.")
    logging.info("setup_advanced_search: --- Kết thúc thiết lập tìm kiếm ---")


def scrape_all_list_pages(driver: webdriver.Chrome) -> List[Dict[str, Any]]:
    """
    Manages pagination and scrapes data from all list pages based on the search criteria.
    Returns a list of dictionaries containing basic document info.
    """
    logging.info("\n--- Bắt đầu scrape dữ liệu từ list (Giai đoạn 1) ---")
    scraped_list_data = []
    current_page_num = 1
    scraping_mode = "forward"
    pages_left_to_scrape_backwards = 2
    last_ul_element_ref = None # Keep track of the last UL element reference for waiting

    try:
        # Initial scrape of the first page
        page_data, last_ul_element_ref = scrape_list_page_data(driver, current_page_num)
        scraped_list_data.extend(page_data)

        if last_ul_element_ref is None and not page_data:
             logging.warning("scrape_all_list_pages: Không tìm thấy kết quả nào trên trang đầu tiên hoặc trang đầu tiên scrape thất bại.")
             return scraped_list_data # Return empty list if first page fails

        while True:
            logging.info(f"\n>>> Phân trang: Chế độ={scraping_mode}, Trang logic={current_page_num}, Lùi còn={pages_left_to_scrape_backwards}")
            action_taken = 'none'
            break_loop = False # Flag to break the while loop

            try:
                # --- Pagination Logic ---
                if scraping_mode == "forward":
                    logging.info("Phân trang: Đang tìm nút 'Sau'...")
                    try:
                        next_button_locator = (By.XPATH, "//div[@class='paging']//a[text()='Sau']")
                        next_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable(next_button_locator))
                        logging.info("Phân trang: -> Tìm thấy 'Sau'. Click và tiếp tục forward.")
                        driver.execute_script("arguments[0].click();", next_button)

                        success, new_ul_ref = wait_for_list_page_load(driver, last_ul_element_ref)
                        if not success or new_ul_ref is None:
                            raise Exception("Chờ trang sau thất bại")
                        last_ul_element_ref = new_ul_ref # Update reference

                        current_page_num += 1
                        action_taken = 'next'

                        # Scrape the new page after waiting
                        page_data, current_ul_ref = scrape_list_page_data(driver, current_page_num)
                        if current_ul_ref is None and page_data: # Should not happen if wait_for_page_load succeeded
                             logging.error(f"scrape_all_list_pages: Scraping page {current_page_num} got data but lost UL ref.")
                             # Continue but note potential instability
                        elif current_ul_ref is None and not page_data:
                             logging.error(f"scrape_all_list_pages: Scraping page {current_page_num} failed after successful navigation.")
                             break_loop = True # Indicate fatal error for this phase
                             break # Exit inner try/except
                        last_ul_element_ref = current_ul_ref # Update reference for the *next* wait
                        scraped_list_data.extend(page_data)

                    except (TimeoutException, NoSuchElementException):
                        logging.info("Phân trang: -> Không tìm thấy 'Sau'. Đã đến trang cuối của dãy số hiển thị. Chuyển sang chế độ 'go_to_last'.")
                        scraping_mode = "go_to_last"
                        action_taken = 'none' # Action handled in next iteration
                    except Exception as e:
                        logging.error(f"Phân trang: Lỗi trong chế độ forward: {e}. Thoát vòng lặp list.")
                        break_loop = True
                        break

                elif scraping_mode == "go_to_last":
                    logging.info("Phân trang: Thực hiện: Click 'Trước' (1 lần, không scrape - để đảm bảo nút cuối xuất hiện)...")
                    try:
                        prev_button_locator = (By.XPATH, "//div[@class='paging']//a[text()='Trước']")
                        prev_button = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.element_to_be_clickable(prev_button_locator))
                        driver.execute_script("arguments[0].click();", prev_button)
                        success, new_ul_ref = wait_for_list_page_load(driver, last_ul_element_ref)
                        if not success or new_ul_ref is None:
                            raise Exception("Chờ trang trước thất bại (go_to_last step 1)")
                        last_ul_element_ref = new_ul_ref # Update reference
                        logging.info("Phân trang: -> Đã click 'Trước'.")
                        time.sleep(SHORT_PAUSE)

                        logging.info("Phân trang: Thực hiện: Click 'Cuối »'...")
                        last_button_locator = (By.XPATH, "//div[@class='paging']//a[text()='Cuối »']")
                        last_button = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.element_to_be_clickable(last_button_locator))
                        driver.execute_script("arguments[0].click();", last_button)
                        success, new_ul_ref = wait_for_list_page_load(driver, last_ul_element_ref)
                        if not success or new_ul_ref is None:
                            raise Exception("Chờ trang cuối thất bại (go_to_last step 2)")
                        last_ul_element_ref = new_ul_ref # Update reference

                        logging.info("Phân trang: -> Đã click 'Cuối »'.")
                        # Scrape this last page at the start of the *next* iteration
                        scraping_mode = "last_page_reached"
                        action_taken = 'last'
                        # Attempt to determine the actual last page number from visible links
                        try:
                            page_links = driver.find_elements(By.XPATH, "//div[@class='paging']/a[number(text())]")
                            if page_links:
                                 last_page_num_detected = max([int(a.text) for a in page_links if a.text.isdigit()])
                                 current_page_num = last_page_num_detected
                                 logging.info(f"Phân trang: -> Đã đến trang cuối (Trang {current_page_num}).")
                            else:
                                 logging.warning("Phân trang: Cảnh báo: Không tìm thấy số trang trong phân trang để xác định trang cuối. Giữ nguyên số trang logic.")
                        except Exception as e_get_last_num:
                            logging.warning(f"Phân trang: Cảnh báo: Lỗi khi cố gắng xác định số trang cuối tự động ({e_get_last_num}).")

                    except Exception as e_go_to_last:
                        logging.error(f"Phân trang: Lỗi trong chế độ go_to_last: {e_go_to_last}. Thoát vòng lặp list.")
                        break_loop = True
                        break

                elif scraping_mode == "last_page_reached":
                     # The scrape of the last page happens at the start of this iteration
                     # Now transition to backward mode for the next iteration
                     logging.info("Phân trang: -> Đã scrape trang cuối. Chuyển sang chế độ 'backward'.")
                     scraping_mode = "backward"
                     action_taken = 'none' # Action handled in next iteration


                elif scraping_mode == "backward":
                    # Logic: Click Trước -> Wait -> Scrape (at start of next loop) -> Repeat `pages_left_to_scrape_backwards` times
                    if pages_left_to_scrape_backwards > 0:
                        logging.info(f"Phân trang: Thực hiện: Click 'Trước' (còn {pages_left_to_scrape_backwards} trang cần scrape lùi)")
                        try:
                            prev_button_locator = (By.XPATH, "//div[@class='paging']//a[text()='Trước']")
                            prev_button = WebDriverWait(driver, WAIT_TIMEOUT).until(EC.element_to_be_clickable(prev_button_locator))
                            driver.execute_script("arguments[0].click();", prev_button)

                            success, new_ul_ref = wait_for_list_page_load(driver, last_ul_element_ref)
                            if not success or new_ul_ref is None:
                                raise Exception("Chờ trang trước (backward) thất bại")
                            last_ul_element_ref = new_ul_ref # Update reference

                            current_page_num -= 1 # Decrement logical page number
                            action_taken = 'prev'

                            # The scraping for this backward step will happen at the start of the *next* loop iteration

                            pages_left_to_scrape_backwards -= 1 # Decrement counter *after* successfully moving back

                            if pages_left_to_scrape_backwards == 0:
                                 logging.info("Phân trang: Đã thực hiện đủ số lần click 'Trước' cho chế độ backward. Loop sẽ kết thúc sau lần scrape của trang hiện tại.")


                        except Exception as e_prev_back:
                            logging.error(f"Phân trang: Lỗi khi click 'Trước' trong chế độ backward: {e_prev_back}. Dừng lại.")
                            break_loop = True
                            break # Exit inner try/except

                    else:
                         # We have scraped all required backward pages (the scraping happens at the loop start)
                         logging.info("Phân trang: Đã scrape đủ số trang backward. Kết thúc giai đoạn 1.")
                         action_taken = 'break'
                         break_loop = True # Set flag to break the outer loop
                         break # Exit inner try/except


            except Exception as e_inner_loop:
                logging.error(f"scrape_all_list_pages: Lỗi trong vòng lặp phân trang (chế độ {scraping_mode}, trang {current_page_num}): {e_inner_loop}. Thoát vòng lặp list.")
                break_loop = True
            finally:
                 # This finally block within the while loop handles breaking
                 if break_loop:
                     break # Exit the while loop


    except Exception as e_outer_loop:
         logging.critical(f"scrape_all_list_pages: Lỗi nghiêm trọng trong quá trình scrape list: {e_outer_loop}")
         # The loop is already broken by inner handlers or this outer catch

    logging.info(f"\n--- Hoàn tất scrape dữ liệu list. Tổng cộng: {len(scraped_list_data)} mục ---")

    # Remove duplicates based on link after scraping all list pages
    if scraped_list_data:
        logging.info(f"Số mục từ list trước khi loại bỏ trùng lặp: {len(scraped_list_data)}")
        df_list = pd.DataFrame(scraped_list_data)
        # Drop duplicates based on link, keeping the last one found
        df_unique_list = df_list.drop_duplicates(subset=['DuongLink'], keep='last')
        processed_list_data = df_unique_list.to_dict('records') # Convert back to list of dicts
        logging.info(f"Số mục duy nhất từ list sau khi loại bỏ trùng lặp: {len(processed_list_data)}")
        return processed_list_data
    else:
        logging.warning("Không có dữ liệu nào được thu thập từ list pages.")
        return []


def scrape_document_details(driver: webdriver.Chrome, item_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Navigates to a single document's URL and scrapes full content and property details.
    Updates the item_data dictionary in place.
    Returns the updated item_data dictionary.
    """
    document_url = item_data.get('DuongLink')
    document_title_preview = item_data.get('TenVanBan', 'N/A')[:50] + '...' if item_data.get('TenVanBan') else 'N/A'

    if not document_url or not isinstance(document_url, str) or not document_url.startswith('http'):
        logging.warning(f"scrape_document_details: URL không hợp lệ hoặc bị thiếu cho '{document_title_preview}'. Bỏ qua scrape chi tiết.")
        # item_data already has None placeholders, return as is
        return item_data

    logging.info(f"scrape_document_details: Truy cập trang: {document_title_preview} ({document_url})")

    try:
        # Navigate to the individual document page
        driver.get(document_url)

        # 1. Scrape Full Content ('Toàn văn')
        logging.info("scrape_document_details: Tìm và lấy nội dung 'Toàn văn'...")
        content_div_locator = (By.ID, "toanvancontent")
        try:
            content_div = WebDriverWait(driver, CONTENT_WAIT_TIMEOUT).until(
                EC.visibility_of_element_located(content_div_locator)
            )
            item_data['NoiDung'] = content_div.text.strip()
            logging.info(f"scrape_document_details: Lấy nội dung 'Toàn văn' thành công ({len(item_data.get('NoiDung', ''))} ký tự).")
        except TimeoutException:
            logging.warning(f"scrape_document_details: Lỗi: Timeout khi chờ nội dung 'Toàn văn' ({document_url}). Nội dung sẽ là null.")
            item_data['NoiDung'] = None
        except Exception as e_content:
            logging.warning(f"scrape_document_details: Lỗi không xác định khi lấy nội dung 'Toàn văn' ({document_url}): {e_content}. Nội dung sẽ là null.")
            item_data['NoiDung'] = None


        # 2. Find and click link "Thuộc tính"
        logging.info("scrape_document_details: Tìm và click link 'Thuộc tính'...")
        properties_page_accessed = False
        try:
            properties_link_locator = (By.XPATH, "//a[.//b[contains(@class, 'properties') and normalize-space(.)='Thuộc tính']]")
            properties_link = WebDriverWait(driver, WAIT_TIMEOUT).until(
                EC.element_to_be_clickable(properties_link_locator)
            )
            # Get URL before clicking, just in case (though execute_script click often prevents new tab)
            properties_url = properties_link.get_attribute('href')
            # logging.debug(f"  -> Tìm thấy link Thuộc tính: {properties_url}")

            # Use execute_script click for better reliability
            driver.execute_script("arguments[0].click();", properties_link)
            properties_page_accessed = True

        except (TimeoutException, NoSuchElementException) as e_prop_link:
             logging.warning(f"scrape_document_details: Lỗi: Không tìm thấy link 'Thuộc tính' ({document_url}): {e_prop_link}. Bỏ qua scrape thuộc tính.")
             # Properties fields will remain None

        if properties_page_accessed:
            # 3. Wait for Properties section to load and scrape info
            logging.info("scrape_document_details: Chờ phần 'Thuộc tính' tải...")
            linh_vuc_label_locator = (By.XPATH, "//td[contains(@class,'label') and normalize-space(.)='Lĩnh vực']")
            try:
                WebDriverWait(driver, WAIT_TIMEOUT).until(
                    EC.visibility_of_element_located(linh_vuc_label_locator)
                )
                logging.info("scrape_document_details: Phần 'Thuộc tính' đã tải.")

                # 3.1 Scrape "Lĩnh vực"
                try:
                    linh_vuc_label_td = driver.find_element(*linh_vuc_label_locator)
                    linh_vuc_value_td = linh_vuc_label_td.find_element(By.XPATH, "./following-sibling::td")
                    item_data['LinhVuc'] = linh_vuc_value_td.text.strip()
                    logging.info(f"scrape_document_details: Lấy Lĩnh vực: {item_data['LinhVuc']}")
                except NoSuchElementException:
                    logging.warning("scrape_document_details: Không tìm thấy giá trị Lĩnh vực. Giá trị sẽ là null.")
                    item_data['LinhVuc'] = None

                # 3.2 Scrape "Loại văn bản" (from Properties)
                try:
                    loai_vb_label_locator = (By.XPATH, "//td[contains(@class,'label') and normalize-space(.)='Loại văn bản']")
                    loai_vb_label_td = driver.find_element(*loai_vb_label_locator)
                    loai_vb_value_td = loai_vb_label_td.find_element(By.XPATH, "./following-sibling::td")
                    item_data['LoaiVanBan_ThuocTinh'] = loai_vb_value_td.text.strip()
                    logging.info(f"scrape_document_details: Lấy Loại văn bản (Thuộc tính): {item_data['LoaiVanBan_ThuocTinh']}")
                except NoSuchElementException:
                    logging.warning("scrape_document_details: Không tìm thấy giá trị Loại văn bản (Thuộc tính). Giá trị sẽ là null.")
                    item_data['LoaiVanBan_ThuocTinh'] = None

            except TimeoutException:
                logging.warning(f"scrape_document_details: Lỗi: Timeout khi chờ phần 'Thuộc tính' tải ({document_url}). Các thuộc tính sẽ là null.")
                item_data['LinhVuc'] = None # Ensure null on timeout
                item_data['LoaiVanBan_ThuocTinh'] = None
            except Exception as e_scrape_props:
                logging.warning(f"scrape_document_details: Lỗi không xác định khi scrape thuộc tính ({document_url}): {e_scrape_props}. Các thuộc tính sẽ là null.")
                item_data['LinhVuc'] = None # Ensure null on error
                item_data['LoaiVanBan_ThuocTinh'] = None


    except (TimeoutException, WebDriverException) as e_scrape_detail_outer:
        logging.error(f"scrape_document_details: Lỗi nghiêm trọng khi xử lý link {document_url}: {type(e_scrape_detail_outer).__name__} - {e_scrape_detail_outer}. Dữ liệu chi tiết có thể thiếu.")
        # Fields are already None if not successfully scraped

    return item_data # Return the potentially updated dictionary

def process_all_documents(driver: webdriver.Chrome, list_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Iterates through the list of documents and scrapes detailed info for each.
    Returns the list with enriched data.
    """
    logging.info(f"\n--- Bắt đầu scrape chi tiết từng link (Giai đoạn 2) ---")
    processed_data = []
    errors_during_detail_scrape = 0
    total_items = len(list_data)
    # Store current window handle if needed, though get() often stays in same tab
    original_window = driver.current_window_handle

    for index, item_data in enumerate(list_data):
        logging.info(f"\nĐang xử lý chi tiết mục {index + 1}/{total_items}...")
        try:
            # Call the function to scrape details for this item
            updated_item_data = scrape_document_details(driver, item_data)
            processed_data.append(updated_item_data) # Add the updated item to the results list

        except Exception as e:
            # Catch unexpected errors during the process of a single item
            logging.error(f"process_all_documents: Lỗi không xác định khi xử lý mục {index+1}: {e}. Item data: {item_data.get('DuongLink', 'N/A')}. Bỏ qua.")
            errors_during_detail_scrape += 1
            processed_data.append(item_data) # Append original item data with None details if failure occurred

        time.sleep(SHORT_PAUSE) # Pause between visiting links

    logging.info(f"\n--- Hoàn tất scrape chi tiết. Xử lý {total_items} mục, gặp lỗi với {errors_during_detail_scrape} mục ---")
    return processed_data

def save_data_to_json(data: List[Dict[str, Any]], file_path: Path):
    """Saves the collected data to a JSON file."""
    if not data:
        logging.warning("save_data_to_json: Không có dữ liệu để lưu.")
        return

    logging.info(f"\nĐang lưu dữ liệu đầy đủ vào file JSON: {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4) # indent=4 để dễ đọc
        logging.info("save_data_to_json: Đã lưu dữ liệu thành công.")
    except Exception as e_save:
        logging.error(f"save_data_to_json: Lỗi khi lưu file JSON: {e_save}")

# --- Main Execution Flow ---

if __name__ == "__main__":
    driver = None
    try:
        # Giai đoạn 0: Khởi tạo Driver
        driver = setup_driver()
        if driver is None:
            raise Exception("Không thể khởi tạo WebDriver.")

        # Giai đoạn 1: Thiết lập tìm kiếm và scrape dữ liệu từ list
        setup_advanced_search(driver)
        scraped_list_data_unique = scrape_all_list_pages(driver)

        # Giai đoạn 2: Scrape chi tiết từng link từ dữ liệu list đã có
        if scraped_list_data_unique:
             final_data = process_all_documents(driver, scraped_list_data_unique)
        else:
             logging.warning("Không có dữ liệu list nào để xử lý chi tiết.")
             final_data = [] # Ensure final_data is an empty list

        # Giai đoạn 3: Lưu dữ liệu cuối cùng
        save_data_to_json(final_data, OUTPUT_JSON_FILE)

    except Exception as e_main:
        logging.critical(f"\n!!! Lỗi nghiêm trọng trong luồng chính của chương trình: {e_main}")
        if driver:
            try:
                driver.save_screenshot("error_screenshot_main.png")
                logging.error("Đã lưu ảnh chụp màn hình lỗi cuối cùng.")
            except Exception as e_ss:
                 logging.error(f"Không thể lưu ảnh chụp màn hình lỗi cuối cùng: {e_ss}")

    finally:
        # Dọn dẹp: Đóng trình duyệt
        logging.info("\nĐang đóng trình duyệt...")
        if driver:
            driver.quit()

    logging.info("\nChương trình kết thúc.")