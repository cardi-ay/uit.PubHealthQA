import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

def setup_driver(headless=False):
    """
    Cấu hình và khởi tạo Selenium WebDriver cho Chrome.

    Args:
        headless (bool): Chạy trình duyệt ở chế độ ẩn (không hiển thị giao diện) nếu True.

    Returns:
        webdriver.Chrome: Đối tượng WebDriver đã được cấu hình.
    """
    print("⚙️ Đang cấu hình trình duyệt...")
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080") # Đặt kích thước cửa sổ để đảm bảo các phần tử hiển thị

    driver = webdriver.Chrome(options=options)
    print("✅ Trình duyệt đã khởi tạo.")
    return driver

def navigate_to_main_page(driver, url):
    """
    Truy cập trang chính và chờ cho các phần tử chính tải xong.

    Args:
        driver (webdriver.Chrome): Đối tượng WebDriver.
        url (str): URL của trang chính.

    Returns:
        bool: True nếu truy cập thành công và các phần tử tải xong, False nếu ngược lại.
    """
    print(f"🌐 Đang truy cập trang chính: {url}")
    try:
        driver.get(url)
        # Chờ cho các khối hỏi đáp chính xuất hiện
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.panel.panel-default"))
        )
        # Thêm chờ cho phần tử chứa topic xuất hiện trong khối đầu tiên để đảm bảo cấu trúc tải xong
        WebDriverWait(driver, 20).until(
             EC.presence_of_element_located((By.CSS_SELECTOR, "div.panel.panel-default:nth-child(1) div.panel-heading div.col-md-9 b"))
        )
        print("✅ Trang chính đã tải xong.")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi chờ trang chính tải: {e}")
        return False

def extract_main_qa_blocks(driver):
    """
    Trích xuất các khối hỏi đáp chính từ trang hiện tại.

    Args:
        driver (webdriver.Chrome): Đối tượng WebDriver.

    Returns:
        list: Danh sách các phần tử web đại diện cho các khối hỏi đáp.
    """
    try:
        qa_blocks = driver.find_elements(By.CSS_SELECTOR, "div.panel.panel-default")
        print(f"🔍 Tìm thấy {len(qa_blocks)} khối hỏi đáp trên trang chính.")
        return qa_blocks
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất khối hỏi đáp chính: {e}")
        return []

def process_detail_page(driver, detail_link, main_tab_handle, badge_count, main_topic):
    """
    Mở trang chi tiết trong tab mới, trích xuất các cặp hỏi-đáp và đóng tab.

    Args:
        driver (webdriver.Chrome): Đối tượng WebDriver.
        detail_link (str): URL của trang chi tiết.
        main_tab_handle (str): Handle của tab chính để quay lại.
        badge_count (str): Số trao đổi từ trang chính (để lưu vào kết quả).
        main_topic (str): Chủ đề chính được lấy từ trang danh sách.

    Returns:
        list: Danh sách các dictionary, mỗi dictionary là một cặp hỏi-đáp từ trang chi tiết.
    """
    qa_pairs = []
    print(f"--- Đang xử lý link chi tiết: {detail_link} ---")
    try:
        # Mở link chi tiết trong một tab mới
        driver.execute_script("window.open(arguments[0]);", detail_link)
        time.sleep(1.5) # Chờ thêm một chút cho tab mới mở

        # Chuyển quyền điều khiển sang tab mới
        tabs = driver.window_handles
        driver.switch_to.window(tabs[-1])

        try:
            # Chờ cho một phần tử đặc trưng của trang chi tiết xuất hiện
            WebDriverWait(driver, 15).until( # Tăng thời gian chờ
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.question-content, [onclick*='showtraloi']"))
            )
            # print("✅ Trang chi tiết đã tải xong.") # Bỏ bớt log chi tiết để gọn hơn

            # Tìm và click tất cả các nút "showtraloi"
            show_buttons = driver.find_elements(By.CSS_SELECTOR, '[onclick*="showtraloi"]')
            # print(f"🔘 Tìm thấy {len(show_buttons)} nút 'showtraloi' cần click.")
            for btn in show_buttons:
                try:
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(0.5) # Chờ nội dung hiển thị sau click
                except Exception as click_e:
                    # print(f"⚠️ Không thể click một nút 'showtraloi': {click_e}")
                    pass # Bỏ qua lỗi click một nút và tiếp tục

            # Lấy mã nguồn HTML sau khi click
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Trích xuất các cặp Hỏi - Đáp
            question_spans = soup.find_all("span", class_="primary--text")
            # print(f"🔍 Tìm thấy {len(question_spans)} cặp Hỏi-Đáp tiềm năng.")

            for span in question_spans:
                question_text = span.get_text(strip=True)
                answer_text = "Không tìm thấy câu trả lời phù hợp" # Mặc định

                # Tìm thẻ <p> thứ 2 ngay sau thẻ span
                p_tags_after_span = span.find_all_next("p", limit=2)

                if len(p_tags_after_span) >= 2:
                    answer_text = p_tags_after_span[1].get_text(strip=True)
                elif len(p_tags_after_span) == 1:
                     answer_text = p_tags_after_span[0].get_text(strip=True)

                qa_pairs.append({
                    "main_topic": main_topic, # Thêm chủ đề chính vào mỗi cặp QA
                    "question": question_text,
                    "answer": answer_text,
                    "source_link": detail_link,
                    "main_block_badge": badge_count
                })

        except Exception as detail_process_e:
            print(f"❌ Lỗi khi xử lý nội dung trang chi tiết {detail_link}: {detail_process_e}")

    except Exception as detail_open_e:
        print(f"❌ Lỗi khi mở hoặc chuyển tab cho link {detail_link}: {detail_open_e}")

    finally:
        # Đóng tab chi tiết và quay lại tab chính
        try:
            driver.close()
            driver.switch_to.window(main_tab_handle)
            time.sleep(0.5) # Chờ một chút sau khi quay lại tab chính
        except Exception as switch_e:
            print(f"❌ Lỗi khi đóng tab hoặc chuyển về tab chính: {switch_e}")
            # Nếu không thể quay lại tab chính, có thể cần khởi động lại driver hoặc xử lý lỗi nghiêm trọng hơn

    return qa_pairs

def main_scraper(main_url, output_file="./data/bronze/raw_QAPair.csv", headless=False):
    """
    Chức năng chính để thu thập dữ liệu hỏi đáp.

    Args:
        main_url (str): URL của trang danh sách hỏi đáp.
        output_file (str): Tên file CSV để lưu kết quả.
        headless (bool): Chạy trình duyệt ở chế độ ẩn nếu True.
    """
    driver = None # Khởi tạo driver là None ban đầu
    try:
        # 1. Cấu hình và khởi tạo driver
        driver = setup_driver(headless)

        # 2. Truy cập trang chính
        if not navigate_to_main_page(driver, main_url):
            print("❌ Không thể truy cập trang chính. Dừng chương trình.")
            return

        # Lấy handle của tab chính
        main_tab = driver.current_window_handle

        # 3. Trích xuất các khối hỏi đáp từ trang chính
        qa_blocks = extract_main_qa_blocks(driver)

        all_qa_details = []

        # Define the specific CSS selector for the main topic within a block
        # This selector targets the <b> tag containing the topic text
        main_topic_selector = "div.panel-heading div.col-md-9 b"

        # 4. Duyệt qua từng khối và xử lý trang chi tiết
        for i, block in enumerate(qa_blocks):
            try:
                # Lấy link chi tiết
                link_el = block.find_element(By.CSS_SELECTOR, "a[href]")
                detail_link = link_el.get_attribute("href")

                # Lấy CHỦ ĐỀ CHÍNH từ thẻ <b> sử dụng selector được cung cấp
                try:
                    topic_el = block.find_element(By.CSS_SELECTOR, main_topic_selector)
                    main_topic = topic_el.text.strip()
                except Exception as topic_e:
                    main_topic = "Không tìm thấy chủ đề chính"
                    print(f"⚠️ Không tìm thấy chủ đề chính cho khối {i+1}: {topic_e}")


                # Lấy số trao đổi (badge count)
                try:
                    badge_el = block.find_element(By.CSS_SELECTOR, "span.badge.badge-primary.badge-pill")
                    badge_count = badge_el.text.strip()
                except:
                    badge_count = "0"

                # Xử lý trang chi tiết và thu thập các cặp hỏi-đáp, truyền kèm main_topic
                qa_pairs_from_detail = process_detail_page(driver, detail_link, main_tab, badge_count, main_topic)
                all_qa_details.extend(qa_pairs_from_detail)

            except Exception as e:
                print(f"❌ Lỗi tổng quát khi xử lý khối {i+1}: {e}")
                # Tiếp tục vòng lặp ngay cả khi một khối bị lỗi

        # 5. Ghi kết quả ra file CSV
        if all_qa_details:
            df_result = pd.DataFrame(all_qa_details)
            df_result.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"\n🎉 Hoàn tất quá trình thu thập dữ liệu!")
            print(f"💾 Đã lưu {len(df_result)} cặp Hỏi-Đáp vào file: {output_file}")
        else:
            print("\n⚠️ Không thu thập được cặp Hỏi-Đáp nào.")

    except Exception as main_e:
        print(f"\n❌ Đã xảy ra lỗi nghiêm trọng trong quá trình chính: {main_e}")

    finally:
        # 6. Đóng trình duyệt
        if driver:
            driver.quit()
            print("Browser đã đóng.")

# ===== Điểm bắt đầu thực thi chương trình =====
if __name__ == "__main__":
    main_page_url = "https://dichvucong.moh.gov.vn/web/guest/hoi-dap?p_p_id=hoidap_WAR_oephoidapportlet&_hoidap_WAR_oephoidapportlet_delta=9999"
    main_scraper(main_page_url, headless=False) # Đặt headless=True nếu muốn chạy ẩn

