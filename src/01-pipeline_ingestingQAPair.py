import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# ===== 1. Cấu hình Chrome Options =====
# Cấu hình các tùy chọn cho trình duyệt Chrome
options = Options()
# Uncomment dòng dưới nếu bạn muốn chạy ẩn trình duyệt (không hiển thị giao diện)
# options.add_argument("--headless")
options.add_argument("--disable-gpu") # Vô hiệu hóa tăng tốc phần cứng GPU
options.add_argument("--no-sandbox") # Chạy ở chế độ sandbox (tăng bảo mật)
options.add_argument("--disable-dev-shm-usage") # Tránh lỗi liên quan đến /dev/shm

# Khởi tạo trình duyệt Chrome
driver = webdriver.Chrome(options=options)

# ===== 2. Truy cập trang chính chứa danh sách hỏi đáp =====
# URL của trang danh sách hỏi đáp
main_url = "https://dichvucong.moh.gov.vn/web/guest/hoi-dap?p_p_id=hoidap_WAR_oephoidapportlet&_hoidap_WAR_oephoidapportlet_delta=9999"
print(f"🌐 Đang truy cập trang chính: {main_url}")
driver.get(main_url)

# Chờ một chút để trang tải hoàn toàn, hoặc chờ một phần tử cụ thể xuất hiện
try:
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.panel.panel-default"))
    )
    print("✅ Trang chính đã tải xong.")
except Exception as e:
    print(f"❌ Lỗi khi chờ trang chính tải: {e}")
    driver.quit()
    exit()

# Lấy handle của tab chính để có thể quay lại sau khi xử lý các tab con
main_tab = driver.current_window_handle

# ===== 3. Thu thập các liên kết chi tiết từ trang chính =====
# Tìm tất cả các khối hỏi đáp trên trang chính
qa_blocks = driver.find_elements(By.CSS_SELECTOR, "div.panel.panel-default")
print(f"🔍 Tìm thấy {len(qa_blocks)} khối hỏi đáp trên trang chính.")

# Danh sách lưu trữ tất cả các cặp hỏi-đáp chi tiết
all_qa_details = []

# Duyệt qua từng khối hỏi đáp tìm được
for i, block in enumerate(qa_blocks):
    try:
        # Lấy liên kết chi tiết từ khối hỏi đáp
        link_el = block.find_element(By.CSS_SELECTOR, "a[href]")
        detail_link = link_el.get_attribute("href")

        # Lấy số câu hỏi con (số trao đổi) từ badge, nếu có
        try:
            badge_el = block.find_element(By.CSS_SELECTOR, "span.badge.badge-primary.badge-pill")
            badge_count = badge_el.text.strip()
        except:
            badge_count = "0" # Mặc định là 0 nếu không tìm thấy badge

        print(f"\n--- Đang xử lý khối hỏi đáp {i+1}/{len(qa_blocks)} ---")
        print(f"🔗 Link chi tiết: {detail_link}")
        print(f"💬 Số trao đổi (dự kiến): {badge_count}")

        # ===== 4. Mở liên kết chi tiết trong tab mới và trích xuất dữ liệu =====
        # Mở link chi tiết trong một tab mới của trình duyệt
        driver.execute_script("window.open(arguments[0]);", detail_link)
        # Chờ một chút để tab mới được mở
        time.sleep(1)

        # Chuyển quyền điều khiển của Selenium sang tab mới nhất
        tabs = driver.window_handles
        driver.switch_to.window(tabs[-1])

        try:
            # Chờ cho một phần tử đặc trưng của trang chi tiết xuất hiện
            # Ví dụ: chờ nút "showtraloi" hoặc nội dung câu hỏi chính
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.question-content, [onclick*='showtraloi']"))
            )
            print("✅ Trang chi tiết đã tải xong.")

            # Tìm và click tất cả các nút "showtraloi" để hiển thị nội dung trả lời/trao đổi
            show_buttons = driver.find_elements(By.CSS_SELECTOR, '[onclick*="showtraloi"]')
            print(f"🔘 Tìm thấy {len(show_buttons)} nút 'showtraloi' cần click.")
            for btn in show_buttons:
                try:
                    # Click bằng JavaScript để đảm bảo hoạt động ngay cả khi nút không hiển thị đầy đủ
                    driver.execute_script("arguments[0].click();", btn)
                    # Chờ một chút sau mỗi lần click để nội dung hiển thị
                    time.sleep(0.5)
                except Exception as click_e:
                    print(f"⚠️ Không thể click một nút 'showtraloi': {click_e}")
                    # Tiếp tục với các nút khác ngay cả khi một nút lỗi

            # Lấy toàn bộ mã nguồn HTML của trang sau khi đã click các nút
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # ===== 5. Trích xuất các cặp Hỏi - Đáp từ trang chi tiết =====
            # Tìm tất cả các span có class "primary--text" (thường là câu hỏi)
            question_spans = soup.find_all("span", class_="primary--text")
            print(f"🔍 Tìm thấy {len(question_spans)} cặp Hỏi-Đáp tiềm năng trên trang chi tiết.")

            # Duyệt qua từng span tìm được để lấy câu hỏi và câu trả lời tương ứng
            for span in question_spans:
                question_text = span.get_text(strip=True)

                # Tìm thẻ <p> thứ 2 ngay sau thẻ span hiện tại
                # Dựa trên cấu trúc HTML quan sát được, câu trả lời thường nằm trong thẻ <p> thứ 2
                p_tags_after_span = span.find_all_next("p", limit=2)
                answer_text = "Không tìm thấy câu trả lời phù hợp" # Mặc định

                if len(p_tags_after_span) >= 2:
                    # Lấy nội dung của thẻ <p> thứ 2
                    answer_text = p_tags_after_span[1].get_text(strip=True)
                elif len(p_tags_after_span) == 1:
                     # Nếu chỉ có 1 thẻ p, có thể đó là câu trả lời duy nhất
                     answer_text = p_tags_after_span[0].get_text(strip=True)


                # Thêm cặp hỏi-đáp và link nguồn vào danh sách kết quả
                all_qa_details.append({
                    "question": question_text,
                    "answer": answer_text,
                    "source_link": detail_link,
                    "main_block_badge": badge_count # Giữ lại thông tin badge từ trang chính
                })
                # print(f"  - Trích xuất QA: {question_text[:50]}...") # In ra một phần câu hỏi để theo dõi

        except Exception as detail_e:
            print(f"❌ Lỗi khi xử lý trang chi tiết {detail_link}: {detail_e}")
            # Nếu có lỗi ở trang chi tiết, vẫn đóng tab và tiếp tục với link khác

        finally:
            # Đóng tab chi tiết hiện tại
            driver.close()
            # Chuyển quyền điều khiển trở lại tab chính
            driver.switch_to.window(main_tab)
            # Chờ một chút trước khi xử lý khối tiếp theo
            time.sleep(0.5)

    except Exception as block_e:
        print(f"❌ Lỗi khi xử lý khối hỏi đáp trên trang chính (index {i}): {block_e}")
        # Nếu lỗi ở bước này, có thể bỏ qua khối này và tiếp tục

# ===== 6. Ghi kết quả ra file CSV =====
# Tạo DataFrame từ danh sách các cặp hỏi-đáp đã thu thập
df_result = pd.DataFrame(all_qa_details)

# Tên file đầu ra
output_file = "qa_dvc_combined.csv"
# Ghi DataFrame ra file CSV với encoding UTF-8 BOM để hiển thị tiếng Việt chính xác trong Excel
df_result.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n🎉 Hoàn tất quá trình thu thập dữ liệu!")
print(f"💾 Đã lưu {len(df_result)} cặp Hỏi-Đáp vào file: {output_file}")

# ===== 7. Đóng trình duyệt =====
driver.quit()
print("Browser đã đóng.")