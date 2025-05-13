import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# ===== 1. Đọc danh sách link =====
input_file = "qa_dvc_with_badge.csv"
df_links = pd.read_csv(input_file)
links = df_links["link"].dropna().unique().tolist()
print(f"🔗 Tổng số link: {len(links)}")

# ===== 2. Cấu hình trình duyệt =====
options = Options()
# options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=options)
data = []

# ===== 3. Duyệt từng link =====
for idx, link in enumerate(links, start=1):
    try:
        print(f"\n📄 ({idx}/{len(links)}) Đang xử lý: {link}")
        driver.get(link)
        time.sleep(2)

        # Click tất cả nút showtraloi
        buttons = driver.find_elements(By.CSS_SELECTOR, '[onclick*="showtraloi"]')
        print(f"🔘 Tìm thấy {len(buttons)} nút cần click.")
        for btn in buttons:
            try:
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.3)
            except:
                print("⚠️ Không thể click một nút.")

        # Parse nội dung
        soup = BeautifulSoup(driver.page_source, "html.parser")
        spans = soup.find_all("span", class_="primary--text")
        print(f"🔍 Tìm thấy {len(spans)} câu hỏi.")

        for span in spans:
            question_text = span.get_text(strip=True)

            # Lấy đúng <p> thứ 2 sau <span>
            p_tags = span.find_all_next("p", limit=2)
            if len(p_tags) >= 2:
                answer_text = p_tags[1].get_text(strip=True)
            else:
                answer_text = "Không tìm thấy câu trả lời"

            data.append({
                "question": question_text,
                "answer": answer_text,
                "source_link": link
            })

    except Exception as e:
        print(f"❌ Lỗi tại link {link}: {e}")
        continue

# ===== 4. Ghi kết quả ra file =====
df_result = pd.DataFrame(data)
df_result.to_csv("qa_pthu2_only.csv", index=False, encoding="utf-8-sig")
print(f"\n🎉 Hoàn tất! Đã lưu {len(df_result)} cặp QA vào file qa_pthu2_only.csv")

driver.quit()