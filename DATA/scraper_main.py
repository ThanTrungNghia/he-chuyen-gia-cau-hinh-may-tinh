import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from sqlalchemy import create_engine
from sqlalchemy.types import NVARCHAR
import urllib
import re
import json

# SỬ DỤNG PLAYWRIGHT CHO CHẶNG 1 ĐỂ TƯƠNG TÁC GIAO DIỆN
from playwright.sync_api import sync_playwright

# ==========================================
# 1. CẤU HÌNH KẾT NỐI SQL SERVER
# ==========================================
SERVER_NAME = r'LAPTOP-V1E19NDA\SQLEXPRESS'
DATABASE_NAME = 'HeChuyenGiaPC'
DRIVER = 'ODBC Driver 17 for SQL Server'

connection_string = f"Driver={{{DRIVER}}};Server={SERVER_NAME};Database={DATABASE_NAME};Trusted_Connection=yes;"
params = urllib.parse.quote_plus(connection_string)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9"
}
BASE_URL = "https://phongvu.vn"

CATEGORIES = {
    "CPU": "https://phongvu.vn/c/cpu",
    "Mainboard": "https://phongvu.vn/c/mainboard-bo-mach-chu",
    "VGA": "https://phongvu.vn/c/vga-card-man-hinh",
    "RAM": "https://phongvu.vn/c/ram",
    "Storage": "https://phongvu.vn/c/o-cung",
    "Case": "https://phongvu.vn/c/case",
    "Cooler": "https://phongvu.vn/c/tan-nhiet",
    "PSU": "https://phongvu.vn/c/psu-nguon-may-tinh"
}

# ==========================================
# CHẶNG 1: GIẢ LẬP CLICK "XEM THÊM" (GIỮ NGUYÊN VER 8.2 CHỐNG TIMEOUT)
# ==========================================
def get_products_from_list(category_url, category_name, max_items=200):
    products = []
    seen_links = set()
    item_count = 1 
    
    print("      ⏳ Đang khởi động lõi Playwright Auto-Click...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=HEADERS['User-Agent'])
        page = context.new_page()

        print(f"      -> Mở trang gốc: {category_url}")
        
        # FIX TIMEOUT VER 8.2
        try:
            page.goto(category_url, wait_until="networkidle", timeout=30000)
        except Exception:
            print("      [i] Chờ mạng tĩnh bị Timeout do quảng cáo ngầm. Ép tiếp tục chạy!")
            pass

        # VÒNG LẶP CLICK XEM THÊM TỚI 200 SP
        while True:
            current_count = page.locator('div.css-13w7uog').count()
            print(f"      -> Đang hiển thị {current_count} sản phẩm...")
            
            if current_count >= max_items:
                print(f"      [v] Đã đạt hoặc vượt ngưỡng {max_items} sản phẩm. Dừng load thêm.")
                break
                
            try:
                btn = page.get_by_text("Xem thêm sản phẩm").last
                if btn.is_visible(timeout=3000):
                    btn.click(force=True)
                    
                    page.wait_for_function(
                        f"document.querySelectorAll('div.css-13w7uog').length > {current_count}", 
                        timeout=15000
                    )
                    page.wait_for_timeout(1000)
                else:
                    print("      [i] Không tìm thấy nút 'Xem thêm' (Đã load toàn bộ danh mục).")
                    break
            except Exception as e:
                print("      [i] Nút 'Xem thêm' đã biến mất hoặc không thể tương tác. Kết thúc load trang.")
                break

        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')
        product_blocks = soup.find_all('div', class_='css-13w7uog')
        
        # CẮT ĐÚNG GIỚI HẠN
        for block in product_blocks[:max_items]:
            try:
                name_tag = block.find('h3', class_='css-1xdyrhj')
                name = name_tag.text.strip() if name_tag else ""
                
                # --- [FIX GÍA ẢO CHẶNG 1] Lọc kỹ để né các thẻ "Giảm giá", "Tặng" ---
                price_str = ""
                price_tags = block.find_all(string=lambda text: text and '₫' in text and len(text.strip()) < 50)
                for pt in price_tags:
                    pt_lower = pt.lower()
                    if "giảm" not in pt_lower and "tặng" not in pt_lower:
                        p_clean = ''.join(filter(str.isdigit, pt))
                        if p_clean.isdigit() and int(p_clean) > 0:
                            price_str = f"{int(p_clean):,}".replace(",", ".") # Ví dụ: 10.490.000
                            break
                # ---------------------------------------------------------------------

                a_tag = block.find('a')
                if a_tag and 'href' in a_tag.attrs:
                    href = a_tag['href']
                    link = BASE_URL + href if href.startswith('/') else href
                    
                    if name and price_str:
                        if link in seen_links:
                            continue
                        
                        seen_links.add(link)
                        
                        prefix = category_name.lower()
                        prod_id = f"{prefix}_{item_count:03d}"
                        
                        products.append({
                            "ID": prod_id,
                            "LoaiLinhKien": category_name,
                            "TenSP": name,
                            "Gia": price_str,
                            "Link": link
                        })
                        item_count += 1
            except Exception:
                continue

        browser.close()
    return products

# ==========================================
# CHẶNG 2: CHUI VÀO LINK (BẮT ĐÚNG GIÁ NEXT.JS VÀ BỎ QUA Ô RỖNG)
# ==========================================
def scrape_details(product):
    try:
        resp = requests.get(product['Link'], headers=HEADERS)
        if resp.status_code == 200:
            
            # --- [TUYỆT KỸ REGEX GIÁ THẬT] Căn cứ vào phát hiện của bạn ---
            # Bắt cụm "priceAndPromotions":{"price":10490000 để lấy giá thực 100%
            match_price = re.search(r'"priceAndPromotions":\{"price":(\d+)', resp.text)
            if match_price:
                price_int = int(match_price.group(1))
                if price_int > 0:
                    product['Gia'] = f"{price_int:,}".replace(",", ".")
            # --------------------------------------------------------------

            soup = BeautifulSoup(resp.text, 'html.parser')
            scripts = soup.find_all('script', type='application/ld+json')
            
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        data = [data]
                        
                    for item in data:
                        if 'additionalProperty' in item:
                            props = item['additionalProperty']
                            for p in props:
                                if isinstance(p, dict) and p.get('@type') == 'PropertyValue':
                                    key = str(p.get('name', '')).strip().replace(":", "")
                                    val_raw = str(p.get('value', ''))
                                    
                                    val_clean = re.sub(r'<br\s*/?>', ' - ', val_raw)
                                    val_clean = BeautifulSoup(val_clean, "html.parser").get_text(separator=" ", strip=True)
                                    
                                    # CHỈ GÁN DỮ LIỆU NẾU NÓ KHÔNG RỖNG. Nếu rỗng, bỏ qua (Pandas sẽ để ô trống hoàn toàn)
                                    if key and val_clean:
                                        product[key] = val_clean
                except Exception:
                    continue

    except Exception:
        pass 
        
    time.sleep(0.5) 
    return product

# ==========================================
# 3. THỰC THI CHIẾN DỊCH TỔNG LỰC
# ==========================================
def run_full_scraper():
    print("🚀 BẮT ĐẦU CHIẾN DỊCH CÀO DỮ LIỆU VER 8.3 (BẢN HOÀN MỸ - BẮT ĐÚNG GIÁ & DẤU CHẤM)...\n")
    
    for category, url in CATEGORIES.items():
        print(f"--- ĐANG QUÉT DANH MỤC: {category} ---")
        
        products = get_products_from_list(url, category, max_items=200)
        
        if not products:
            print(f"  [!] Không tìm thấy sản phẩm. Vui lòng kiểm tra lại URL.\n")
            continue
            
        print(f"-> THÀNH CÔNG: Đã tóm gọn chuẩn xác {len(products)} sản phẩm. Bắt đầu rút ruột JSON...")
        
        final_data = []
        for i, prod in enumerate(products):
            if (i+1) % 10 == 0 or (i+1) == len(products):
                print(f"   ... Đang xử lý {i+1}/{len(products)}...")
            final_data.append(scrape_details(prod))
            
        df = pd.DataFrame(final_data)
        csv_name = f"Data_{category}.csv"
        table_name = f"PC_{category}"
        
        df.to_csv(csv_name, index=False, encoding='utf-8-sig')
        print(f"  [v] Đã xuất file {csv_name} ({len(products)} dòng, {len(df.columns)} cột siêu sạch).")
        
        try:
            # FIX LỖI FONT SQL VÀ ÉP KIỂU
            string_cols = df.select_dtypes(include=['object']).columns
            dtype_mapping = {col: NVARCHAR for col in string_cols}
            
            df.to_sql(table_name, con=engine, if_exists='replace', index=False, dtype=dtype_mapping)
            print(f"  [v] Đã đẩy dữ liệu vào SQL Server, bảng: {table_name} (Đã fix lỗi font)")
        except Exception as e:
            print(f"  [x] Lỗi đẩy SQL: {e}")
            
        print("-" * 40)
        
    print("🎉 TẤT CẢ ĐÃ XONG! THU DỌN CHIẾN TRƯỜNG AN TOÀN.")

if __name__ == "__main__":
    run_full_scraper()