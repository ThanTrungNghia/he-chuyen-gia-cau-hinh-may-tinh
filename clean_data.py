"""
clean_data.py
-------------
Clean toàn bộ DATA_V1 và output ra DATA_V1_CLEANED.
Xử lý: format giá, chuẩn hóa type, map cột còn thiếu.

Chạy: python clean_data.py
      python clean_data.py --input DATA_V1 --output DATA_V1_CLEANED
"""

import os
import re
import argparse
import pandas as pd

RESET  = "\033[0m"; RED = "\033[91m"; GREEN = "\033[92m"
YELLOW = "\033[93m"; CYAN = "\033[96m"; BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def err(msg):  print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")

# ─── PARSE GIÁ TIỀN ─────────────────────────────────────────────────────────

def parse_price(val):
    """
    Chuyển các format giá về số nguyên (VND).
    Xử lý: '2.500.000', '2,500,000', '2500000đ', '2.500.000đ', '950', '950đ'...
    Nếu giá trị nhỏ hơn 1000 → có thể đang ở đơn vị nghìn đồng → nhân 1000.
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Bỏ ký tự không phải số, dấu chấm, dấu phẩy
    s = re.sub(r'[^\d.,]', '', s)
    if not s:
        return None
    # Nếu có cả dấu chấm và dấu phẩy → xác định format
    if '.' in s and ',' in s:
        # Format: 2,500,000.00 → dấu phẩy là ngàn, dấu chấm là thập phân
        if s.index(',') < s.index('.'):
            s = s.replace(',', '')
        else:
            # Format: 2.500.000,00 → dấu chấm là ngàn, dấu phẩy là thập phân
            s = s.replace('.', '').replace(',', '.')
    elif '.' in s:
        parts = s.split('.')
        # Nếu phần sau dấu chấm cuối có 3 chữ số → dấu chấm là ngàn (VN format)
        if len(parts[-1]) == 3:
            s = s.replace('.', '')
        # Nếu phần sau dấu chấm < 3 chữ số → dấu thập phân
        else:
            s = s.replace('.', '', len(parts) - 2)
    elif ',' in s:
        parts = s.split(',')
        if len(parts[-1]) == 3:
            s = s.replace(',', '')
        else:
            s = s.replace(',', '.')

    try:
        val_float = float(s)
        # Nếu giá quá nhỏ (< 1000) → có thể đơn vị là nghìn đồng
        if val_float < 1000:
            val_float *= 1000
        return int(val_float)
    except ValueError:
        return None

# ─── CHUẨN HÓA TỪNG FILE ────────────────────────────────────────────────────

def clean_cpu(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'TenSP': 'name', 'Gia': 'price', 'Thương hiệu': 'brand',
        'Socket': 'socket', 'Tiêu thụ điện năng': 'tdp',
        'Số nhân xử lý': 'cores', 'Số luồng của CPU': 'threads',
        'Tốc độ xử lý': 'base_clock', 'Cache': 'cache',
        'RAM hỗ trợ': 'supported_ddr', 'Thế hệ': 'generation',
        'Nhu cầu': 'use_case', 'LoaiLinhKien': 'category',
        'Link': 'link', 'Bảo hành': 'warranty',
    })
    df['price'] = df['price'].apply(parse_price)
    # Clean TDP: '65W' → 65
    if 'tdp' in df.columns:
        df['tdp'] = df['tdp'].astype(str).str.extract(r'(\d+)').astype(float)
    # Clean cores/threads
    for col in ['cores', 'threads']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # TDP: extract số watt từ cột 'Tiêu thụ điện năng' (vd: '65W' → 65)
    if 'tdp' in df.columns:
        df['tdp_w'] = df['tdp'].astype(str).str.extract(r'(\d+)').astype(float)
    else:
        # Fallback: estimate TDP từ tên (Ryzen 5 = ~65W, i9 = ~125W)
        def estimate_cpu_tdp(name):
            n = str(name).upper()
            if any(x in n for x in ['I9','RYZEN 9','THREADRIPPER']):  return 125.0
            if any(x in n for x in ['I7','RYZEN 7']):                  return 95.0
            if any(x in n for x in ['I5','RYZEN 5']):                  return 65.0
            if any(x in n for x in ['I3','RYZEN 3']):                  return 58.0
            return 65.0
        df['tdp_w'] = df['name'].apply(estimate_cpu_tdp)

    # CPU tier từ giá
    def cpu_tier(price):
        if pd.isna(price): return 'unknown'
        if price <= 3_000_000:  return 'budget'
        if price <= 8_000_000:  return 'mid'
        if price <= 15_000_000: return 'high'
        return 'extreme'
    df['cpu_tier'] = df['price'].apply(cpu_tier)

    # Chuẩn hóa socket: đảm bảo 'AM4', 'AM5', 'LGA1700', 'LGA1851'
    SOCKET_MAP = {
        '1700': 'LGA1700', '1200': 'LGA1200', '1851': 'LGA1851',
        'lga1700': 'LGA1700', 'lga1200': 'LGA1200', 'lga1851': 'LGA1851',
        'am4': 'AM4', 'am5': 'AM5',
    }
    if 'socket' in df.columns:
        df['socket'] = df['socket'].astype(str).str.strip().apply(
            lambda x: SOCKET_MAP.get(x.lower(), x.upper() if x.startswith('AM') else x)
        )

    return df

def clean_vga(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'TenSP': 'name', 'Gia': 'price', 'Thương hiệu': 'brand',
        'GPU': 'chipset', 'Bộ nhớ': 'vram', 'Tản nhiệt': 'cooling',
        'Nguồn đề xuất': 'tdp', 'Giao tiếp PCI': 'pci_interface',
        'Kích thước': 'size', 'Series chip đồ họa': 'series',
        'Nhu cầu': 'use_case', 'LoaiLinhKien': 'category',
        'Link': 'link', 'Bảo hành': 'warranty',
        'Nhà sản xuất chipset': 'chipset_brand',
        'Số lượng đơn vị xử lý': 'shaders',
        'Cổng kết nối': 'ports', 'Đầu cắp nguồn': 'power_connector',
    })
    df['price'] = df['price'].apply(parse_price)

    # VRAM: ưu tiên cột 'vram', fallback extract từ tên SP (vd: '8GB GDDR6')
    def extract_vram(row):
        if 'vram' in row and pd.notna(row['vram']):
            m = re.search(r'(\d+)', str(row['vram']))
            if m:
                return float(m.group(1))
        name = str(row.get('name', ''))
        m = re.search(r'(\d+)\s*GB', name, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 2 <= val <= 48:
                return float(val)
        return None
    df['vram_gb'] = df.apply(extract_vram, axis=1)

    # TDP proxy: extract số watt từ cột 'Nguồn đề xuất' (vd: '300W' → 300)
    if 'tdp' in df.columns:
        df['tdp_w'] = df['tdp'].astype(str).str.extract(r'(\d+)').astype(float)
    else:
        df['tdp_w'] = None

    # GPU tier từ tên chipset
    def infer_gpu_tier(name):
        n = str(name).upper()
        if any(x in n for x in ['4090','4080','4070 TI','5090','5080','5070 TI','7900 XTX','7900 XT']):
            return 'high'
        if any(x in n for x in ['4070','4060 TI','3080','3070','5070','5060 TI','7700','6800']):
            return 'mid-high'
        if any(x in n for x in ['4060','3060','5060','7600','6700','6600','3050','1660']):
            return 'mid'
        return 'budget'
    df['gpu_tier'] = df['name'].apply(infer_gpu_tier)

    return df

def clean_ram(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'TenSP': 'name', 'Gia': 'price', 'Thương hiệu': 'brand',
        'Loại hàng': 'type', 'Dung lượng': 'capacity',
        'Bus': 'speed', 'Timing': 'timing', 'Voltage': 'voltage',
        'Nhu cầu': 'use_case', 'ECC': 'ecc',
        'LoaiLinhKien': 'category', 'Link': 'link', 'Bảo hành': 'warranty',
        'Màu sắc': 'color', 'Đèn LED': 'led', 'Part-number': 'part_number',
    })
    df['price'] = df['price'].apply(parse_price)

    # Chuẩn hóa RAM type từ 'Hàng thông thường' → DDR4/DDR5
    # Dựa vào tên sản phẩm hoặc speed để suy ra
    def infer_ram_type(row):
        name = str(row.get('name', '')).upper()
        speed = str(row.get('speed', ''))
        current = str(row.get('type', '')).upper()
        if 'DDR5' in name or 'DDR5' in current:
            return 'DDR5'
        if 'DDR4' in name or 'DDR4' in current:
            return 'DDR4'
        if 'DDR3' in name or 'DDR3' in current:
            return 'DDR3'
        # Suy từ speed: DDR5 thường >= 4800MHz
        try:
            spd = int(re.sub(r'[^\d]', '', speed))
            if spd >= 4800:
                return 'DDR5'
            elif spd >= 1600:
                return 'DDR4'
        except (ValueError, TypeError):
            pass
        return 'DDR4'  # default

    df['type'] = df.apply(infer_ram_type, axis=1)

    # Capacity: '16GB' hoặc '2x8GB' → số GB
    def parse_capacity(val):
        s = str(val)
        # Kit: '2x8GB' → 16
        kit = re.search(r'(\d+)\s*x\s*(\d+)', s)
        if kit:
            return int(kit.group(1)) * int(kit.group(2))
        # Đơn: '16GB'
        single = re.search(r'(\d+)', s)
        if single:
            return int(single.group(1))
        return None

    # Capacity từ cột riêng hoặc extract từ tên SP
    # Tên thường có dạng: '(1 x 8GB)', '(2 x 16GB)', '32GB'
    def extract_ram_capacity(row):
        # Thử cột capacity trước
        cap_col = row.get('capacity', '')
        if pd.notna(cap_col) and str(cap_col).strip():
            val = parse_capacity(cap_col)
            if val and val > 0:
                return val
        # Fallback: extract từ tên SP
        name = str(row.get('name', ''))
        # Pattern: (2 x 16GB) hoặc (1 x 8GB)
        kit = re.search(r'\((\d+)\s*[xX]\s*(\d+)\s*GB\)', name)
        if kit:
            return int(kit.group(1)) * int(kit.group(2))
        # Pattern: 32GB standalone
        single = re.search(r'(\d+)\s*GB', name, re.IGNORECASE)
        if single:
            val = int(single.group(1))
            if 1 <= val <= 512:
                return val
        return None

    df['capacity_gb'] = df.apply(extract_ram_capacity, axis=1)

    # Speed: ưu tiên cột speed, fallback từ tên SP (vd: 'DDR4 3200MHz')
    def extract_ram_speed(row):
        speed_col = row.get('speed', '')
        if pd.notna(speed_col) and str(speed_col).strip():
            m = re.search(r'(\d+)', str(speed_col))
            if m and int(m.group(1)) >= 1600:
                return float(m.group(1))
        name = str(row.get('name', ''))
        m = re.search(r'(\d{4})\s*[Mm][Hh][Zz]', name)
        if m:
            return float(m.group(1))
        # Pattern: DDR4-3200 hoặc DDR5-6000
        m = re.search(r'DDR[345]-?(\d{4,5})', name, re.IGNORECASE)
        if m:
            return float(m.group(1))
        return None

    df['speed_mhz'] = df.apply(extract_ram_speed, axis=1)

    return df

def clean_mainboard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'TenSP': 'name', 'Gia': 'price', 'Thương hiệu': 'brand',
        'Socket': 'socket', 'Chipset': 'chipset',
        'Kích thước': 'form_factor',
        'Kiểu RAM hỗ trợ': 'supported_ddr',
        'Khe RAM tối đa': 'max_ram_slots',
        'Hỗ trợ bộ nhớ tối đa': 'max_ram_gb',
        'Bus RAM hỗ trợ': 'ram_speed',
        'Nhu cầu': 'use_case', 'LoaiLinhKien': 'category',
        'Link': 'link', 'Bảo hành': 'warranty',
        'Kiểu khe M.2 hỗ trợ': 'm2_slots',
        'Cổng xuất hình': 'display_ports',
        'Khe PCI': 'pci_slots', 'Số cổng USB': 'usb_ports',
        'LAN': 'lan', 'Kết nối không dây': 'wifi',
        'Âm thanh': 'audio', 'Đèn LED': 'led',
        'Multi-GPU': 'multi_gpu',
        'Series mainboard': 'series',
    })
    df['price'] = df['price'].apply(parse_price)

    # Chuẩn hóa form_factor: 'ATX', 'Micro ATX' → chuẩn
    FORM_FACTOR_MAP = {
        'atx': 'ATX', 'micro atx': 'mATX', 'microatx': 'mATX',
        'matx': 'mATX', 'mini itx': 'ITX', 'mini-itx': 'ITX',
        'itx': 'ITX', 'eatx': 'EATX', 'e-atx': 'EATX',
        'extended atx': 'EATX',
    }
    if 'form_factor' in df.columns:
        df['form_factor'] = df['form_factor'].fillna('').astype(str).str.strip().apply(
            lambda x: FORM_FACTOR_MAP.get(x.lower(), x) if x else ''
        )

    # supported_ddr: chuẩn hóa thành 'DDR4', 'DDR5', hoặc 'DDR4,DDR5'
    if 'supported_ddr' in df.columns:
        def normalize_ddr(val):
            s = str(val).upper().strip()
            has4 = 'DDR4' in s
            has5 = 'DDR5' in s
            if has4 and has5: return 'DDR4,DDR5'
            if has5:          return 'DDR5'
            if has4:          return 'DDR4'
            # Fallback: suy từ chipset/socket
            return 'DDR4'  # safe default
        df['supported_ddr'] = df['supported_ddr'].fillna('').apply(normalize_ddr)
    else:
        # Suy từ socket: AM5/LGA1700+ thường hỗ trợ DDR5
        def infer_ddr_from_socket(socket):
            s = str(socket).upper()
            if s in ['AM5', 'LGA1851']:     return 'DDR5'
            if s in ['AM4', 'LGA1700', 'LGA1200']: return 'DDR4'
            return 'DDR4'
        if 'socket' in df.columns:
            df['supported_ddr'] = df['socket'].apply(infer_ddr_from_socket)

    # Chuẩn hóa socket giống CPU
    SOCKET_MAP = {
        '1700': 'LGA1700', '1200': 'LGA1200', '1851': 'LGA1851',
        'lga1700': 'LGA1700', 'lga1200': 'LGA1200', 'lga1851': 'LGA1851',
        'am4': 'AM4', 'am5': 'AM5',
    }
    if 'socket' in df.columns:
        df['socket'] = df['socket'].fillna('').astype(str).str.strip().apply(
            lambda x: SOCKET_MAP.get(x.lower(), x) if x else ''
        )

    # MB tier từ giá
    def mb_tier(price):
        if pd.isna(price): return 'unknown'
        if price <= 2_000_000:  return 'budget'
        if price <= 5_000_000:  return 'mid'
        return 'high'
    df['mb_tier'] = df['price'].apply(mb_tier)

    return df

def clean_psu(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'TenSP': 'name', 'Gia': 'price', 'Thương hiệu': 'brand',
        'Công suất tối đa': 'wattage',
        'Hiệu suất': 'efficiency', 'Cáp rời': 'modular',
        'Chuẩn kích thước': 'form_factor',
        'Chứng nhận bảo vệ': 'protection_cert',
        'Loại hàng': 'tier', 'LoaiLinhKien': 'category',
        'Link': 'link', 'Bảo hành': 'warranty',
        'Tên': 'model_name', 'Part-number': 'part_number',
        'Series': 'series', 'Màu sắc': 'color',
        'Số cổng cắm': 'connectors', 'Quạt làm mát': 'fan',
        'Nguồn đầu vào': 'input_voltage',
        'Khối lượng': 'weight', 'Kích thước': 'size',
        'PFC': 'pfc', 'SKU gốc': 'sku',
        'Tình trạng': 'status', 'Đặc điểm': 'features',
        'Địa điểm có hàng': 'availability',
        'Đèn LED': 'led', 'Mô tả bảo hành': 'warranty_desc',
        'Hạn bảo hành': 'warranty_period',
        'Hệ số công suất': 'power_factor',
        'Nhiệt độ môi trường hoạt động lý tưởng của nguồn': 'operating_temp',
    })
    df['price'] = df['price'].apply(parse_price)

    # Wattage: ưu tiên cột 'wattage', fallback extract từ tên SP
    # Tên thường có dạng: 'Corsair RM850e - 850W', 'HX1500i - 1500W'
    def extract_wattage(row):
        # Thử cột wattage trước
        w_col = row.get('wattage', '')
        if pd.notna(w_col) and str(w_col).strip() not in ['', 'nan']:
            m = re.search(r'(\d+)', str(w_col))
            if m and int(m.group(1)) >= 100:
                return float(m.group(1))
        # Fallback: extract từ tên SP — pattern: '850W', '1000W', '1500W'
        name = str(row.get('name', ''))
        # Ưu tiên pattern XxxW (wattage thường 300-2000W)
        matches = re.findall(r'(\d{3,4})\s*W', name, re.IGNORECASE)
        for m in matches:
            val = int(m)
            if 300 <= val <= 2000:
                return float(val)
        return None

    df['wattage_w'] = df.apply(extract_wattage, axis=1)

    # PSU tier từ wattage
    def psu_tier(w):
        if pd.isna(w): return 'unknown'
        if w <= 550:   return 'budget'
        if w <= 750:   return 'mid'
        if w <= 1000:  return 'high'
        return 'extreme'
    df['psu_tier'] = df['wattage_w'].apply(psu_tier)

    # Bỏ duplicate theo name
    before = len(df)
    df = df.drop_duplicates(subset=['name'], keep='first')
    after = len(df)
    if before != after:
        print(f"    {YELLOW}⚠{RESET} Đã xóa {before - after} bản ghi PSU trùng tên")

    return df

def clean_storage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'TenSP': 'name', 'Gia': 'price', 'Thương hiệu': 'brand',
        'Kiểu ổ cứng': 'type', 'Dung lượng': 'capacity',
        'Kết nối': 'interface', 'Tốc độ đọc': 'read_speed',
        'Tốc độ ghi': 'write_speed', 'Kích thước': 'form_factor',
        'Hỗ trợ hệ điều hành': 'os_support',
        'LoaiLinhKien': 'category', 'Link': 'link', 'Bảo hành': 'warranty',
        'Màu sắc của ổ cứng': 'color', 'Tốc độ vòng quay': 'rpm',
        'Bộ nhớ NAND': 'nand_type', 'Cache': 'cache',
        'Mô tả bảo hành': 'warranty_desc',
        'Hỗ trợ hệ điều hành': 'os_support',
    })
    df['price'] = df['price'].apply(parse_price)

    # Chuẩn hóa Storage type
    TYPE_MAP = {
        'ổ cứng ssd': 'SSD', 'ssd': 'SSD', 'ssd m.2 nvme': 'NVMe',
        'nvme': 'NVMe', 'ổ cứng hdd': 'HDD', 'hdd': 'HDD',
        'di động ssd': 'SSD Portable', 'di động hdd': 'HDD Portable',
        'ổ gắn ngoài': 'External',
        'sata ssd': 'SSD', 'ssd nvme': 'NVMe',
    }
    if 'type' in df.columns:
        # fillna('') trước để tránh lỗi float.lower() khi có NaN
        df['type_clean'] = df['type'].fillna('').astype(str).str.strip().apply(
            lambda x: TYPE_MAP.get(x.lower(), x) if x else 'Unknown'
        )
        df['type'] = df['type_clean']
        df.drop(columns=['type_clean'], inplace=True)

    # Capacity: '1TB' → 1000, '512GB' → 512
    def parse_storage_capacity(val):
        s = str(val).upper()
        tb = re.search(r'([\d.]+)\s*TB', s)
        gb = re.search(r'(\d+)\s*GB', s)
        if tb:
            return int(float(tb.group(1)) * 1000)
        if gb:
            return int(gb.group(1))
        return None

    if 'capacity' in df.columns:
        df['capacity_gb'] = df['capacity'].apply(parse_storage_capacity)

    return df

def clean_case(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'TenSP': 'name', 'Gia': 'price', 'Thương hiệu': 'brand',
        'Tên của case': 'model_name', 'Loại case': 'form_factor',
        'Hỗ trợ mainboard': 'mb_support',
        'Nhu cầu': 'use_case', 'Series': 'series',
        'Màu sắc': 'color', 'Chất liệu': 'material',
        'Chất liệu nắp hông': 'side_panel',
        'LoaiLinhKien': 'category', 'Link': 'link', 'Bảo hành': 'warranty',
        'Số lượng ổ đĩa hỗ trợ': 'drive_bays',
        'Cổng kết nối': 'front_io',
        'Hỗ trợ tản nhiệt CPU cao': 'max_cooler_height',
        'Loại quạt hỗ trợ mặt trước': 'front_fan',
        'Loại quạt hỗ trợ phía trên': 'top_fan',
        'Loại quạt hỗ trợ phía sau': 'rear_fan',
        'Số slot PCI': 'pci_slots',
        'Part-number': 'part_number',
        'SKU gốc': 'sku', 'Tình trạng': 'status',
        'Đặc điểm': 'features',
        'Địa điểm có hàng': 'availability',
        'Mô tả bảo hành': 'warranty_desc',
        'Tính năng nổi bật': 'highlights',
        'Đèn LED': 'led', 'Loại hàng': 'tier',
        'Hạn bảo hành': 'warranty_period',
        'Số lượng quạt tặng kèm': 'included_fans',
    })
    df['price'] = df['price'].apply(parse_price)

    # Chuẩn hóa form_factor từ mb_support nếu form_factor trống
    CASE_FF_MAP = {
        'atx': 'ATX', 'micro atx': 'mATX', 'matx': 'mATX',
        'mini itx': 'ITX', 'itx': 'ITX', 'eatx': 'EATX',
        'full tower': 'Full Tower', 'mid tower': 'Mid Tower',
        'mini tower': 'Mini Tower',
    }
    if 'form_factor' in df.columns:
        df['form_factor'] = df['form_factor'].fillna('').astype(str).str.strip().apply(
            lambda x: CASE_FF_MAP.get(x.lower(), x) if x else ''
        )
    # Bỏ duplicate
    df = df.drop_duplicates(subset=['name'], keep='first')
    return df

def clean_cooler(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'TenSP': 'name', 'Gia': 'price', 'Thương hiệu': 'brand',
        'Dạng tản nhiệt': 'type',
        'Socket được hỗ trợ': 'socket_support',
        'Chiều cao (cm)': 'height_cm',
        'Độ ồn (dBA)': 'noise_dba',
        'Khối lượng (kg)': 'weight_kg',
        'Loại hàng': 'tier', 'LoaiLinhKien': 'category',
        'Link': 'link', 'Bảo hành': 'warranty',
        'Kích thước quạt (mm)': 'fan_size_mm',
        'Kích thước Radiator (cm)': 'radiator_size',
        'Số vòng quay của bơm (RPM)': 'pump_rpm',
        'Số vòng quay của quạt (RPM)': 'fan_rpm',
        'Lưu lượng không khí (CFM)': 'airflow_cfm',
        'Đèn LED': 'led', 'Màu sắc': 'color',
        'SKU gốc': 'sku', 'Tình trạng': 'status',
        'Đặc điểm': 'features',
        'Địa điểm có hàng': 'availability',
        'Tính năng nổi bật': 'highlights',
        'Hạn bảo hành': 'warranty_period',
        'Chất liệu tản nhiệt': 'material',
    })
    df['price'] = df['price'].apply(parse_price)

    # Height: '16.5' → float
    if 'height_cm' in df.columns:
        df['height_cm'] = pd.to_numeric(df['height_cm'], errors='coerce')

    return df

# ─── DISPATCH ────────────────────────────────────────────────────────────────

CLEANERS = {
    'Data_CPU':       clean_cpu,
    'Data_VGA':       clean_vga,
    'Data_RAM':       clean_ram,
    'Data_Mainboard': clean_mainboard,
    'Data_PSU':       clean_psu,
    'Data_Storage':   clean_storage,
    'Data_Case':      clean_case,
    'Data_Cooler':    clean_cooler,
}

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='.', help='Folder DATA_V1')
    parser.add_argument('--output', default='DATA_V1_CLEANED', help='Folder output')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  CLEAN DATA — HỆ CHUYÊN GIA TƯ VẤN CẤU HÌNH MÁY TÍNH{RESET}")
    print(f"{BOLD}{'='*58}{RESET}")
    print(f"  Input : {os.path.abspath(args.input)}")
    print(f"  Output: {os.path.abspath(args.output)}\n")

    total_ok = 0

    for file_key, cleaner in CLEANERS.items():
        filepath = os.path.join(args.input, file_key + '.csv')
        print(f"{BOLD}[{file_key}]{RESET}")

        if not os.path.exists(filepath):
            err(f"File không tồn tại: {filepath}")
            print()
            continue

        # Đọc file gốc
        for enc in ['utf-8-sig', 'utf-8', 'latin-1']:
            try:
                df_raw = pd.read_csv(filepath, encoding=enc)
                df_raw.columns = df_raw.columns.str.strip()
                break
            except UnicodeDecodeError:
                continue

        rows_before = len(df_raw)

        try:
            df_clean = cleaner(df_raw.copy())
        except Exception as e:
            err(f"Lỗi clean: {e}")
            print()
            continue

        # Bỏ rows không có name hoặc price
        df_clean = df_clean.dropna(subset=['name'])
        df_clean = df_clean[df_clean['price'].notna() & (df_clean['price'] > 0)]

        rows_after = len(df_clean)
        dropped = rows_before - rows_after

        # Kiểm tra price sau clean
        price_ok = df_clean['price'].notna().sum()
        price_min = int(df_clean['price'].min()) if price_ok > 0 else 0
        price_max = int(df_clean['price'].max()) if price_ok > 0 else 0

        # Lưu file
        out_path = os.path.join(args.output, file_key + '_cleaned.csv')
        df_clean.to_csv(out_path, index=False, encoding='utf-8-sig')

        ok(f"Saved: {file_key}_cleaned.csv")
        info(f"{rows_after} records hợp lệ (bỏ {dropped} rows thiếu/lỗi)")
        info(f"Giá: {price_min:,}đ – {price_max:,}đ")
        info(f"Cột sau clean: {list(df_clean.columns)}")
        print()
        total_ok += 1

    print(f"{BOLD}{'='*58}{RESET}")
    print(f"{BOLD}  KẾT QUẢ: {total_ok}/{len(CLEANERS)} file đã clean{RESET}")
    if total_ok == len(CLEANERS):
        print(f"  {GREEN}{BOLD}✓ Xong! Dùng DATA_V1_CLEANED để build thuật toán.{RESET}")
        print(f"  {CYAN}→ Bước tiếp: chạy validate_data_v2.py --path DATA_V1_CLEANED{RESET}")
    print(f"{BOLD}{'='*58}{RESET}\n")

if __name__ == '__main__':
    main()
