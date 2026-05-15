"""
csp_checker.py — CSP (Constraint Satisfaction Problem) + Forward Checking
==========================================================================
Nguồn lý thuyết:
  - Module6 — Tìm kiếm Thỏa mãn ràng buộc (slide CSP, FC, MRV)
  - lesson_5 — Constraint Satisfaction (slide 24: MRV; slide 27-31: FC)

Variables:
  cpu, mainboard, ram, vga, psu, storage, case, cooler

Domains: tập sản phẩm thỏa mãn (a) tier do FC chọn và (b) ngân sách
         được phân bổ cho linh kiện đó (cpu_budget, gpu_budget, ...).

Constraints (5 chính + 2 bonus):
  C1  socket    : cpu.socket == mainboard.socket
  C2  ddr_type  : ram.type    == mainboard.supported_ddr
  C3  power     : cpu.tdp + vga.tdp <= psu.wattage * 0.8        (headroom 20%)
  C4  form_fact : case_form_factor accommodates mainboard_ff
  C5  budget    : sum(prices) <= ngan_sach
  C6  cooler-cpu: cpu.socket  ∈ cooler.socket_support           (bonus)
  C7  cooler-tdp: cpu.tdp_w  <= cooler_tdp_max(cooler.type)     (bonus)

Forward Checking (lesson_5 slide 27-31):
  "Keep track of remaining legal values for unassigned variables.
   Terminate search when any variable has no legal values."

MRV — Most Constrained Variable (lesson_5 slide 24):
  Gán biến có nhiều ràng buộc nhất trước → giảm bùng nổ tổ hợp.
  Thứ tự: mainboard → cpu → ram → vga → psu → case → cooler → storage
"""

import os
import re
import unicodedata
import pandas as pd
from copy import copy

from knowledge_base import WorkingMemory, COOLER_TDP_SUPPORT
from forward_chaining import TIER_MAP_CPU, TIER_MAP_GPU


DATA_DIR = "DATA"


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

# Map Case form_factor (CSV) → Mainboard form_factor (chuẩn ATX/mATX/ITX)
# Vì case CSV dùng "Tower" naming chứ không dùng ATX/mATX/ITX trực tiếp.
# Quy ước: case lớn chứa được mọi mainboard nhỏ hơn (xem ff_compatible).
CASE_FF_MAP: dict[str, str] = {
    "Super Tower":  "EATX",   # > ATX
    "Full Tower":   "ATX",
    "Mid Tower":    "ATX",
    "Micro Tower":  "mATX",
    "Mini Tower":   "ITX",
}

# Map Mainboard form_factor (CSV "Kích thước") → chuẩn ATX/mATX/ITX
MB_FF_MAP: dict[str, str] = {
    "ATX":        "ATX",
    "Micro-ATX":  "mATX",
    "MicroATX":   "mATX",
    "mATX":       "mATX",
    "Mini-ITX":   "ITX",
    "ITX":        "ITX",
    "E-ATX":      "EATX",
    "EATX":       "EATX",
}

# Map Cooler type (tiếng Việt) → tier key trong COOLER_TDP_SUPPORT (KB)
COOLER_TYPE_MAP: dict[str, str] = {
    "Tản khí":          "air-mid",      # default mid; nâng/hạ tùy giá nếu cần
    "Tản nước AIO":     "aio-240",
    "Tản nước":         "aio-240",
    "Quạt":             "stock",
    "Tản nhiệt Laptop": "stock",        # ngoài scope, exclude trong filter
}


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Một số CSV có tên cột tiếng Việt ở dạng NFD (combining chars),
    còn source code Python dùng NFC. Normalize NFC để dùng tên cột
    trực tiếp được trong code.
    """
    df.columns = [unicodedata.normalize("NFC", c) for c in df.columns]
    return df


def _normalize_str_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Normalize NFC cho TẤT CẢ giá trị string trong 1 cột (vd cooler.type)."""
    if col in df.columns:
        df[col] = df[col].apply(
            lambda v: unicodedata.normalize("NFC", str(v)) if pd.notna(v) else v
        )
    return df


def load_data() -> dict[str, list[dict]]:
    """
    Đọc 8 CSV trong DATA/, normalize columns, rename cột tiếng Việt
    sang English snake_case, trả về dict cho CSP.
    """
    files = {
        "cpu":       "Data_CPU.csv",
        "vga":       "Data_VGA.csv",
        "ram":       "Data_RAM.csv",
        "mainboard": "Data_Mainboard.csv",
        "psu":       "Data_PSU.csv",
        "storage":   "Data_Storage.csv",
        "case":      "Data_Case.csv",
        "cooler":    "Data_Cooler.csv",
    }

    out: dict[str, list[dict]] = {}
    for key, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = _normalize_cols(df)

        # Rename cột tiếng Việt → English.
        # CHỈ rename khi (a) cột nguồn tồn tại VÀ (b) cột đích chưa tồn tại
        # — tránh duplicate (Case/PSU đã có sẵn form_factor từ clean_data.py).
        rename_map = {
            "Kích thước":       "form_factor",
            "Số nhân xử lý":    "cores",
            "Dung lượng":       "capacity_raw",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

        # Normalize NFC cho các cột categorical hay so sánh string
        # (CSV có thể chứa NFD — e.g. "Tản nước" 'án vs á)
        for col in ("type", "form_factor", "supported_ddr", "socket"):
            df = _normalize_str_col(df, col)

        # Mainboard: map form_factor sang chuẩn ATX/mATX/ITX
        if key == "mainboard" and "form_factor" in df.columns:
            df["form_factor"] = df["form_factor"].map(
                lambda x: MB_FF_MAP.get(str(x).strip(), "ATX") if pd.notna(x) else "ATX"
            )

        # Case: map "Tower" → ATX/mATX/ITX/EATX
        if key == "case" and "form_factor" in df.columns:
            df["form_factor"] = df["form_factor"].map(
                lambda x: CASE_FF_MAP.get(str(x).strip(), "ATX") if pd.notna(x) else "ATX"
            )

        # CPU: parse cores từ "Số nhân xử lý" nếu có (vd "6 nhân" → 6)
        if key == "cpu" and "cores" in df.columns:
            df["cores"] = df["cores"].apply(
                lambda x: int(re.search(r"\d+", str(x)).group()) if pd.notna(x) and re.search(r"\d+", str(x)) else 4
            )

        out[key] = df.to_dict("records")

    return out


# ══════════════════════════════════════════════════════════════════
# DOMAIN FILTERING — lọc tier + budget + ràng buộc tối thiểu
# ══════════════════════════════════════════════════════════════════

def _f(x, default=0.0) -> float:
    """Safe-cast NaN / None → default."""
    try:
        v = float(x)
        return v if v == v else default     # NaN check
    except (TypeError, ValueError):
        return default


def filter_domains(data: dict[str, list[dict]], wm: WorkingMemory) -> dict[str, list[dict]]:
    """
    Lọc domain mỗi biến theo:
      - tier (FC đã chọn, map sang CSV tier)
      - budget (FC đã phân bổ)
      - min specs (ram_min_gb, vram_min_gb)
    """
    cpu_csv_tier = TIER_MAP_CPU.get(wm.cpu_tier)
    gpu_csv_tier = TIER_MAP_GPU.get(wm.gpu_tier)

    # ── CPU ───────────────────────────────────────────────────────
    cpu_dom = [
        r for r in data["cpu"]
        if 0 < _f(r.get("price")) <= wm.cpu_budget
        and (cpu_csv_tier is None or r.get("cpu_tier") == cpu_csv_tier)
    ]

    # ── VGA ───────────────────────────────────────────────────────
    if wm.gpu_tier == "none":
        # Không cần GPU rời — dùng iGPU placeholder
        vga_dom = [{
            "name": "iGPU (tích hợp trong CPU)",
            "price": 0, "tdp_w": 0.0, "vram_gb": 0.0,
            "chipset": "Integrated", "gpu_tier": None, "link": "",
        }]
    else:
        vga_dom = [
            r for r in data["vga"]
            if 0 < _f(r.get("price")) <= wm.gpu_budget
            and (gpu_csv_tier is None or r.get("gpu_tier") == gpu_csv_tier)
            and _f(r.get("vram_gb")) >= wm.vram_min_gb
        ]

    # ── RAM ───────────────────────────────────────────────────────
    ram_dom = [
        r for r in data["ram"]
        if 0 < _f(r.get("price")) <= wm.ram_budget
        and r.get("type") == wm.ram_type
        and _f(r.get("capacity_gb")) >= wm.ram_min_gb
    ]

    # ── Mainboard ─────────────────────────────────────────────────
    mb_dom = [
        r for r in data["mainboard"]
        if 0 < _f(r.get("price")) <= wm.mb_budget
        and r.get("supported_ddr") == wm.ram_type        # DDR4/DDR5 phải khớp RAM
    ]

    # ── PSU ───────────────────────────────────────────────────────
    psu_dom = [
        r for r in data["psu"]
        if 0 < _f(r.get("price")) <= wm.psu_budget
        and _f(r.get("wattage_w")) >= wm.psu_wattage_min
    ]

    # ── Storage ───────────────────────────────────────────────────
    # KB output: ssd-only/nvme-only/ssd+hdd/nvme+hdd
    # CSV chỉ có SSD/HDD/External → filter ưu tiên SSD (phù hợp tất cả)
    sto_dom = [
        r for r in data["storage"]
        if 0 < _f(r.get("price")) <= wm.storage_budget
        and r.get("type") == "SSD"
        and _f(r.get("capacity_gb")) >= wm.storage_min_gb
    ]
    # Fallback: nới capacity nếu không có SSD đủ lớn
    if not sto_dom:
        sto_dom = [
            r for r in data["storage"]
            if 0 < _f(r.get("price")) <= wm.storage_budget
            and r.get("type") in ("SSD", "HDD")
        ]

    # ── Case ──────────────────────────────────────────────────────
    case_dom = [
        r for r in data["case"]
        if 0 < _f(r.get("price")) <= wm.ngan_sach * 0.06
    ]

    # ── Cooler ────────────────────────────────────────────────────
    # Loại "Tản nhiệt Laptop" và các loại không relevant
    cooler_dom = [
        r for r in data["cooler"]
        if 0 < _f(r.get("price")) <= wm.ngan_sach * 0.04
        and r.get("type") in ("Tản khí", "Tản nước AIO", "Tản nước")
    ]

    return {
        "cpu":       cpu_dom,
        "mainboard": mb_dom,
        "ram":       ram_dom,
        "vga":       vga_dom,
        "psu":       psu_dom,
        "case":      case_dom,
        "cooler":    cooler_dom,
        "storage":   sto_dom,
    }


# ══════════════════════════════════════════════════════════════════
# CONSTRAINTS
# ══════════════════════════════════════════════════════════════════

# Form factor compatibility: case lớn chứa được mainboard nhỏ hơn
FF_ORDER: dict[str, int] = {"EATX": 4, "ATX": 3, "mATX": 2, "ITX": 1}


def ff_compatible(case_ff: str, mb_ff: str) -> bool:
    return FF_ORDER.get(case_ff, 0) >= FF_ORDER.get(mb_ff, 0)


def _cooler_supports_socket(cooler_socket_support: str, cpu_socket: str) -> bool:
    """Kiểm tra cpu socket có nằm trong chuỗi socket_support của cooler."""
    s = str(cooler_socket_support or "").upper()
    target = str(cpu_socket or "").upper()
    if not target:
        return True
    # Nới: nếu CSV không có socket_support thì bỏ qua check
    if not s.strip() or s.strip() == "NAN":
        return True
    return target in s


def _check_pair(a: dict, var1: str, var2: str) -> bool:
    """Check tất cả binary constraint giữa var1 và var2 (đã có trong assignment a)."""
    pair = {var1, var2}

    # C1 socket
    if pair == {"cpu", "mainboard"}:
        return a["cpu"].get("socket") == a["mainboard"].get("socket")

    # C2 ddr_type
    if pair == {"ram", "mainboard"}:
        return a["ram"].get("type") == a["mainboard"].get("supported_ddr")

    # C4 form_factor
    if pair == {"case", "mainboard"}:
        return ff_compatible(a["case"].get("form_factor", "ATX"),
                             a["mainboard"].get("form_factor", "ATX"))

    # C6 cooler-cpu socket
    if pair == {"cooler", "cpu"}:
        return _cooler_supports_socket(a["cooler"].get("socket_support", ""),
                                        a["cpu"].get("socket", ""))

    return True


def _check_partial(a: dict) -> bool:
    """
    Check tất cả constraint áp dụng được trên assignment hiện tại
    (bỏ qua constraint nếu chưa đủ biến).
    """
    if "cpu" in a and "mainboard" in a:
        if a["cpu"].get("socket") != a["mainboard"].get("socket"):
            return False
    if "ram" in a and "mainboard" in a:
        if a["ram"].get("type") != a["mainboard"].get("supported_ddr"):
            return False
    if "case" in a and "mainboard" in a:
        if not ff_compatible(a["case"].get("form_factor", "ATX"),
                             a["mainboard"].get("form_factor", "ATX")):
            return False
    if "cooler" in a and "cpu" in a:
        if not _cooler_supports_socket(a["cooler"].get("socket_support", ""),
                                        a["cpu"].get("socket", "")):
            return False
        # C7 cooler TDP — bỏ qua nếu CSV không cung cấp tdp_w cho CPU
        cpu_tdp = _f(a["cpu"].get("tdp_w"), 65)
        cooler_key = COOLER_TYPE_MAP.get(str(a["cooler"].get("type", "")).strip(), "air-mid")
        max_tdp = COOLER_TDP_SUPPORT.get(cooler_key, 200)
        if cpu_tdp > max_tdp:
            return False
    if "cpu" in a and "vga" in a and "psu" in a:
        cpu_tdp = _f(a["cpu"].get("tdp_w"), 65)
        vga_tdp = _f(a["vga"].get("tdp_w"), 0)
        psu_w   = _f(a["psu"].get("wattage_w"), 500)
        if (cpu_tdp + vga_tdp) > psu_w * 0.8:
            return False
    return True


# ══════════════════════════════════════════════════════════════════
# CSP + FORWARD CHECKING
# ══════════════════════════════════════════════════════════════════

# MRV order: mainboard nhiều constraint nhất → gán đầu tiên
ASSIGNMENT_ORDER: list[str] = [
    "mainboard",  # bị ràng buộc bởi cpu (socket), ram (ddr), case (form_factor)
    "cpu",        # phải khớp socket với mainboard
    "ram",        # phải khớp ddr với mainboard
    "vga",        # ảnh hưởng PSU (TDP)
    "psu",        # phụ thuộc tổng TDP
    "case",       # phụ thuộc form_factor mainboard
    "cooler",     # phụ thuộc socket cpu + cpu TDP
    "storage",    # độc lập nhất
]


def _forward_check(assignment: dict, domains: dict, last_var: str) -> dict | None:
    """
    Sau khi gán last_var, thu hẹp domain các biến chưa gán
    bằng cách loại bỏ giá trị vi phạm constraint với last_var.
    Trả về dict domain mới, hoặc None nếu có domain rỗng.
    """
    new_domains = {k: list(v) for k, v in domains.items()}

    for var in ASSIGNMENT_ORDER:
        if var in assignment or var == last_var:
            continue
        # Lọc domain[var] giữ lại các value không vi phạm khi kết hợp với last_var
        filtered = []
        for value in new_domains.get(var, []):
            test = {**assignment, var: value}
            if _check_partial(test):
                filtered.append(value)
        if not filtered:
            return None  # domain rỗng → terminate early
        new_domains[var] = filtered

    return new_domains


def csp_with_forward_checking(
    domains: dict[str, list[dict]],
    budget: float,
    max_results: int = 50,
) -> list[dict]:
    """
    Backtracking + Forward Checking + MRV-static-order.

    Args:
      domains      — kết quả filter_domains()
      budget       — wm.ngan_sach (cho C5)
      max_results  — cap để tránh bùng nổ tổ hợp

    Returns: list các assignment hợp lệ, mỗi assignment có thêm key "total".
    """
    results: list[dict] = []

    def backtrack(assignment: dict, remaining: dict):
        if len(results) >= max_results:
            return

        # Đã gán đủ biến → check C5 (total budget)
        if len(assignment) == len(ASSIGNMENT_ORDER):
            total = sum(_f(assignment[k].get("price")) for k in ASSIGNMENT_ORDER)
            if total <= budget:
                cfg = {k: assignment[k] for k in ASSIGNMENT_ORDER}
                cfg["total"] = total
                results.append(cfg)
            return

        # Chọn biến tiếp theo theo thứ tự MRV-static
        var = ASSIGNMENT_ORDER[len(assignment)]

        for value in remaining.get(var, []):
            test = {**assignment, var: value}
            if not _check_partial(test):
                continue

            # Forward Checking: thu hẹp domain các biến còn lại
            new_dom = _forward_check(test, remaining, var)
            if new_dom is None:
                continue  # backtrack sớm

            backtrack(test, new_dom)
            if len(results) >= max_results:
                return

    backtrack({}, domains)
    return results


# ══════════════════════════════════════════════════════════════════
# FALLBACK — nới ngân sách per-component khi domain rỗng
# ══════════════════════════════════════════════════════════════════

def filter_with_fallback(
    data: dict[str, list[dict]],
    wm: WorkingMemory,
    max_attempts: int = 3,
    scale_step: float = 1.5,
) -> tuple[dict[str, list[dict]], list[str]]:
    """
    Lọc domain. Nếu có biến nào domain rỗng, nới ngân sách CHO ĐÚNG biến đó
    (×1.5 mỗi lần) tới max_attempts lần. Trả về (domains, log) trong đó
    log là chuỗi mô tả các lần nới.
    """
    log: list[str] = []
    domains = filter_domains(data, wm)
    empty = [k for k, v in domains.items() if not v]

    attempt = 0
    while empty and attempt < max_attempts:
        attempt += 1
        log.append(f"[Attempt {attempt}] domain rỗng: {empty} → nới ×{scale_step}")
        for var in empty:
            if   var == "cpu":       wm.cpu_budget     *= scale_step
            elif var == "vga":       wm.gpu_budget     *= scale_step
            elif var == "ram":       wm.ram_budget     *= scale_step
            elif var == "mainboard": wm.mb_budget      *= scale_step
            elif var == "psu":       wm.psu_budget     *= scale_step
            elif var == "storage":   wm.storage_budget *= scale_step
        # case + cooler dùng % của ngan_sach → nới ngan_sach để không phá tier %
        if "case" in empty or "cooler" in empty:
            wm.ngan_sach = int(wm.ngan_sach * 1.2)
        domains = filter_domains(data, wm)
        empty = [k for k, v in domains.items() if not v]

    if empty:
        log.append(f"[Bỏ cuộc] vẫn còn rỗng sau {max_attempts} lần: {empty}")
    return domains, log


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from forward_chaining import run_forward_chaining

    test_cases = [
        ("Gaming 25tr",      25_000_000, "gaming",   ""),
        ("Office 12tr",      12_000_000, "office",   ""),
        ("Graphics 30tr",    30_000_000, "graphics", ""),
        ("Study 15tr value", 15_000_000, "study",    "value"),
    ]

    print("Loading data...")
    data = load_data()
    for k, v in data.items():
        print(f"  {k:<10}: {len(v):>4} records")

    for label, ngan_sach, muc_dich, uu_tien in test_cases:
        print(f"\n{'═'*68}")
        print(f"TEST: {label}")
        print(f"{'─'*68}")

        wm = WorkingMemory(ngan_sach=ngan_sach, muc_dich=muc_dich, uu_tien=uu_tien)
        wm, fired = run_forward_chaining(wm)
        print(f"  FC fired       : {fired}")
        print(f"  CPU/GPU/RAM    : {wm.cpu_tier}/{wm.gpu_tier}/{wm.ram_capacity}GB {wm.ram_type}")
        print(f"  Budgets        : CPU={wm.cpu_budget:,.0f}  GPU={wm.gpu_budget:,.0f}  RAM={wm.ram_budget:,.0f}")

        domains, fallback_log = filter_with_fallback(data, wm)
        print(f"  Domains        : " + " | ".join(f"{k}={len(v)}" for k, v in domains.items()))
        for line in fallback_log:
            print(f"    {line}")

        configs = csp_with_forward_checking(domains, wm.ngan_sach, max_results=50)
        print(f"  ✅ Valid configs: {len(configs)}")

        if configs:
            best = min(configs, key=lambda c: c["total"])
            print(f"  Cheapest valid : {best['total']:,.0f}đ")
            print(f"    CPU       : {best['cpu']['name'][:50]}")
            print(f"    Mainboard : {best['mainboard']['name'][:50]}")
            print(f"    RAM       : {best['ram']['name'][:50]}")
            print(f"    VGA       : {best['vga']['name'][:50]}")
            print(f"    PSU       : {best['psu']['name'][:50]}")
            print(f"    Case      : {best['case']['name'][:50]}")
            print(f"    Cooler    : {best['cooler']['name'][:50]}")
            print(f"    Storage   : {best['storage']['name'][:50]}")
