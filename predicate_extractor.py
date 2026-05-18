"""
predicate_extractor.py — Phân tích ngôn ngữ tự nhiên để extract thông tin người dùng
======================================================================================
Nhận dạng 3 thông tin từ câu chat:
  MucDich(x, value) — nhận diện từ khóa mục đích sử dụng
  NganSach(x, n)    — nhận diện số tiền ngân sách từ văn bản
  UuTien(x, value)  — nhận diện ưu tiên người dùng
"""

import re
from typing import Optional


# ══════════════════════════════════════════════════════════════════
# KEYWORD MAPS — ánh xạ từ khóa → domain value (KB vocab)
# ══════════════════════════════════════════════════════════════════

# Thứ tự quan trọng: pattern dài hơn đặt trước để tránh khớp sai
MUC_DICH_KEYWORDS: list[tuple[str, list[str]]] = [
    ("graphics",  ["đồ họa", "thiết kế", "render", "photoshop", "illustrator",
                   "3d", "blender", "autocad", "sketchup"]),
    ("editing",   ["dựng phim", "video editing", "premiere", "after effect",
                   "davinci", "dựng video", "edit video"]),
    ("streaming", ["streaming", "stream", "youtube", "obs", "livestream", "live stream"]),
    ("study",     ["lập trình", "code", "dev", "developer", "programming",
                   "python", "java", "học lập trình", "software"]),
    ("gaming",    ["gaming", "game", "chơi game", "chơi games", "esport",
                   "fps", "moba", "pubg", "valorant"]),
    ("office",    ["văn phòng", "office", "làm việc", "word", "excel",
                   "powerpoint", "zoom", "email", "họp online"]),
]

UU_TIEN_KEYWORDS: list[tuple[str, list[str]]] = [
    ("performance", ["ưu tiên gpu", "card mạnh", "gpu mạnh", "vga mạnh",
                     "card đồ họa mạnh", "performance", "hiệu năng cao", "fps cao"]),
    ("cpu",         ["ưu tiên cpu", "vi xử lý mạnh", "cpu mạnh", "nhân nhiều"]),
    ("value",       ["tiết kiệm", "rẻ nhất", "giá thấp", "giá rẻ", "giá tốt",
                     "budget", "ít tiền", "rẻ", "tiết kiệm nhất"]),
    ("quiet",       ["yên tĩnh", "im lặng", "không tiếng ồn", "quiet",
                     "silent", "ít ồn", "không ồn"]),
    ("compact",     ["nhỏ gọn", "mini", "compact", "itx", "nhỏ", "gọn nhẹ"]),
]

# Patterns nhận dạng số tiền VND — thứ tự từ cụ thể đến tổng quát
_MONEY_PATTERNS: list[tuple[re.Pattern, callable]] = [
    # "15.5 triệu" / "15,5 triệu" (thập phân với dấu chấm/phẩy)
    (re.compile(r'(\d+)[.,](\d+)\s*triệu', re.IGNORECASE),
     lambda m: int(float(f"{m.group(1)}.{m.group(2)}") * 1_000_000)),
    # "15tr" / "15 tr" / "15m" / "15 triệu" / "15 trieu" (không dấu)
    (re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:triệu|trieu|tr\b|m\b)', re.IGNORECASE),
     lambda m: int(float(m.group(1).replace(',', '.')) * 1_000_000)),
    # "15,000,000" / "15.000.000" (đã đủ định dạng VND)
    (re.compile(r'(\d{1,3}(?:[,.]\d{3})+)(?:\s*(?:đ|đồng|vnd))?', re.IGNORECASE),
     lambda m: int(re.sub(r'[,.]', '', m.group(1)))),
    # "15000000" (số thô ≥ 7 chữ số)
    (re.compile(r'\b(\d{7,})\b'),
     lambda m: int(m.group(1))),
]


# ══════════════════════════════════════════════════════════════════
# PREDICATE FUNCTIONS (theo UNIT_9)
# ══════════════════════════════════════════════════════════════════

def MucDich(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Predicate MucDich(x, value):
      ∃ keyword ∈ text → trả về (value, keyword_found)
      Dùng lượng từ tồn tại (∃): chỉ cần tìm thấy 1 keyword là đủ suy ra mục đích.

    Returns: (value, matched_keyword) hoặc (None, None)
    """
    t = text.lower()
    for value, keywords in MUC_DICH_KEYWORDS:
        for kw in keywords:
            if kw in t:
                return value, kw
    return None, None


def NganSach(text: str) -> tuple[Optional[int], Optional[str]]:
    """
    Predicate NganSach(x, n):
      ∀ pattern ∈ _MONEY_PATTERNS → thử lần lượt đến khi match → trả về n (VND)
      Dùng lượng từ với mọi (∀): duyệt toàn bộ pattern set.

    Sanity check: chỉ chấp nhận 1.000.000đ – 500.000.000đ.
    Returns: (vnd_value, matched_text) hoặc (None, None)
    """
    for pattern, converter in _MONEY_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                value = converter(m)
                if 1_000_000 <= value <= 500_000_000:
                    return value, m.group(0).strip()
            except (ValueError, IndexError):
                continue
    return None, None


def UuTien(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Predicate UuTien(x, value):
      ∃ keyword ưu tiên ∈ text → trả về (value, keyword_found)

    Returns: (value, matched_keyword) hoặc (None, None)
    """
    t = text.lower()
    for value, keywords in UU_TIEN_KEYWORDS:
        for kw in keywords:
            if kw in t:
                return value, kw
    return None, None


# ══════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════

def extract_predicates(text: str) -> dict:
    """
    Áp dụng 3 predicate functions lên câu chat tự nhiên.

    Returns dict facts cho Working Memory:
      {
        "muc_dich":  str | None,
        "ngan_sach": int | None,
        "uu_tien":   str,          # "" nếu không tìm thấy
        "_matched": {              # internal: keyword đã khớp (cho explain)
          "muc_dich":  str | None,
          "ngan_sach": str | None,
          "uu_tien":   str | None,
        }
      }
    """
    muc_dich_val, muc_dich_kw   = MucDich(text)
    ngan_sach_val, ngan_sach_kw = NganSach(text)
    uu_tien_val, uu_tien_kw     = UuTien(text)

    return {
        "muc_dich":  muc_dich_val,
        "ngan_sach": ngan_sach_val,
        "uu_tien":   uu_tien_val or "",
        "_matched": {
            "muc_dich":  muc_dich_kw,
            "ngan_sach": ngan_sach_kw,
            "uu_tien":   uu_tien_kw,
        },
    }


def explain_predicates(facts: dict) -> list[str]:
    """
    Trả về list string giải thích từng predicate đã extract,
    theo ký hiệu logic vị từ (UNIT_9).

    Ví dụ:
      ["MucDich(user, gaming) ← tìm thấy keyword 'gaming'",
       "NganSach(user, 15.000.000) ← extract từ '15 triệu'",
       "UuTien(user, performance) ← tìm thấy 'ưu tiên gpu'"]
    """
    matched = facts.get("_matched", {})
    result: list[str] = []

    if facts.get("muc_dich"):
        kw = matched.get("muc_dich", "?")
        result.append(f"MucDich(user, {facts['muc_dich']}) ← tìm thấy keyword '{kw}'")
    else:
        result.append("MucDich(user, ?) ← không tìm thấy keyword mục đích")

    if facts.get("ngan_sach"):
        kw  = matched.get("ngan_sach", "?")
        vnd = f"{facts['ngan_sach']:,}".replace(",", ".")
        result.append(f"NganSach(user, {vnd}) ← extract từ '{kw}'")
    else:
        result.append("NganSach(user, ?) ← không tìm thấy số tiền")

    uu = facts.get("uu_tien", "")
    if uu:
        kw = matched.get("uu_tien", "?")
        result.append(f"UuTien(user, {uu}) ← tìm thấy '{kw}'")
    else:
        result.append("UuTien(user, balanced) ← không có từ khóa ưu tiên → dùng mặc định")

    return result


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tests = [
        "Tôi muốn build máy gaming 15 triệu, ưu tiên GPU mạnh",
        "Cần máy văn phòng dưới 10tr dùng word excel zoom",
        "Build máy đồ họa 30,000,000 để dùng Photoshop và render 3D",
        "Máy lập trình Python khoảng 20tr tiết kiệm",
        "Muốn máy gaming 25tr fps cao",
        "PC stream youtube 18 triệu yên tĩnh",
        "Cần máy 15000000 để chơi game",
        "Máy dựng phim 40tr hiệu năng cao",
        "Máy tính nhỏ gọn ITX 12tr chơi game nhẹ",
        "Mình cần máy 8tr văn phòng đơn giản",
    ]

    print(f"{'─'*70}")
    for sent in tests:
        facts = extract_predicates(sent)
        exps  = explain_predicates(facts)
        print(f"Input  : {sent}")
        for e in exps:
            print(f"  {e}")
        print(f"  → muc_dich={facts['muc_dich']}, ngan_sach={facts['ngan_sach']}, uu_tien='{facts['uu_tien']}'")
        print(f"{'─'*70}")
