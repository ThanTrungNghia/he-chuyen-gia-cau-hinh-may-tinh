"""
astar_selector.py — A* Search để chọn cấu hình tối ưu
=======================================================
Ánh xạ A* vào bài toán chọn cấu hình:
  - Mỗi cấu hình hợp lệ (do CSP trả về) là một goal-state candidate.
  - g(config) = chi phí thực tế đã tích lũy = total_price / ngan_sach   ∈ [0, 1]
  - h(config) = ước tính chi phí còn lại = 1 - performance_score(config) ∈ [0, 1]
  - f(config) = g + h (càng nhỏ càng tốt — vừa rẻ vừa mạnh)

Heuristic h là admissible vì performance_score ∈ [0,1] nên h ≥ 0 và h ≤ 1.
A* trả về cấu hình có f tối thiểu trong tập hợp lệ từ CSP.

Trọng số WEIGHTS theo mục đích sử dụng — phản ánh ưu tiên thực tế:
  ví dụ gaming ưu tiên GPU 45%, office ưu tiên CPU 40% và RAM 35%.
"""

import re
import heapq
from knowledge_base import WorkingMemory


def _extract_from_name(name: str, kind: str) -> float:
    """Extract numeric specs từ tên sản phẩm khi cột CSV bị NaN/0."""
    text = str(name).upper()
    if kind == "capacity_gb":
        m = re.search(r"(\d+(?:\.\d+)?)\s*TB", text)
        if m: return float(m.group(1)) * 1000
        m = re.search(r"(\d+(?:\.\d+)?)\s*GB", text)
        if m: return float(m.group(1))
    elif kind == "speed_mhz":
        m = re.search(r"(\d{4,5})\s*MHZ", text)
        if m: return float(m.group(1))
        m = re.search(r"(\d{4,5})\s*MT/S", text)
        if m: return float(m.group(1))
    elif kind == "cores":
        for pat in (r"(\d+)\s*(?:NHÂN|CORE|COR)", r"(\d+)-CORE", r"(\d+)\s*X\s*\d+"):
            m = re.search(pat, text)
            if m: return float(m.group(1))
    elif kind == "vram_gb":
        # Match "8GB GDDR" or "8G GDDR" before GDDR keyword
        m = re.search(r"(\d+)\s*G\s*(?:B\s*)?GDDR", text)
        if m: return float(m.group(1))
    return 0.0


# ══════════════════════════════════════════════════════════════════
# WEIGHTS — phân bổ trọng số performance theo muc_dich
# Mỗi hàng tổng = 1.0
# Vocab khớp với KB (English): office/gaming/graphics/streaming/study/editing
# ══════════════════════════════════════════════════════════════════
WEIGHTS: dict[str, dict[str, float]] = {
    "gaming":    {"cpu": 0.25, "gpu": 0.45, "ram": 0.15, "storage": 0.15},
    "graphics":  {"cpu": 0.30, "gpu": 0.40, "ram": 0.20, "storage": 0.10},
    "editing":   {"cpu": 0.30, "gpu": 0.30, "ram": 0.25, "storage": 0.15},
    "streaming": {"cpu": 0.40, "gpu": 0.30, "ram": 0.20, "storage": 0.10},
    "office":    {"cpu": 0.40, "gpu": 0.05, "ram": 0.35, "storage": 0.20},
    "study":     {"cpu": 0.35, "gpu": 0.10, "ram": 0.35, "storage": 0.20},
}


def _f(x, default=0.0) -> float:
    """Safe float cast — NaN / None → default."""
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════════
# PERFORMANCE SCORE — ∈ [0, 1]
# ══════════════════════════════════════════════════════════════════
def performance_score(config: dict, wm: WorkingMemory) -> float:
    """
    Tính điểm hiệu năng tổng hợp của 1 cấu hình ∈ [0, 1].
    Bao gồm yếu tố phụ để phân biệt các cấu hình cùng tier:
      - price_efficiency: cấu hình rẻ hơn trong cùng tier được điểm cao hơn
      - psu_quality: 80+ Gold/Platinum thêm điểm
      - storage_capacity: 512GB/1TB/2TB được normalize
      - ram_speed: tốc độ bus RAM (normalize max 6000MHz)
    """
    w = WEIGHTS.get(wm.muc_dich, WEIGHTS["office"])

    cpu     = config["cpu"]
    cpu_name = str(cpu.get("name", ""))
    cores_raw = _f(cpu.get("cores"), 0)
    cores = cores_raw if cores_raw >= 2 else _extract_from_name(cpu_name, "cores") or 4
    cpu_score = min(cores / 16.0, 1.0)

    vga  = config.get("vga", {})
    vga_name = str(vga.get("name", ""))
    vram_raw = _f(vga.get("vram_gb"), 0)
    vram = vram_raw if vram_raw > 0 else _extract_from_name(vga_name, "vram_gb")
    gpu_score = min(vram / 24.0, 1.0)

    ram      = config["ram"]
    ram_name = str(ram.get("name", ""))
    ram_gb_raw = _f(ram.get("capacity_gb"), 0)
    ram_gb  = ram_gb_raw if ram_gb_raw >= 4 else _extract_from_name(ram_name, "capacity_gb") or 8
    ram_base = min(ram_gb / 64.0, 1.0)
    spd_raw = _f(ram.get("speed_mhz"), 0)
    ram_speed = spd_raw if spd_raw >= 1600 else _extract_from_name(ram_name, "speed_mhz") or 3200
    ram_speed_score = min(ram_speed / 6000.0, 1.0)
    ram_score = (ram_base * 0.7 + ram_speed_score * 0.3)

    sto      = config["storage"]
    sto_name = str(sto.get("name", ""))
    storage_type = str(sto.get("type", "")).strip()
    # Detect NVMe từ tên sản phẩm nếu type không ghi rõ
    if storage_type == "SSD" and any(kw in sto_name.upper() for kw in ("NVME", "M.2", "PCIE")):
        storage_type = "NVMe"
    storage_type_bonus = {"NVMe": 1.0, "SSD": 0.7, "HDD": 0.3}
    sto_type_score = storage_type_bonus.get(storage_type, 0.5)
    sto_cap_raw = _f(sto.get("capacity_gb"), 0)
    sto_cap = sto_cap_raw if sto_cap_raw >= 64 else _extract_from_name(sto_name, "capacity_gb") or 256
    sto_cap_score = 1.0 if sto_cap >= 2000 else (0.7 if sto_cap >= 1000 else (0.5 if sto_cap >= 512 else 0.3))
    storage_score = (sto_type_score * 0.6 + sto_cap_score * 0.4)

    base = (w["cpu"] * cpu_score + w["gpu"] * gpu_score
            + w["ram"] * ram_score + w["storage"] * storage_score)

    # PSU quality bonus
    efficiency = str(config.get("psu", {}).get("efficiency", "")).lower()
    psu_bonus = 0.03 if "platinum" in efficiency else (0.02 if "gold" in efficiency else 0.0)

    # Price efficiency bonus: cấu hình rẻ hơn trong cùng tier được điểm cao hơn
    total_price = _f(config.get("total"), 0)
    price_eff = max(0.0, 1.0 - total_price / wm.ngan_sach) if wm.ngan_sach > 0 else 0.0

    return min(base + psu_bonus + 0.08 * price_eff, 1.0)


# ══════════════════════════════════════════════════════════════════
# g(n) and h(n)
# ══════════════════════════════════════════════════════════════════
def g(config: dict, wm: WorkingMemory) -> float:
    """g(n) = chi phí thực tế = total_price / ngan_sach ∈ [0, 1]."""
    if wm.ngan_sach <= 0:
        return 0.0
    return min(_f(config.get("total")) / wm.ngan_sach, 1.5)   # cap 1.5 cho overshoot


def h(config: dict, wm: WorkingMemory) -> float:
    """
    h(n) = ước tính chi phí còn lại = 1 - performance_score ∈ [0, 1].
    Admissible vì h ≤ h* = 0 khi đạt cấu hình hoàn hảo (perf=1).
    """
    return 1.0 - performance_score(config, wm)


def f_score(config: dict, wm: WorkingMemory) -> float:
    """f(n) = g(n) + h(n) — càng nhỏ càng tốt."""
    return g(config, wm) + h(config, wm)


# ══════════════════════════════════════════════════════════════════
# A* SELECTOR
# ══════════════════════════════════════════════════════════════════
def astar_select(
    valid_configs: list[dict],
    wm: WorkingMemory,
) -> tuple[dict, list[dict], float]:
    """
    Chọn cấu hình tối ưu từ valid_configs do CSP trả về.

    Args:
      valid_configs — list dict, mỗi dict chứa cpu/mainboard/.../total
      wm            — WorkingMemory để lấy muc_dich và ngan_sach

    Returns:
      (best, top3, best_f)
        best   : cấu hình tốt nhất theo f-score
        top3   : 3 cấu hình tốt nhất (best ở index 0)
        best_f : giá trị f(best)

    Vì h admissible, cấu hình trả về có f tối thiểu = lựa chọn tối ưu.
    """
    if not valid_configs:
        raise ValueError("astar_select: valid_configs rỗng")

    # Priority queue (min-heap): (f_score, tie_breaker_idx, config)
    # idx làm tie-breaker để tránh dict comparison error khi f bằng nhau
    heap: list[tuple[float, int, dict]] = [
        (f_score(c, wm), i, c) for i, c in enumerate(valid_configs)
    ]
    heapq.heapify(heap)

    best_f, _, best = heapq.heappop(heap)

    top3 = [best]
    for _ in range(min(2, len(heap))):
        _, _, cfg = heapq.heappop(heap)
        top3.append(cfg)

    return best, top3, best_f


def explain_score(config: dict, wm: WorkingMemory) -> dict:
    """Trả về breakdown chi tiết để hiển thị trên UI: {g, h, f, perf, sub_scores, weights}."""
    w = WEIGHTS.get(wm.muc_dich, WEIGHTS["office"])
    cpu_s = min(_f(config["cpu"].get("cores"), 4) / 16.0, 1.0)
    vram  = _f(config.get("vga", {}).get("vram_gb"), 0)
    gpu_s = min(vram / 24.0, 1.0)
    ram_gb = _f(config["ram"].get("capacity_gb"), 8)
    ram_s  = min(ram_gb / 64.0, 1.0)
    storage_t = str(config["storage"].get("type", "")).strip()
    sto_s = {"NVMe": 1.0, "SSD": 0.7, "HDD": 0.3}.get(storage_t, 0.5)

    perf = performance_score(config, wm)
    g_v  = g(config, wm)
    h_v  = 1.0 - perf
    return {
        "g": g_v, "h": h_v, "f": g_v + h_v,
        "perf": perf,
        "sub_scores": {"cpu": cpu_s, "gpu": gpu_s, "ram": ram_s, "storage": sto_s},
        "weights": w,
    }


# ══════════════════════════════════════════════════════════════════
# SELF-TEST — 3 mock configs cùng muc_dich gaming, khác giá/hiệu năng
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    wm = WorkingMemory(ngan_sach=25_000_000, muc_dich="gaming", uu_tien="")

    # Mock 3 cấu hình:
    # A — đắt nhất, hiệu năng cao nhất
    # B — trung bình
    # C — rẻ nhất, hiệu năng thấp
    mock_configs = [
        {
            "label": "A_premium",
            "cpu":       {"cores": 12, "name": "i7-13700"},
            "vga":       {"vram_gb": 12, "name": "RTX 4070"},
            "ram":       {"capacity_gb": 32, "name": "DDR5 32GB"},
            "storage":   {"type": "NVMe",   "name": "Samsung 980 1TB"},
            "mainboard": {}, "psu": {}, "case": {}, "cooler": {},
            "total": 24_000_000,
        },
        {
            "label": "B_balanced",
            "cpu":       {"cores": 6, "name": "Ryzen 5 7500F"},
            "vga":       {"vram_gb": 8, "name": "RTX 4060"},
            "ram":       {"capacity_gb": 16, "name": "DDR5 16GB"},
            "storage":   {"type": "SSD", "name": "WD Blue 500GB"},
            "mainboard": {}, "psu": {}, "case": {}, "cooler": {},
            "total": 18_000_000,
        },
        {
            "label": "C_cheap",
            "cpu":       {"cores": 4, "name": "i3-12100"},
            "vga":       {"vram_gb": 4, "name": "GTX 1650"},
            "ram":       {"capacity_gb": 8, "name": "DDR4 8GB"},
            "storage":   {"type": "HDD", "name": "Seagate 1TB"},
            "mainboard": {}, "psu": {}, "case": {}, "cooler": {},
            "total": 10_000_000,
        },
    ]

    print(f"Test: gaming 25tr — 3 mock configs")
    print(f"{'─'*68}")
    print(f"{'Config':<14} {'total':>11} {'g':>6} {'h':>6} {'f':>6} {'perf':>6}")
    print(f"{'─'*68}")
    for cfg in mock_configs:
        ex = explain_score(cfg, wm)
        print(f"{cfg['label']:<14} {cfg['total']:>11,} "
              f"{ex['g']:>6.3f} {ex['h']:>6.3f} {ex['f']:>6.3f} {ex['perf']:>6.3f}")

    best, top3, best_f = astar_select(mock_configs, wm)
    print(f"\n🏆 Best (gaming): {best['label']}  (f={best_f:.3f})")
    print(f"   Top3 order   : {[c['label'] for c in top3]}")

    # Test với muc_dich='office' — kỳ vọng A vẫn tốt nhưng C có thể leo lên
    wm.muc_dich = "office"
    print(f"\n{'═'*68}")
    print(f"Test cùng 3 configs nhưng muc_dich='office' (gpu trọng số chỉ 5%)")
    print(f"{'─'*68}")
    print(f"{'Config':<14} {'total':>11} {'g':>6} {'h':>6} {'f':>6} {'perf':>6}")
    print(f"{'─'*68}")
    for cfg in mock_configs:
        ex = explain_score(cfg, wm)
        print(f"{cfg['label']:<14} {cfg['total']:>11,} "
              f"{ex['g']:>6.3f} {ex['h']:>6.3f} {ex['f']:>6.3f} {ex['perf']:>6.3f}")

    best, top3, best_f = astar_select(mock_configs, wm)
    print(f"\n🏆 Best (office): {best['label']}  (f={best_f:.3f})")
    print(f"   Top3 order   : {[c['label'] for c in top3]}")

    # Tính nhất quán: best phải có f thấp nhất, không có config nào có f thấp hơn
    all_f = [(c["label"], f_score(c, wm)) for c in mock_configs]
    all_f.sort(key=lambda x: x[1])
    print(f"\n   Verify: min(f) = {all_f[0]} — {'✅ PASS' if all_f[0][0]==best['label'] else '❌ FAIL'}")
