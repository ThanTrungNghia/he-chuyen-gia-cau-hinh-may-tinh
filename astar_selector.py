"""
astar_selector.py — A* Search để chọn cấu hình tối ưu
=======================================================
Nguồn lý thuyết:
  - Module 3 — Tìm kiếm heuristic (slide 13-28: f(n) = g(n) + h(n))
  - lesson_34 — Informed search
      * slide 29: "If h(n) is admissible, A* using TREE-SEARCH is optimal"
      * slide 32: consistent — h(n) ≤ c(n,a,n') + h(n')

Ánh xạ A* vào bài toán chọn cấu hình:
  - Mỗi cấu hình hợp lệ (do CSP trả về) là một goal-state candidate.
  - g(config) = chi phí thực tế đã tích lũy = total_price / ngan_sach   ∈ [0, 1]
  - h(config) = ước tính chi phí "còn lại" = 1 - performance_score(config)
                                                                        ∈ [0, 1]
  - f(config) = g + h (càng nhỏ càng tốt — vừa rẻ vừa mạnh)

Tính ADMISSIBLE của h:
  - performance_score ∈ [0, 1]  (0 = tệ nhất, 1 = hoàn hảo)
  - h = 1 - perf      ∈ [0, 1]
  - Goal lý tưởng có perf = 1 (đạt 100% hiệu năng) → h* = 0
  - h(n) ≤ h*(n) luôn đúng vì h ≥ 0 và h* ≥ 0
  → A* đảm bảo trả về cấu hình có f tối thiểu (định lý slide 29).

Tính CONSISTENT:
  - Trong setting flat list (không có successor), tự động consistent.

Trọng số WEIGHTS theo mục đích sử dụng — phản ánh ưu tiên thực tế:
  ví dụ gaming ưu tiên GPU 45%, office ưu tiên CPU 40% và RAM 35%.
"""

import heapq
from knowledge_base import WorkingMemory


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
      cpu_score = min(cores / 16, 1.0)        — 16 nhân = max
      gpu_score = min(vram_gb / 24, 1.0)      — 24GB VRAM = max (RTX 4090)
      ram_score = min(capacity_gb / 64, 1.0)  — 64GB = max
      storage_score = {NVMe: 1.0, SSD: 0.7, HDD: 0.3}

    Trọng số WEIGHTS[muc_dich] phản ánh đâu là linh kiện quan trọng.
    """
    w = WEIGHTS.get(wm.muc_dich, WEIGHTS["office"])

    cpu_score = min(_f(config["cpu"].get("cores"), 4) / 16.0, 1.0)

    vga = config.get("vga", {})
    vram = _f(vga.get("vram_gb"), 0)
    # GPU score: nếu không có GPU rời (vga_iGPU placeholder), vram=0 → score=0
    gpu_score = min(vram / 24.0, 1.0)

    ram_gb = _f(config["ram"].get("capacity_gb"), 8)
    ram_score = min(ram_gb / 64.0, 1.0)

    storage_type = str(config["storage"].get("type", "")).strip()
    storage_bonus = {"NVMe": 1.0, "SSD": 0.7, "HDD": 0.3}
    storage_score = storage_bonus.get(storage_type, 0.5)

    return (w["cpu"]     * cpu_score
            + w["gpu"]   * gpu_score
            + w["ram"]   * ram_score
            + w["storage"] * storage_score)


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

    Định lý A* (lesson_34 slide 29): nếu h admissible,
    cấu hình trả về có f tối thiểu = lựa chọn tối ưu.
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
    """
    Trả về breakdown chi tiết để hiển thị trên UI/báo cáo:
      {g, h, f, perf, sub_scores: {cpu, gpu, ram, storage}, weights}
    """
    w = WEIGHTS.get(wm.muc_dich, WEIGHTS["office"])
    cpu_s     = min(_f(config["cpu"].get("cores"), 4) / 16.0, 1.0)
    vram      = _f(config.get("vga", {}).get("vram_gb"), 0)
    gpu_s     = min(vram / 24.0, 1.0)
    ram_s     = min(_f(config["ram"].get("capacity_gb"), 8) / 64.0, 1.0)
    storage_t = str(config["storage"].get("type", "")).strip()
    sto_s     = {"NVMe": 1.0, "SSD": 0.7, "HDD": 0.3}.get(storage_t, 0.5)

    perf = w["cpu"]*cpu_s + w["gpu"]*gpu_s + w["ram"]*ram_s + w["storage"]*sto_s
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
