"""
forward_chaining.py — Kết nối luật với dữ liệu thực tế, phân bổ ngân sách
==========================================================================
File này thực hiện 2 bước:
  Bước 1 — Chạy 31 luật từ knowledge_base.py để xác định tầm giá linh kiện
  Bước 2 — Chạy 9 luật trong file này (R32–R40) để tính ngân sách cho từng linh kiện

Cả 2 bước đều lặp cho đến khi không còn luật nào áp dụng được nữa.

Kết quả trả về:
  - Tầm giá linh kiện (từ bước 1): cpu_tier, gpu_tier, ram_capacity, ...
  - Ngân sách từng linh kiện (từ bước 2): cpu_budget, gpu_budget, ram_budget, ...
  - Yêu cầu tối thiểu: ram_min_gb, vram_min_gb, storage_min_gb
"""

from typing import Callable
from dataclasses import dataclass

from knowledge_base import WorkingMemory
from inference_engine import ForwardChaining


# ══════════════════════════════════════════════════════════════════
# Bảng chuyển đổi tầm giá từ tên đầy đủ sang tên ngắn trong file CSV
# Ví dụ: "mid-range" trong luật → "mid" trong cột cpu_tier của Data_CPU.csv
# ══════════════════════════════════════════════════════════════════

TIER_MAP_CPU: dict[str, str] = {
    # KB cpu_tier  →  CSV cpu_tier (cột Data_CPU.csv)
    "budget":    "budget",
    "mid-range": "mid",
    "high-end":  "high",
    "ultra":     "extreme",
}

TIER_MAP_GPU: dict[str, str | None] = {
    # KB gpu_tier  →  CSV gpu_tier (cột Data_VGA.csv)
    # "none" map về None — báo cho CSP biết KHÔNG cần GPU rời (dùng iGPU)
    "none":         None,
    "budget":       "budget",
    "gaming-mid":   "mid",
    "gaming-high":  "mid-high",
    "workstation":  "high",
}

# Chuyển đổi tầm giá CPU sang tên dùng trong bảng phân bổ ngân sách
OVERALL_TIER: dict[str, str] = {
    "budget":    "budget",
    "mid-range": "mid",
    "high-end":  "high",
    "ultra":     "extreme",
}

# Tỉ lệ % ngân sách phân bổ cho 6 linh kiện chính theo tầm giá.
# Tổng mỗi hàng = 100%. Vỏ máy và tản nhiệt tính riêng trong csp_checker.py
# (vỏ = 6% ngân sách, tản nhiệt = 4% ngân sách).
BUDGET_PERCENTS: dict[str, dict[str, float]] = {
    "budget":  {"cpu": 0.25, "gpu": 0.30, "ram": 0.15, "mb": 0.15, "psu": 0.08, "storage": 0.07},
    "mid":     {"cpu": 0.22, "gpu": 0.35, "ram": 0.13, "mb": 0.15, "psu": 0.07, "storage": 0.08},
    "high":    {"cpu": 0.20, "gpu": 0.40, "ram": 0.12, "mb": 0.13, "psu": 0.07, "storage": 0.08},
    "extreme": {"cpu": 0.18, "gpu": 0.45, "ram": 0.11, "mb": 0.12, "psu": 0.06, "storage": 0.08},
}

# Cấu hình tối thiểu theo mục đích sử dụng (dùng ở luật R39).
MIN_SPECS: dict[str, dict[str, int]] = {
    "office":    {"ram_min_gb":  8, "vram_min_gb":  0, "storage_min_gb":  256},
    "study":     {"ram_min_gb": 16, "vram_min_gb":  0, "storage_min_gb":  512},
    "gaming":    {"ram_min_gb": 16, "vram_min_gb":  8, "storage_min_gb":  512},
    "graphics":  {"ram_min_gb": 32, "vram_min_gb": 12, "storage_min_gb": 1000},
    "streaming": {"ram_min_gb": 32, "vram_min_gb":  8, "storage_min_gb": 1000},
    "editing":   {"ram_min_gb": 32, "vram_min_gb":  8, "storage_min_gb": 1000},
}


# ══════════════════════════════════════════════════════════════════
# BudgetRule — cấu trúc đơn giản cho các luật phân bổ ngân sách (R32–R40)
# Không cần priority/name như luật trong knowledge_base.py
# ══════════════════════════════════════════════════════════════════
@dataclass
class BudgetRule:
    id:        str
    doc:       str                                   # mô tả IF-THEN bằng tiếng Việt
    condition: Callable[[WorkingMemory], bool]
    action:    Callable[[WorkingMemory], None]


def _alloc(wm: WorkingMemory, tier_key: str) -> None:
    """Tính và gán ngân sách cho 6 linh kiện chính theo tỉ lệ đã định sẵn."""
    pct = BUDGET_PERCENTS[tier_key]
    wm.cpu_budget     = wm.ngan_sach * pct["cpu"]
    wm.gpu_budget     = wm.ngan_sach * pct["gpu"]
    wm.ram_budget     = wm.ngan_sach * pct["ram"]
    wm.mb_budget      = wm.ngan_sach * pct["mb"]
    wm.psu_budget     = wm.ngan_sach * pct["psu"]
    wm.storage_budget = wm.ngan_sach * pct["storage"]


BUDGET_RULES: list[BudgetRule] = [

    # ── Nhóm A: Tính ngân sách cho từng linh kiện theo tầm giá (R32–R35) ──
    BudgetRule(
        id="R32",
        doc="IF cpu_tier=budget AND chưa phân bổ THEN gán % budget tier",
        condition=lambda wm: wm.cpu_tier == "budget" and wm.cpu_budget == 0.0,
        action=lambda wm: _alloc(wm, "budget"),
    ),
    BudgetRule(
        id="R33",
        doc="IF cpu_tier=mid-range AND chưa phân bổ THEN gán % mid tier",
        condition=lambda wm: wm.cpu_tier == "mid-range" and wm.cpu_budget == 0.0,
        action=lambda wm: _alloc(wm, "mid"),
    ),
    BudgetRule(
        id="R34",
        doc="IF cpu_tier=high-end AND chưa phân bổ THEN gán % high tier",
        condition=lambda wm: wm.cpu_tier == "high-end" and wm.cpu_budget == 0.0,
        action=lambda wm: _alloc(wm, "high"),
    ),
    BudgetRule(
        id="R35",
        doc="IF cpu_tier=ultra AND chưa phân bổ THEN gán % extreme tier",
        condition=lambda wm: wm.cpu_tier == "ultra" and wm.cpu_budget == 0.0,
        action=lambda wm: _alloc(wm, "extreme"),
    ),

    # ── Nhóm B: Nếu không cần card rời thì chuyển tiền sang CPU (R36) ─────
    BudgetRule(
        id="R36",
        doc="IF gpu_tier=none AND gpu_budget>0 THEN dồn 60% gpu_budget sang cpu_budget",
        condition=lambda wm: wm.gpu_tier == "none" and wm.gpu_budget > 0.0,
        action=lambda wm: setattr(wm, "cpu_budget", wm.cpu_budget + wm.gpu_budget * 0.6) or
                         setattr(wm, "gpu_budget", 0.0),
    ),

    # ── Nhóm C: Điều chỉnh ngân sách theo ưu tiên người dùng (R37–R38) ────
    BudgetRule(
        id="R37",
        doc="IF uu_tien=performance AND đã phân bổ THEN +5% gpu, -5% cpu",
        condition=lambda wm: (
            wm.uu_tien == "performance"
            and wm.gpu_budget > 0.0      # phải có GPU rời mới tăng được
            and wm.cpu_budget > 0.0
            and not getattr(wm, "_perf_adjusted", False)
        ),
        action=lambda wm: (
            setattr(wm, "gpu_budget", wm.gpu_budget + wm.ngan_sach * 0.05),
            setattr(wm, "cpu_budget", wm.cpu_budget - wm.ngan_sach * 0.05),
            setattr(wm, "_perf_adjusted", True),
        ),
    ),
    BudgetRule(
        id="R38",
        doc="IF uu_tien=value AND đã phân bổ THEN giảm 5% mỗi linh kiện (giữ buffer)",
        condition=lambda wm: (
            wm.uu_tien == "value"
            and wm.cpu_budget > 0.0
            and not getattr(wm, "_value_adjusted", False)
        ),
        action=lambda wm: (
            setattr(wm, "cpu_budget",     wm.cpu_budget * 0.95),
            setattr(wm, "gpu_budget",     wm.gpu_budget * 0.95),
            setattr(wm, "ram_budget",     wm.ram_budget * 0.95),
            setattr(wm, "mb_budget",      wm.mb_budget  * 0.95),
            setattr(wm, "psu_budget",     wm.psu_budget * 0.95),
            setattr(wm, "storage_budget", wm.storage_budget * 0.95),
            setattr(wm, "_value_adjusted", True),
        ),
    ),

    # ── Nhóm D: Xác định cấu hình tối thiểu theo mục đích (R39) ───────────
    BudgetRule(
        id="R39",
        doc="IF chưa có ram_min_gb THEN tra MIN_SPECS[muc_dich] gán min specs",
        condition=lambda wm: wm.ram_min_gb == 0,
        action=lambda wm: _set_min_specs(wm),
    ),

    # ── Nhóm E: Tăng thêm ngân sách GPU/CPU để tận dụng tốt hơn (R40) ────
    # Nới rộng lên 88-95% thay vì 80-85%, CSP vẫn kiểm soát tổng tiền thực.
    BudgetRule(
        id="R40",
        doc="Tăng gpu_budget thêm 10%, cpu_budget thêm 5% để tối đa hóa hiệu năng trong ngân sách",
        condition=lambda wm: wm.cpu_budget > 0 and not getattr(wm, "_r40_adjusted", False),
        action=lambda wm: (
            setattr(wm, "gpu_budget", wm.gpu_budget + wm.ngan_sach * 0.10),
            setattr(wm, "cpu_budget", wm.cpu_budget + wm.ngan_sach * 0.05),
            setattr(wm, "_r40_adjusted", True),
        ),
    ),
]


def _set_min_specs(wm: WorkingMemory) -> None:
    """Điền yêu cầu tối thiểu về RAM/VRAM/ổ cứng theo mục đích sử dụng."""
    spec = MIN_SPECS.get(wm.muc_dich, MIN_SPECS["office"])
    wm.ram_min_gb     = spec["ram_min_gb"]
    wm.vram_min_gb    = spec["vram_min_gb"] if wm.gpu_tier != "none" else 0
    wm.storage_min_gb = spec["storage_min_gb"]


# ══════════════════════════════════════════════════════════════════
# Hàm chính — streamlit_app.py gọi hàm này để chạy toàn bộ quá trình tư vấn
# ══════════════════════════════════════════════════════════════════
def run_forward_chaining(wm: WorkingMemory) -> tuple[WorkingMemory, list[str]]:
    """
    Chạy toàn bộ quá trình suy luận tự động từ thông tin người dùng.

    Tham số:
        wm: thông tin người dùng (ngân sách, mục đích, ưu tiên)
    Trả về:
        (wm đã được điền đầy đủ, danh sách ID các luật đã chạy)

    Bước 1: Chạy 31 luật để xác định tầm giá từng linh kiện
    Bước 2: Chạy 9 luật để tính ngân sách cho từng linh kiện và yêu cầu tối thiểu
    Mỗi bước lặp đến khi không còn luật nào khớp, mỗi luật chỉ chạy 1 lần.
    """
    # ── Stage 1: tier-based IE ────────────────────────────────────
    ie = ForwardChaining()
    wm = ie.run(wm)
    fired_all: list[str] = list(wm.fired_rules)

    # ── Stage 2: budget allocation fixpoint ───────────────────────
    fired_stage2: set[str] = set()
    changed = True
    while changed:
        changed = False
        for rule in BUDGET_RULES:
            if rule.id in fired_stage2:
                continue
            try:
                cond_met = rule.condition(wm)
            except Exception:
                cond_met = False
            if not cond_met:
                continue

            try:
                rule.action(wm)
            except Exception as e:
                wm.warnings.append(f"Rule {rule.id} lỗi action: {e}")
                fired_stage2.add(rule.id)
                continue

            fired_stage2.add(rule.id)
            wm.fired_rules.append(rule.id)
            wm.explanation.append(f"[{rule.id}] {rule.doc}")
            fired_all.append(rule.id)
            changed = True

    return wm, fired_all


# ══════════════════════════════════════════════════════════════════
# SELF-TEST — 5 test cases CLAUDE.md mục 10
# Vocab CLAUDE.md (van_phong/lap_trinh/do_hoa) đã được map sang
# vocab KB (office/study/graphics) trước khi chạy.
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Map vocab CLAUDE.md → vocab KB
    MUC_DICH_MAP = {
        "van_phong": "office",
        "gaming":    "gaming",
        "do_hoa":    "graphics",
        "lap_trinh": "study",
    }

    test_cases = [
        {"ngan_sach":  8_000_000, "muc_dich": "van_phong", "uu_tien": "",            "expect": "tier=budget"},
        {"ngan_sach": 15_000_000, "muc_dich": "gaming",    "uu_tien": "",            "expect": "tier=mid"},
        {"ngan_sach": 25_000_000, "muc_dich": "gaming",    "uu_tien": "performance", "expect": "tier=high, gpu++"},
        {"ngan_sach": 20_000_000, "muc_dich": "do_hoa",    "uu_tien": "",            "expect": "gpu rời, vram>=12"},
        {"ngan_sach": 12_000_000, "muc_dich": "lap_trinh", "uu_tien": "value",       "expect": "gpu=none, value-5%"},
    ]

    all_passed = True
    for i, tc in enumerate(test_cases, 1):
        wm = WorkingMemory(
            ngan_sach=tc["ngan_sach"],
            muc_dich=MUC_DICH_MAP[tc["muc_dich"]],
            uu_tien=tc["uu_tien"],
        )
        wm, fired = run_forward_chaining(wm)

        total_pct = (wm.cpu_budget + wm.gpu_budget + wm.ram_budget
                     + wm.mb_budget + wm.psu_budget + wm.storage_budget) / wm.ngan_sach

        print(f"\n{'─'*68}")
        print(f"Test {i} — {tc['muc_dich']} {tc['ngan_sach']:,}đ + {tc['uu_tien'] or 'no priority'}")
        print(f"  Expect       : {tc['expect']}")
        print(f"  Fired rules  : {fired}")
        print(f"  CPU/GPU tier : {wm.cpu_tier} / {wm.gpu_tier}")
        print(f"  Min specs    : RAM>={wm.ram_min_gb}GB, VRAM>={wm.vram_min_gb}GB, STO>={wm.storage_min_gb}GB")
        print(f"  Budget VND   : CPU={wm.cpu_budget:>11,.0f}  GPU={wm.gpu_budget:>11,.0f}  RAM={wm.ram_budget:>10,.0f}")
        print(f"               : MB ={wm.mb_budget:>11,.0f}  PSU={wm.psu_budget:>11,.0f}  STO={wm.storage_budget:>10,.0f}")
        print(f"  Tổng % alloc : {total_pct*100:.1f}% ngân sách")

        # Quick assertions
        if tc["muc_dich"] == "van_phong" and wm.cpu_tier not in ("budget", "mid-range"):
            print(f"  ❌ FAIL: expect cpu_tier in (budget, mid-range), got {wm.cpu_tier}")
            all_passed = False
        elif tc["muc_dich"] == "do_hoa" and wm.vram_min_gb < 12:
            print(f"  ❌ FAIL: expect vram_min_gb>=12, got {wm.vram_min_gb}")
            all_passed = False
        elif tc["muc_dich"] == "lap_trinh" and wm.ngan_sach < 20_000_000 and wm.gpu_tier not in ("none", "budget"):
            print(f"  ❌ FAIL: expect gpu_tier in (none, budget), got {wm.gpu_tier}")
            all_passed = False
        else:
            print(f"  ✅ PASS")

    print(f"\n{'═'*68}")
    print(f"{'✅ TẤT CẢ 5 TEST PASS' if all_passed else '❌ CÓ TEST FAIL'}")
