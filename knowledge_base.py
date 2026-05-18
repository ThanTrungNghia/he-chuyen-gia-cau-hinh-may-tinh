"""
knowledge_base.py — Knowledge Base của hệ chuyên gia tư vấn cấu hình PC
=========================================================================
Chứa:
  - Ngưỡng ngân sách (CONSTANTS block)
  - Lookup dicts: PSU_TIER_WATTAGE, COOLER_TDP_SUPPORT, STORAGE_CAPACITY
  - 31 luật IF-THEN (R01–R31, R04 tách thành R04a/R04b)
  - Cấu trúc Working Memory
  - Hàm get_all_rules() để Inference Engine nạp vào

Nguyên tắc thiết kế:
  - Mỗi luật có ID duy nhất (R01–R31, R04a, R04b)
  - Luật cụ thể hơn (nhiều điều kiện hơn) được ưu tiên trước → specificity ordering
  - Mỗi luật khi kích hoạt sẽ ghi lý do vào explanation[] — DO inference_engine quản lý
  - action lambda KHÔNG được gọi wm.explanation.append() — chỉ engine mới ghi explanation
  - Không có hardcode sản phẩm cụ thể — KB chỉ xác định TIER
    (CSP checker sẽ map tier → sản phẩm thực từ products.json)

Nhóm 1  (R01–R05):    Office builds
Nhóm 2  (R06–R11):    Gaming builds
Nhóm 3  (R12–R15):    Đồ họa / Workstation
Nhóm 4  (R16–R17):    Streaming / Content Creation
Nhóm 5  (R18–R20):    Override rules (ưu tiên người dùng)
Nhóm 6  (R21–R24):    Study / Editing
Nhóm 7  (R25–R27):    Override compact / quiet / value
Nhóm 8  (R28–R31):    Gap coverage bổ sung
"""

from dataclasses import dataclass, field
from typing import Callable


# ══════════════════════════════════════════════════════════════════
# CONSTANTS — Ngưỡng ngân sách (VNĐ)
# Tập trung vào một chỗ để dễ bảo trì, tránh magic numbers trong lambda
# ══════════════════════════════════════════════════════════════════

# ── Office ────────────────────────────────────────────────────────
BUDGET_OFFICE_LOW   =  8_000_000   # R01/R02: cực rẻ / phổ thông
BUDGET_OFFICE_MID   = 15_000_000   # R02/R03: phổ thông / cao cấp
BUDGET_OFFICE_HIGH  = 25_000_000   # R03/R04a: cao cấp / workstation nhẹ
BUDGET_OFFICE_ULTRA = 50_000_000   # R04a/R04b: workstation nhẹ / workstation cao cấp

# ── Gaming ────────────────────────────────────────────────────────
BUDGET_GAMING_ENTRY   = 15_000_000   # R06/R07: nhập môn / phổ thông 1080p
BUDGET_GAMING_MID     = 25_000_000   # R07/R08: 1080p / 1440p
BUDGET_GAMING_HIGH    = 40_000_000   # R08/R09: 1440p / 4K high-end
BUDGET_GAMING_ULTRA   = 60_000_000   # R09/R10: 4K / no-compromise
BUDGET_GAMING_WARNING = 30_000_000   # R20: ngưỡng cảnh báo GPU gaming-high quá đắt

# ── Graphics / Workstation ────────────────────────────────────────
BUDGET_GRAPHICS_LOW  = 20_000_000   # R12/R13: sinh viên thiết kế / 3D video
BUDGET_GRAPHICS_MID  = 40_000_000   # R13/R14: 3D video / workstation chuyên nghiệp
BUDGET_GRAPHICS_HIGH = 70_000_000   # R14/R15 cùng dùng ngưỡng 70tr:
                                     #   R14: < 70tr (workstation), R15: >= 70tr (AI/ML)

# ── Streaming ─────────────────────────────────────────────────────
BUDGET_STREAMING_MID  = 20_000_000   # R16 lower bound / R31 upper bound: < 20tr quá thấp
BUDGET_STREAMING_HIGH = 35_000_000   # R16/R17: streaming trung / streaming chuyên nghiệp

# ── Study (sinh viên lập trình) ───────────────────────────────────
BUDGET_STUDY_LOW  = BUDGET_OFFICE_MID   # 15tr: R21/R22 split
BUDGET_STUDY_HIGH = 25_000_000           # 25tr: R22 upper bound / R28 lower bound

# ── Editing (dựng phim) ───────────────────────────────────────────
BUDGET_EDITING_MID = 30_000_000   # R23/R24 split; cũng dùng cho R27 storage check

# ── Override rules ────────────────────────────────────────────────
BUDGET_PERF_OVERRIDE = 20_000_000   # R18: ngưỡng tối thiểu để nâng GPU tier


# ══════════════════════════════════════════════════════════════════
# PSU_TIER_WATTAGE — Công suất nguồn tối thiểu gợi ý theo tier
# Inference Engine sẽ tra dict này để điền wm.psu_wattage_min
# ══════════════════════════════════════════════════════════════════
PSU_TIER_WATTAGE: dict[str, int] = {
    "basic":  450,    # đủ cho build iGPU hoặc GPU budget TDP thấp
    "mid":    650,    # đủ cho GPU gaming-mid + CPU mid-range
    "high":   850,    # đủ cho GPU gaming-high + CPU high-end (TDP ~300-350W cả hệ)
    "ultra": 1000,    # bắt buộc cho GPU workstation — RTX 4090 alone ~450W
}


# ══════════════════════════════════════════════════════════════════
# COOLER_TDP_SUPPORT — Công suất nhiệt tối đa cooler có thể tản (W)
# Dùng để kiểm tra xem cooler có phù hợp với CPU TDP không
# ══════════════════════════════════════════════════════════════════
COOLER_TDP_SUPPORT: dict[str, int] = {
    "stock":      65,    # cooler đi kèm CPU — chỉ đủ cho TDP cơ bản (65W)
    "air-budget": 150,   # Deepcool AK400, ID-Cooling SE-224-XT
    "air-mid":    200,   # Thermalright Assassin X, Noctua U12S
    "air-high":   250,   # Noctua NH-D15, be quiet! Dark Rock Pro 4
    "aio-240":    280,   # AIO 240mm — tốt cho CPU 125-165W
    "aio-360":    350,   # AIO 360mm — bắt buộc cho CPU TDP 170W+ (i9, Ryzen 9)
}


# ══════════════════════════════════════════════════════════════════
# STORAGE_CAPACITY — Dung lượng gợi ý theo kiểu cấu hình storage
# nvme_gb: dung lượng ổ NVMe/SSD chính; hdd_gb: ổ cứng HDD phụ
# ══════════════════════════════════════════════════════════════════
STORAGE_CAPACITY: dict[str, dict[str, int]] = {
    "ssd-only":  {"nvme_gb":  500, "hdd_gb":    0},   # SSD SATA 500GB đủ dùng văn phòng
    "nvme-only": {"nvme_gb": 1000, "hdd_gb":    0},   # NVMe 1TB cho tốc độ cao, game
    "ssd+hdd":   {"nvme_gb":  500, "hdd_gb": 2000},   # SSD boot/phần mềm + HDD lưu trữ
    "nvme+hdd":  {"nvme_gb": 1000, "hdd_gb": 2000},   # NVMe công việc + HDD lưu project lớn
}


# ══════════════════════════════════════════════════════════════════
# WORKING MEMORY — trạng thái hiện tại của hệ thống
# ══════════════════════════════════════════════════════════════════
@dataclass
class WorkingMemory:
    # ── Input từ người dùng ───────────────────────────────────────
    ngan_sach: int = 0        # VNĐ — phải > 0, khuyến nghị >= 5_000_000
    muc_dich:  str = ""       # Valid: office | gaming | graphics | streaming | study | editing
    uu_tien:   str = ""       # Valid: performance | value | quiet | compact

    # ── Output — KB sẽ điền vào các field này ────────────────────
    cpu_tier:       str = ""   # budget | mid-range | high-end | ultra
    gpu_tier:       str = ""   # none | budget | gaming-mid | gaming-high | workstation
    ram_capacity:   int = 0    # GB: 8 | 16 | 32 | 64
    ram_type:       str = ""   # DDR4 | DDR5
    storage_config: str = ""   # ssd-only | ssd+hdd | nvme-only | nvme+hdd
    form_factor:    str = ""   # ATX | mATX | ITX
    cooler_type:    str = ""   # stock | air-budget | air-mid | air-high | aio-240 | aio-360
    psu_tier:       str = ""   # basic | mid | high | ultra

    # ── Output chi tiết — _enrich_output() điền sau Forward Chaining ──
    psu_wattage_min:  int = 0   # W — tra từ PSU_TIER_WATTAGE
    cooler_tdp_max:   int = 0   # W — tra từ COOLER_TDP_SUPPORT
    nvme_capacity_gb: int = 0   # GB — tra từ STORAGE_CAPACITY[storage_config]["nvme_gb"]
    hdd_capacity_gb:  int = 0   # GB — tra từ STORAGE_CAPACITY[storage_config]["hdd_gb"]

    # ── Budget allocation — forward_chaining.py điền (rules R32–R39) ──
    # Phân bổ % ngân sách cho từng linh kiện (VNĐ, đã quy đổi từ %)
    cpu_budget:     float = 0.0
    gpu_budget:     float = 0.0
    ram_budget:     float = 0.0
    mb_budget:      float = 0.0
    psu_budget:     float = 0.0
    storage_budget: float = 0.0

    # ── Min specs — suy từ tier + muc_dich (rules R39) ──
    ram_min_gb:     int = 0   # RAM tối thiểu (GB) cho mục đích này
    vram_min_gb:    int = 0   # VRAM tối thiểu (GB) — 0 nếu không cần GPU rời
    storage_min_gb: int = 0   # Dung lượng storage tối thiểu (GB)

    # ── Metadata ──────────────────────────────────────────────────
    explanation: list = field(default_factory=list)   # chuỗi luật đã kích hoạt (DO engine ghi)
    fired_rules: list = field(default_factory=list)   # ID các luật đã fire
    warnings:    list = field(default_factory=list)   # cảnh báo cho người dùng


# ══════════════════════════════════════════════════════════════════
# RULE — cấu trúc một luật IF-THEN
# ══════════════════════════════════════════════════════════════════
@dataclass
class Rule:
    id:          str                                   # VD: "R01", "R04a"
    name:        str                                   # Tên ngắn gọn
    priority:    int                                   # Số càng cao càng được xét trước
    condition:   Callable[[WorkingMemory], bool]       # Hàm kiểm tra điều kiện
    action:      Callable[[WorkingMemory], None]       # Hàm cập nhật WM
    explanation: str                                   # Giải thích luật bằng tiếng Việt


# ══════════════════════════════════════════════════════════════════
# 31 LUẬT IF-THEN
# QUAN TRỌNG: action lambda KHÔNG được gọi wm.explanation.append()
#             Inference Engine tự ghi explanation sau mỗi rule fire.
#             action chỉ được gọi: wm.__dict__.update() và wm.warnings.append()
# ══════════════════════════════════════════════════════════════════

def _rules() -> list[Rule]:
    return [

        # ─────────────────────────────────────────────────────────
        # NHÓM 1 — OFFICE BUILDS
        # Ưu tiên: tiết kiệm điện, yên tĩnh, iGPU, không cần GPU rời
        # ─────────────────────────────────────────────────────────

        Rule(
            id="R01", priority=90,
            name="Office cực rẻ (< 8 triệu)",
            condition=lambda wm: wm.muc_dich == "office" and wm.ngan_sach < BUDGET_OFFICE_LOW,
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "budget",
                "gpu_tier":       "none",       # dùng iGPU tích hợp
                "ram_capacity":   8,
                "ram_type":       "DDR4",
                "storage_config": "ssd-only",   # SSD SATA 256-480GB đủ dùng
                "form_factor":    "mATX",
                "cooler_type":    "stock",       # CPU TDP thấp, tản nhiệt stock đủ
                "psu_tier":       "basic",
            }),
            explanation=(
                "Ngân sách < 8tr + mục đích văn phòng → chọn CPU budget có iGPU tích hợp, "
                "không cần GPU rời. RAM 8GB DDR4 đủ cho tác vụ office. "
                "SSD SATA tiết kiệm chi phí. PSU 450W là đủ."
            ),
        ),

        Rule(
            id="R02", priority=88,
            name="Office phổ thông (8–15 triệu)",
            condition=lambda wm: (
                wm.muc_dich == "office"
                and BUDGET_OFFICE_LOW <= wm.ngan_sach < BUDGET_OFFICE_MID
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "mid-range",
                "gpu_tier":       "none",
                "ram_capacity":   16,
                "ram_type":       "DDR4",
                "storage_config": "ssd-only",
                "form_factor":    "mATX",
                "cooler_type":    "stock",
                "psu_tier":       "basic",
            }),
            explanation=(
                "Ngân sách 8–15tr + office → CPU mid-range có iGPU (i5-12400, Ryzen 5 5600G). "
                "Nâng RAM lên 16GB để multitask tốt hơn. "
                "Vẫn không cần GPU rời — tiết kiệm ngân sách cho CPU/RAM tốt hơn."
            ),
        ),

        Rule(
            id="R03", priority=85,
            name="Office cao cấp / làm việc nặng (15–25 triệu)",
            condition=lambda wm: (
                wm.muc_dich == "office"
                and BUDGET_OFFICE_MID <= wm.ngan_sach < BUDGET_OFFICE_HIGH
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "mid-range",
                "gpu_tier":       "budget",     # GPU rời nhẹ cho xuất file, Photoshop nhẹ
                "ram_capacity":   32,
                "ram_type":       "DDR4",
                "storage_config": "nvme-only",  # NVMe nhanh hơn cho công việc
                "form_factor":    "ATX",
                "cooler_type":    "air-budget",
                "psu_tier":       "mid",
            }),
            explanation=(
                "Ngân sách 15–25tr + office nặng → CPU mid-range mạnh, thêm GPU rời budget "
                "hỗ trợ xuất file đồ họa nhẹ, Photoshop, video call chất lượng cao. "
                "RAM 32GB để mở nhiều tab/ứng dụng cùng lúc. NVMe cho tốc độ làm việc."
            ),
        ),

        # R04 CŨ đã được tách thành R04a (25–50tr) và R04b (> 50tr)
        Rule(
            id="R04a", priority=83,
            name="Workstation văn phòng trung (25–50 triệu)",
            condition=lambda wm: (
                wm.muc_dich == "office"
                and BUDGET_OFFICE_HIGH <= wm.ngan_sach < BUDGET_OFFICE_ULTRA
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "high-end",
                "gpu_tier":       "gaming-mid",
                "ram_capacity":   32,
                "ram_type":       "DDR5",
                "storage_config": "nvme+hdd",
                "form_factor":    "ATX",
                "cooler_type":    "air-mid",
                "psu_tier":       "mid",
            }),
            explanation=(
                "Ngân sách 25–50tr + office → workstation chuyên nghiệp tầm trung. "
                "CPU high-end nhiều nhân cho đa nhiệm nặng. DDR5 cho băng thông cao. "
                "NVMe + HDD: NVMe cho OS/phần mềm, HDD lưu trữ file lớn."
            ),
        ),

        Rule(
            id="R04b", priority=82,
            name="Workstation văn phòng cao cấp (> 50 triệu)",
            condition=lambda wm: (
                wm.muc_dich == "office"
                and wm.ngan_sach >= BUDGET_OFFICE_ULTRA
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "ultra",
                "gpu_tier":       "gaming-high",
                "ram_capacity":   64,
                "ram_type":       "DDR5",
                "storage_config": "nvme+hdd",
                "form_factor":    "ATX",
                "cooler_type":    "aio-240",
                "psu_tier":       "high",
            }),
            explanation=(
                "Ngân sách > 50tr + office → workstation cao cấp không giới hạn. "
                "CPU ultra (i9, Ryzen 9) cho đa nhiệm cực nặng. RAM 64GB DDR5 cho "
                "ảo hóa, phân tích dữ liệu lớn. GPU gaming-high hỗ trợ tăng tốc AI/ML nhẹ."
            ),
        ),

        Rule(
            id="R05", priority=80,
            name="Office compact / mini PC",
            condition=lambda wm: wm.muc_dich == "office" and wm.uu_tien == "compact",
            action=lambda wm: wm.__dict__.update({
                "form_factor": "ITX",
                "cooler_type": "air-budget",   # Cooler thấp profile cho case ITX
            }),
            explanation=(
                "Ưu tiên compact + office → chuyển sang form factor ITX. "
                "Case nhỏ gọn, đặt trên bàn hoặc sau màn hình. "
                "Lưu ý: ITX giới hạn số slot RAM và khả năng nâng cấp sau này."
            ),
        ),

        # ─────────────────────────────────────────────────────────
        # NHÓM 2 — GAMING BUILDS
        # Ưu tiên: GPU mạnh, balance CPU-GPU, không bottleneck
        # ─────────────────────────────────────────────────────────

        Rule(
            id="R06", priority=95,
            name="Gaming nhập môn (< 15 triệu)",
            condition=lambda wm: wm.muc_dich == "gaming" and wm.ngan_sach < BUDGET_GAMING_ENTRY,
            action=lambda wm: (
                wm.__dict__.update({
                    "cpu_tier":       "budget",
                    "gpu_tier":       "budget",
                    "ram_capacity":   16,
                    "ram_type":       "DDR4",
                    "storage_config": "ssd-only",
                    "form_factor":    "mATX",
                    "cooler_type":    "air-budget",
                    "psu_tier":       "basic",
                }),
                wm.warnings.append(
                    "Ngân sách gaming < 15tr khá hạn chế — "
                    "kỳ vọng chơi game ở cài đặt thấp-trung, 1080p."
                )
            ),
            explanation=(
                "Gaming < 15tr → ưu tiên GPU rời budget (GTX 1650, RX 6500 XT). "
                "CPU budget tránh bottleneck với GPU tầm này. "
                "RAM 16GB DDR4 là tối thiểu cho gaming 2024. "
                "Chơi được 1080p cài đặt medium-low."
            ),
        ),

        Rule(
            id="R07", priority=93,
            name="Gaming phổ thông 1080p (15–25 triệu)",
            condition=lambda wm: (
                wm.muc_dich == "gaming"
                and BUDGET_GAMING_ENTRY <= wm.ngan_sach < BUDGET_GAMING_MID
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "mid-range",
                "gpu_tier":       "gaming-mid",
                "ram_capacity":   16,
                "ram_type":       "DDR4",
                "storage_config": "nvme-only",
                "form_factor":    "ATX",
                "cooler_type":    "air-budget",
                "psu_tier":       "mid",
            }),
            explanation=(
                "Gaming 15–25tr → sweet spot cho 1080p gaming. "
                "GPU gaming-mid (RTX 3060, RX 6600) cho 60-144fps ở 1080p Ultra. "
                "CPU mid-range tránh bottleneck. NVMe giảm thời gian load game. "
                "RAM 16GB DDR4 đủ cho mọi game hiện tại."
            ),
        ),

        Rule(
            id="R08", priority=92,
            name="Gaming 1440p (25–40 triệu)",
            condition=lambda wm: (
                wm.muc_dich == "gaming"
                and BUDGET_GAMING_MID <= wm.ngan_sach < BUDGET_GAMING_HIGH
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "mid-range",
                "gpu_tier":       "gaming-high",
                "ram_capacity":   32,
                "ram_type":       "DDR5",
                "storage_config": "nvme-only",
                "form_factor":    "ATX",
                "cooler_type":    "air-mid",
                "psu_tier":       "mid",
            }),
            explanation=(
                "Gaming 25–40tr → target 1440p 144fps. "
                "GPU gaming-high (RTX 4070, RX 7800 XT) xử lý tốt 1440p. "
                "Nâng RAM lên 32GB DDR5 cho hiệu năng tốt hơn và tương lai. "
                "Cooler air-mid cần thiết vì CPU mid-range chạy tải cao liên tục."
            ),
        ),

        Rule(
            id="R09", priority=90,
            name="Gaming 4K / High-end (40–60 triệu)",
            condition=lambda wm: (
                wm.muc_dich == "gaming"
                and BUDGET_GAMING_HIGH <= wm.ngan_sach < BUDGET_GAMING_ULTRA
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "high-end",
                "gpu_tier":       "gaming-high",
                "ram_capacity":   32,
                "ram_type":       "DDR5",
                "storage_config": "nvme+hdd",
                "form_factor":    "ATX",
                "cooler_type":    "aio-240",
                "psu_tier":       "high",
            }),
            explanation=(
                "Gaming 40–60tr → target 4K hoặc 1440p 240fps. "
                "GPU gaming-high tier cao (RTX 4070 Ti, RX 7900 XT). "
                "CPU high-end để không bottleneck GPU mạnh. "
                "AIO 240mm cần thiết vì CPU high-end TDP cao (125W+). "
                "PSU high tier vì tổng TDP hệ thống tăng đáng kể."
            ),
        ),

        Rule(
            id="R10", priority=88,
            name="Gaming Ultra / No compromise (> 60 triệu)",
            condition=lambda wm: wm.muc_dich == "gaming" and wm.ngan_sach >= BUDGET_GAMING_ULTRA,
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "ultra",
                "gpu_tier":       "workstation",  # RTX 4090 / RX 7900 XTX
                "ram_capacity":   64,
                "ram_type":       "DDR5",
                "storage_config": "nvme+hdd",
                "form_factor":    "ATX",
                "cooler_type":    "aio-360",
                "psu_tier":       "ultra",
            }),
            explanation=(
                "Gaming > 60tr → build no-compromise. RTX 4090 / RX 7900 XTX. "
                "CPU ultra (i9-13900K, Ryzen 9 7950X) để không bottleneck GPU flagship. "
                "AIO 360mm bắt buộc cho CPU TDP 170W+. "
                "PSU ultra (1000W+) vì GPU alone có thể ngốn 450W."
            ),
        ),

        Rule(
            id="R11", priority=85,
            name="Gaming + ưu tiên yên tĩnh",
            condition=lambda wm: wm.muc_dich == "gaming" and wm.uu_tien == "quiet",
            # BUG1 FIX: bỏ wm.explanation.append() khỏi action — engine sẽ ghi explanation
            action=lambda wm: wm.__dict__.update({
                "cooler_type": "air-high",   # Noctua, be quiet! — yên nhưng hiệu quả
            }),
            explanation=(
                "Gaming + quiet → override cooler sang air-high tier. "
                "Noctua NH-D15 hoặc be quiet! Dark Rock 4 hiệu năng ngang AIO 240mm "
                "nhưng không có tiếng bơm và độ bền cao hơn."
            ),
        ),

        # ─────────────────────────────────────────────────────────
        # NHÓM 3 — ĐỒ HỌA / WORKSTATION
        # Ưu tiên: VRAM nhiều, CPU nhiều nhân, RAM lớn
        # ─────────────────────────────────────────────────────────

        Rule(
            id="R12", priority=95,
            name="Đồ họa nhẹ / Sinh viên thiết kế (< 20 triệu)",
            condition=lambda wm: wm.muc_dich == "graphics" and wm.ngan_sach < BUDGET_GRAPHICS_LOW,
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "mid-range",
                "gpu_tier":       "gaming-mid",   # VRAM 8-12GB đủ cho Photoshop, Illustrator
                "ram_capacity":   32,              # RAM quan trọng hơn GPU với đồ họa 2D
                "ram_type":       "DDR4",
                "storage_config": "nvme-only",
                "form_factor":    "ATX",
                "cooler_type":    "air-budget",
                "psu_tier":       "mid",
            }),
            explanation=(
                "Đồ họa 2D/nhẹ < 20tr → GPU gaming-mid đủ cho Photoshop, Illustrator, "
                "Premiere nhẹ. Ưu tiên RAM 32GB vì phần mềm đồ họa rất ngốn RAM. "
                "NVMe cho tốc độ mở/lưu file lớn."
            ),
        ),

        Rule(
            id="R13", priority=93,
            name="Đồ họa 3D / Video editing (20–40 triệu)",
            condition=lambda wm: (
                wm.muc_dich == "graphics"
                and BUDGET_GRAPHICS_LOW <= wm.ngan_sach < BUDGET_GRAPHICS_MID
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "high-end",     # Nhiều nhân cho render
                "gpu_tier":       "gaming-high",  # VRAM 12-16GB cho 3D, video 4K
                "ram_capacity":   32,
                "ram_type":       "DDR5",
                "storage_config": "nvme+hdd",     # NVMe làm việc, HDD lưu project
                "form_factor":    "ATX",
                "cooler_type":    "air-mid",
                "psu_tier":       "high",
            }),
            explanation=(
                "Đồ họa 3D/video 20–40tr → CPU high-end nhiều nhân (Ryzen 7, i7) "
                "cho render nhanh hơn. GPU gaming-high VRAM ≥ 12GB cho Blender, "
                "DaVinci Resolve 4K. NVMe + HDD lưu trữ project lớn."
            ),
        ),

        Rule(
            id="R14", priority=91,
            name="Workstation chuyên nghiệp (40–70 triệu)",
            condition=lambda wm: (
                wm.muc_dich == "graphics"
                and BUDGET_GRAPHICS_MID <= wm.ngan_sach < BUDGET_GRAPHICS_HIGH
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "ultra",         # Ryzen 9, i9 — nhiều nhân nhất
                "gpu_tier":       "workstation",   # RTX 4080/4090 VRAM 16-24GB
                "ram_capacity":   64,
                "ram_type":       "DDR5",
                "storage_config": "nvme+hdd",
                "form_factor":    "ATX",
                "cooler_type":    "aio-360",
                "psu_tier":       "ultra",
            }),
            explanation=(
                "Workstation chuyên nghiệp 40–70tr → CPU ultra nhiều nhân cho render farm. "
                "GPU workstation VRAM 16-24GB cho AI/ML, VFX, 3D animation phức tạp. "
                "RAM 64GB bắt buộc cho project lớn. AIO 360mm vì CPU chạy 100% liên tục lúc render."
            ),
        ),

        Rule(
            id="R15", priority=89,
            name="AI / Machine Learning workstation (>= 70 triệu)",
            # C1 FIX: dùng BUDGET_GRAPHICS_HIGH thay vì BUDGET_GRAPHICS_ULTRA (đã xóa)
            condition=lambda wm: wm.muc_dich == "graphics" and wm.ngan_sach >= BUDGET_GRAPHICS_HIGH,
            action=lambda wm: (
                wm.__dict__.update({
                    "cpu_tier":       "ultra",
                    "gpu_tier":       "workstation",
                    "ram_capacity":   64,
                    "ram_type":       "DDR5",
                    "storage_config": "nvme+hdd",
                    "form_factor":    "ATX",
                    "cooler_type":    "aio-360",
                    "psu_tier":       "ultra",
                }),
                wm.warnings.append(
                    "Build AI/ML: Cân nhắc RTX 4090 (24GB VRAM) — "
                    "VRAM là bottleneck chính khi train model lớn."
                )
            ),
            explanation=(
                "AI/ML >= 70tr → RTX 4090 là lựa chọn tốt nhất cho train model. "
                "24GB VRAM cho phép fine-tune LLM 7B-13B parameters. "
                "CPU ultra nhiều nhân cho data preprocessing song song."
            ),
        ),

        # ─────────────────────────────────────────────────────────
        # NHÓM 4 — STREAMING / CONTENT CREATION
        # Ưu tiên: CPU nhiều nhân (encode), GPU gaming, RAM lớn
        # ─────────────────────────────────────────────────────────

        Rule(
            id="R16", priority=90,
            name="Streaming + Gaming (20–35 triệu)",
            # C2 FIX: thêm lower bound BUDGET_STREAMING_MID để R16 không bắt < 20tr
            condition=lambda wm: (
                wm.muc_dich == "streaming"
                and BUDGET_STREAMING_MID <= wm.ngan_sach < BUDGET_STREAMING_HIGH
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "high-end",     # Nhiều nhân: nhân chơi game + nhân encode
                "gpu_tier":       "gaming-mid",
                "ram_capacity":   32,             # Stream ngốn thêm 4-8GB RAM
                "ram_type":       "DDR5",
                "storage_config": "nvme+hdd",     # HDD lưu VOD recording
                "form_factor":    "ATX",
                "cooler_type":    "air-mid",
                "psu_tier":       "mid",
            }),
            explanation=(
                "Streaming + gaming 20–35tr → CPU high-end bắt buộc (≥8 nhân) vì "
                "cần nhân riêng để encode stream (OBS) trong khi nhân khác chạy game. "
                "RAM 32GB vì stream + game + Discord + browser cùng lúc dễ ngốn 20GB+."
            ),
        ),

        Rule(
            id="R17", priority=88,
            name="Streaming chuyên nghiệp (> 35 triệu)",
            condition=lambda wm: (
                wm.muc_dich == "streaming" and wm.ngan_sach >= BUDGET_STREAMING_HIGH
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "ultra",
                "gpu_tier":       "gaming-high",
                "ram_capacity":   32,
                "ram_type":       "DDR5",
                "storage_config": "nvme+hdd",
                "form_factor":    "ATX",
                "cooler_type":    "aio-240",
                "psu_tier":       "high",
            }),
            explanation=(
                "Streaming chuyên nghiệp > 35tr → CPU ultra (Ryzen 9, i9) nhiều nhân "
                "cho encode AV1/HEVC chất lượng cao. GPU gaming-high cho chơi game 1440p "
                "trong khi stream 1080p60 không drop frame."
            ),
        ),

        # ─────────────────────────────────────────────────────────
        # NHÓM 5 — LUẬT ĐIỀU CHỈNH THEO ƯU TIÊN (Override rules)
        # Các luật này KHÔNG ghi đè toàn bộ WM mà chỉ điều chỉnh
        # một số field cụ thể sau khi nhóm 1-4 đã chạy
        # ─────────────────────────────────────────────────────────

        Rule(
            id="R18", priority=70,
            name="Ưu tiên hiệu năng → nâng GPU tier",
            condition=lambda wm: (
                wm.uu_tien == "performance"
                and wm.gpu_tier in ("budget", "gaming-mid")
                and wm.ngan_sach >= BUDGET_PERF_OVERRIDE
            ),
            # BUG1+BUG2 FIX: bỏ wm.explanation.append() + không còn read-after-write.
            # .get(wm.gpu_tier, ...) được evaluate TRƯỚC khi update(), nên đọc giá trị cũ đúng.
            action=lambda wm: wm.__dict__.update({
                "gpu_tier": {
                    "budget":     "gaming-mid",
                    "gaming-mid": "gaming-high",
                }.get(wm.gpu_tier, wm.gpu_tier)
            }),
            explanation=(
                "Ưu tiên performance + ngân sách đủ → nâng GPU lên tier cao hơn (budget→gaming-mid, "
                "gaming-mid→gaming-high). Trade-off: có thể vượt ngân sách nhẹ, "
                "CSP checker sẽ cân bằng lại."
            ),
        ),

        Rule(
            id="R19", priority=68,
            name="Ưu tiên tiết kiệm → tối ưu linh kiện",
            condition=lambda wm: wm.uu_tien == "value",
            # BUG1 FIX: bỏ wm.explanation.append() khỏi action
            # Không có read-after-write: wm.ram_type / cooler_type đọc khi build dict argument,
            # tức là TRƯỚC khi update() được gọi → an toàn.
            action=lambda wm: wm.__dict__.update({
                "ram_type":    "DDR4" if wm.ram_type == "DDR5" else wm.ram_type,
                "cooler_type": "air-budget" if wm.cooler_type in ("air-mid", "aio-240") else wm.cooler_type,
            }),
            explanation=(
                "Ưu tiên value/tiết kiệm → downgrade RAM về DDR4 (tiết kiệm 20-30%) "
                "và cooler về air-budget (tiết kiệm 300-500k). "
                "Hiệu năng thực tế chênh lệch không đáng kể, "
                "ngân sách tiết kiệm dành cho GPU hoặc storage tốt hơn."
            ),
        ),

        Rule(
            id="R20", priority=65,
            name="Cảnh báo ngân sách không thực tế",
            condition=lambda wm: (
                wm.muc_dich == "gaming"
                and wm.gpu_tier in ("gaming-high", "workstation")
                and wm.ngan_sach < BUDGET_GAMING_WARNING
            ),
            action=lambda wm: (
                wm.warnings.append(
                    "💡 Với ngân sách hiện tại, hệ thống đề xuất cấu hình gaming tầm trung. "
                    "Để có hiệu năng cao hơn, bạn có thể tăng ngân sách lên 25–30tr."
                ),
                wm.__dict__.update({
                    "gpu_tier": "gaming-mid"
                })
            ),
            explanation=(
                "Ngân sách chưa đủ cho cấu hình gaming cao cấp — "
                "tự động điều chỉnh sang gaming tầm trung để tìm được kết quả phù hợp."
            ),
        ),

        # ─────────────────────────────────────────────────────────
        # NHÓM 6 — STUDY / EDITING
        # ─────────────────────────────────────────────────────────

        Rule(
            id="R21", priority=89,
            name="Study / Lập trình ngân sách thấp (< 15 triệu)",
            condition=lambda wm: wm.muc_dich == "study" and wm.ngan_sach < BUDGET_STUDY_LOW,
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "mid-range",   # CPU nhiều nhân quan trọng hơn GPU
                "gpu_tier":       "none",         # iGPU đủ dùng, không cần GPU rời
                "ram_capacity":   16,
                "ram_type":       "DDR4",
                "storage_config": "nvme-only",   # NVMe giảm thời gian compile, build Docker
                "form_factor":    "mATX",
                "cooler_type":    "stock",
                "psu_tier":       "basic",
            }),
            explanation=(
                "Sinh viên lập trình < 15tr → CPU mid-range nhiều nhân để chạy "
                "IDE/Docker/VM song song. Không cần GPU rời, tiết kiệm ngân sách dành cho CPU. "
                "16GB DDR4 đủ cho lập trình thông thường. NVMe nhanh hơn cho compile/build."
            ),
        ),

        Rule(
            id="R22", priority=87,
            name="Study / Lập trình ngân sách trung (15–25 triệu)",
            # Dùng BUDGET_STUDY_HIGH thay BUDGET_STUDY_MID (đã đổi tên cho nhất quán)
            condition=lambda wm: (
                wm.muc_dich == "study"
                and BUDGET_STUDY_LOW <= wm.ngan_sach < BUDGET_STUDY_HIGH
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "mid-range",
                "gpu_tier":       "budget",       # GPU budget hỗ trợ học ML cơ bản (CUDA)
                "ram_capacity":   32,             # 32GB cho chạy nhiều Docker container / máy ảo
                "ram_type":       "DDR4",
                "storage_config": "nvme-only",
                "form_factor":    "mATX",
                "cooler_type":    "air-budget",
                "psu_tier":       "mid",
            }),
            explanation=(
                "Sinh viên lập trình 15–25tr → nâng RAM lên 32GB để chạy máy ảo, "
                "nhiều Docker container song song mà không bị thiếu RAM. "
                "GPU budget hỗ trợ học ML cơ bản (PyTorch, TensorFlow với CUDA). "
                "air-budget cooler cho CPU mid-range tản nhiệt bền bỉ hơn khi compile nặng."
            ),
        ),

        Rule(
            id="R23", priority=88,
            name="Editing / Dựng phim ngân sách trung (< 30 triệu)",
            condition=lambda wm: wm.muc_dich == "editing" and wm.ngan_sach < BUDGET_EDITING_MID,
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "high-end",     # CPU nhiều nhân cho export nhanh
                "gpu_tier":       "gaming-mid",   # GPU gaming-mid cho hardware encode/decode
                "ram_capacity":   32,
                "ram_type":       "DDR4",
                "storage_config": "nvme+hdd",     # HDD bắt buộc để lưu footage thô dung lượng lớn
                "form_factor":    "ATX",
                "cooler_type":    "air-mid",
                "psu_tier":       "mid",
            }),
            explanation=(
                "Dựng phim < 30tr → CPU high-end nhiều nhân cho export video nhanh "
                "hơn đáng kể (render H.264/H.265 dùng nhiều nhân). "
                "HDD bắt buộc để lưu footage thô — 1 giờ video 4K RAW có thể 50-200GB. "
                "GPU gaming-mid đủ cho hardware encode NVENC/AMF."
            ),
        ),

        Rule(
            id="R24", priority=86,
            name="Editing / Dựng phim cao cấp (>= 30 triệu)",
            condition=lambda wm: wm.muc_dich == "editing" and wm.ngan_sach >= BUDGET_EDITING_MID,
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "ultra",         # Ryzen 9 / i9 để export nhanh nhất
                "gpu_tier":       "gaming-high",   # GPU gaming-high cho hardware encode 4K/8K
                "ram_capacity":   64,              # 64GB cho timeline 4K nhiều track + effects
                "ram_type":       "DDR5",
                "storage_config": "nvme+hdd",
                "form_factor":    "ATX",
                "cooler_type":    "aio-240",       # AIO vì CPU chạy 100% khi export lâu
                "psu_tier":       "high",
            }),
            explanation=(
                "Dựng phim cao cấp >= 30tr → RAM 64GB DDR5 cần thiết cho timeline 4K "
                "nhiều track video + audio + effects cùng lúc. "
                "GPU gaming-high cho hardware encode/decode 4K/8K (NVENC tier cao). "
                "AIO 240mm vì CPU ultra chạy tải cao liên tục khi export."
            ),
        ),

        # ─────────────────────────────────────────────────────────
        # NHÓM 7 — OVERRIDE MỚI (R25–R27)
        # Priority 60-70: chạy sau các rule chính để override cụ thể
        # ─────────────────────────────────────────────────────────

        Rule(
            id="R25", priority=65,
            name="Gaming + ưu tiên compact → mATX thay ITX",
            condition=lambda wm: wm.muc_dich == "gaming" and wm.uu_tien == "compact",
            # BUG1 FIX: bỏ wm.explanation.append(), giữ nguyên wm.warnings.append()
            action=lambda wm: (
                wm.__dict__.update({
                    "form_factor": "mATX",   # Không dùng ITX vì GPU gaming dài khó fit
                }),
                wm.warnings.append(
                    "ITX case giới hạn GPU dài, dùng mATX để có nhiều lựa chọn GPU gaming hơn."
                ),
            ),
            explanation=(
                "Gaming + compact → dùng mATX thay vì ITX. "
                "GPU gaming tầm mid-high thường dài 280-340mm, nhiều ITX case không hỗ trợ. "
                "mATX vẫn nhỏ gọn hơn ATX nhưng cho nhiều lựa chọn GPU hơn."
            ),
        ),

        Rule(
            id="R26", priority=64,
            name="Đồ họa/Streaming + yên tĩnh → air-high cooler",
            condition=lambda wm: (
                wm.muc_dich in ("graphics", "streaming") and wm.uu_tien == "quiet"
            ),
            # BUG1 FIX: bỏ wm.explanation.append() khỏi action
            action=lambda wm: wm.__dict__.update({
                "cooler_type": "air-high",   # Yên hơn AIO khi render/encode dài giờ
            }),
            explanation=(
                "Đồ họa/streaming + quiet → override cooler sang air-high. "
                "Khi render hoặc encode stream liên tục nhiều giờ, AIO pump chạy ồn liên tục. "
                "Noctua NH-D15 / be quiet! Dark Rock yên hơn, hiệu năng ngang AIO 240mm."
            ),
        ),

        Rule(
            id="R27", priority=62,
            name="Đồ họa + ưu tiên value → tối ưu RAM/Storage cho VRAM",
            condition=lambda wm: wm.muc_dich == "graphics" and wm.uu_tien == "value",
            # BUG1 FIX: bỏ wm.explanation.append() khỏi action
            action=lambda wm: wm.__dict__.update({
                # Downgrade DDR5 → DDR4: tiết kiệm ~2-4tr, hiệu năng đồ họa không đổi nhiều
                "ram_type": "DDR4" if wm.ram_type == "DDR5" else wm.ram_type,
                # Downgrade storage nếu ngân sách thấp để dành tiền cho GPU VRAM
                "storage_config": (
                    "nvme-only"
                    if wm.storage_config == "nvme+hdd" and wm.ngan_sach < BUDGET_EDITING_MID
                    else wm.storage_config
                ),
            }),
            explanation=(
                "Đồ họa + value → tiết kiệm RAM và storage để dành ngân sách cho GPU VRAM nhiều hơn. "
                "VRAM là bottleneck chính của workload đồ họa (Blender, Stable Diffusion, video 4K). "
                "DDR4 vs DDR5 chênh lệch không đáng kể với workload render."
            ),
        ),

        # ─────────────────────────────────────────────────────────
        # NHÓM 8 — GAP COVERAGE BỔ SUNG (R28–R31)
        # Các kịch bản bị hở được phát hiện qua đánh giá coverage
        # ─────────────────────────────────────────────────────────

        Rule(
            id="R28", priority=86,
            name="Study / Lập trình ngân sách cao (>= 25 triệu)",
            condition=lambda wm: (
                wm.muc_dich == "study" and wm.ngan_sach >= BUDGET_STUDY_HIGH
            ),
            action=lambda wm: wm.__dict__.update({
                "cpu_tier":       "high-end",    # Nhiều nhân cho compile nhanh, chạy nhiều VM
                "gpu_tier":       "budget",       # Đủ cho học PyTorch/CUDA cơ bản
                "ram_capacity":   32,
                "ram_type":       "DDR5",
                "storage_config": "nvme+hdd",    # HDD lưu dataset, project lớn
                "form_factor":    "ATX",
                "cooler_type":    "air-mid",
                "psu_tier":       "mid",
            }),
            explanation=(
                "Study >= 25tr → high-end CPU nhiều nhân cho compile nhanh, "
                "chạy nhiều máy ảo song song. RAM 32GB DDR5 cho Docker + IDE + browser "
                "cùng lúc không bị thiếu. GPU budget đủ cho học PyTorch/CUDA cơ bản. "
                "HDD lưu dataset/project lớn khi học ML."
            ),
        ),

        Rule(
            id="R29", priority=72,
            name="Office + yên tĩnh → nâng cooler",
            condition=lambda wm: wm.muc_dich == "office" and wm.uu_tien == "quiet",
            action=lambda wm: wm.__dict__.update({
                "cooler_type": "air-budget",   # Tản nhiệt tốt hơn stock, ít tiếng ồn hơn
            }),
            explanation=(
                "Office + yên tĩnh → dùng cooler air-budget thay stock. "
                "Tản nhiệt tốt hơn, ít tiếng ồn hơn stock đi kèm. "
                "Phù hợp môi trường làm việc yêu cầu không gian yên tĩnh."
            ),
        ),

        Rule(
            id="R30", priority=70,
            name="Editing + yên tĩnh → air-high cooler",
            condition=lambda wm: wm.muc_dich == "editing" and wm.uu_tien == "quiet",
            action=lambda wm: wm.__dict__.update({
                "cooler_type": "air-high",   # Render dài giờ cần mát, không tiếng bơm AIO
            }),
            explanation=(
                "Editing + yên tĩnh → air-high tier (Noctua NH-D15, be quiet! Dark Rock) "
                "thay AIO. Render video dài giờ cần tản nhiệt tốt nhưng "
                "không có tiếng bơm AIO khi làm việc trong phòng yên tĩnh."
            ),
        ),

        Rule(
            id="R31", priority=91,
            name="Streaming ngân sách thấp (< 20 triệu) — cảnh báo",
            condition=lambda wm: (
                wm.muc_dich == "streaming" and wm.ngan_sach < BUDGET_STREAMING_MID
            ),
            action=lambda wm: (
                wm.warnings.append(
                    "Ngân sách < 20tr quá thấp cho streaming chuyên nghiệp. "
                    "Khuyến nghị tối thiểu 20tr để có cấu hình stream ổn định."
                ),
                wm.__dict__.update({
                    "cpu_tier":       "mid-range",   # mid-range có đủ nhân để encode cơ bản
                    "gpu_tier":       "gaming-mid",
                    "ram_capacity":   16,
                    "ram_type":       "DDR4",
                    "storage_config": "nvme+hdd",    # HDD lưu VOD dù ngân sách thấp
                    "form_factor":    "ATX",
                    "cooler_type":    "air-budget",
                    "psu_tier":       "mid",
                }),
            ),
            explanation=(
                "Streaming < 20tr — ngân sách quá thấp cho setup stream ổn định. "
                "Cấu hình cơ bản với mid-range CPU vẫn đủ encode ở chất lượng thấp. "
                "Cần nâng ngân sách lên 20tr+ để có trải nghiệm stream tốt hơn."
            ),
        ),
    ]


# ══════════════════════════════════════════════════════════════════
# PUBLIC API — Inference Engine gọi hàm này
# ══════════════════════════════════════════════════════════════════
def get_all_rules() -> list[Rule]:
    """Trả về tất cả rules đã sắp xếp theo priority giảm dần."""
    return sorted(_rules(), key=lambda r: r.priority, reverse=True)


def get_rule_by_id(rule_id: str) -> Rule | None:
    return next((r for r in _rules() if r.id == rule_id), None)


# ══════════════════════════════════════════════════════════════════
# QUICK TEST — chạy trực tiếp để kiểm tra KB
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from inference_engine import ForwardChaining

    test_cases = [
        {"ngan_sach":  7_000_000, "muc_dich": "office",    "uu_tien": "value",       "label": "Test 1 — Office 7tr + value"},
        {"ngan_sach": 20_000_000, "muc_dich": "gaming",    "uu_tien": "performance", "label": "Test 2 — Gaming 20tr + performance"},
        {"ngan_sach": 35_000_000, "muc_dich": "graphics",  "uu_tien": "quiet",       "label": "Test 3 — Graphics 35tr + quiet"},
        {"ngan_sach": 15_000_000, "muc_dich": "study",     "uu_tien": "value",       "label": "Test 4 — Study 15tr + value"},
        {"ngan_sach": 30_000_000, "muc_dich": "study",     "uu_tien": "performance", "label": "Test 5 — Study 30tr + performance"},
        {"ngan_sach": 25_000_000, "muc_dich": "editing",   "uu_tien": "quiet",       "label": "Test 6 — Editing 25tr + quiet"},
        {"ngan_sach": 10_000_000, "muc_dich": "streaming", "uu_tien": "value",       "label": "Test 7 — Streaming 10tr + value"},
        {"ngan_sach": 0,          "muc_dich": "GAMING",    "uu_tien": "",            "label": "Test 8 — Invalid: ngan_sach=0, muc_dich='GAMING'"},
    ]

    engine = ForwardChaining()
    all_passed = True

    for tc in test_cases:
        try:
            wm = WorkingMemory(
                ngan_sach=tc["ngan_sach"],
                muc_dich=tc["muc_dich"],
                uu_tien=tc["uu_tien"],
            )
            result = engine.run(wm)

            # Kiểm tra explanation không bị double (phải = số rule đã fire)
            expected_exp_count = len(result.fired_rules)
            actual_exp_count   = len(result.explanation)
            double_ok = actual_exp_count == expected_exp_count

            print(f"\n{'─'*60}")
            print(f"{'✅' if double_ok else '❌'} {tc['label']}")
            print(f"   fired_rules  : {result.fired_rules}")
            print(f"   CPU tier     : {result.cpu_tier}")
            print(f"   GPU tier     : {result.gpu_tier}")
            print(f"   RAM          : {result.ram_capacity}GB {result.ram_type}")
            print(f"   Cooler       : {result.cooler_type}  (max {result.cooler_tdp_max}W TDP)")
            print(f"   PSU tier     : {result.psu_tier}  ({result.psu_wattage_min}W)")
            print(f"   Explanation  : {actual_exp_count} dòng (fired={expected_exp_count})"
                  f"  {'✅ OK' if double_ok else '❌ DOUBLE BUG!'}")
            if result.warnings:
                print(f"   ⚠ Warnings  : {result.warnings}")
            if not double_ok:
                all_passed = False

        except Exception as e:
            print(f"\n{'─'*60}")
            print(f"❌ {tc['label']} — EXCEPTION: {e}")
            all_passed = False

    print(f"\n{'═'*60}")
    print(f"{'✅ TẤT CẢ TEST PASS' if all_passed else '❌ CÓ TEST FAIL — kiểm tra lại!'}")
