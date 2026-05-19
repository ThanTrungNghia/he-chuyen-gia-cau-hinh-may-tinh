"""
inference_engine.py — Bộ xử lý chạy toàn bộ các luật tư vấn
==============================================================
Nhận thông tin người dùng → kiểm tra hợp lệ → chạy từng luật → trả về kết quả.

Các bước trong hàm run():
  [0] validate_input()   — kiểm tra và chuẩn hóa thông tin đầu vào
  [1] Chạy luật          — lặp liên tục cho đến khi không còn luật nào áp dụng được
  [2] _apply_defaults()  — điền giá trị mặc định cho các trường còn trống
  [3] _enrich_output()   — tra bảng để điền công suất nguồn, tản nhiệt, dung lượng ổ
"""

from knowledge_base import (
    WorkingMemory,
    Rule,
    get_all_rules,
    PSU_TIER_WATTAGE,
    COOLER_TDP_SUPPORT,
    STORAGE_CAPACITY,
)


class ForwardChaining:

    def __init__(self):
        self.rules: list[Rule] = get_all_rules()  # đã sort theo priority giảm dần

    # ──────────────────────────────────────────────────────────────
    # BƯỚC 0 — Kiểm tra và chuẩn hóa thông tin đầu vào
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def validate_input(wm: WorkingMemory) -> list[str]:
        """
        Kiểm tra thông tin người dùng nhập vào có hợp lệ không.

        Tự động chuẩn hóa: chuyển muc_dich và uu_tien về chữ thường, bỏ khoảng trắng thừa.
        Trả về danh sách thông báo lỗi (danh sách rỗng = không có lỗi nào).
        """
        errors: list[str] = []

        # [BUG4] Type safety cho ngan_sach — None hoặc chuỗi không hợp lệ sẽ gây
        # TypeError khi so sánh số học; ép về int trước mọi kiểm tra
        try:
            wm.ngan_sach = int(wm.ngan_sach or 0)
        except (TypeError, ValueError):
            errors.append(
                f"ngan_sach '{wm.ngan_sach}' không phải số hợp lệ. "
                f"Hệ thống sẽ tiếp tục với giá trị đã nhập — kết quả có thể không chính xác."
            )
            wm.ngan_sach = 0

        # Chuẩn hóa string — tránh lỗi do người dùng nhập "Office" hay " gaming "
        wm.muc_dich = wm.muc_dich.lower().strip()
        wm.uu_tien  = wm.uu_tien.lower().strip()

        valid_muc_dich = {"office", "gaming", "graphics", "streaming", "study", "editing"}
        valid_uu_tien  = {"performance", "value", "quiet", "compact"}

        # Kiểm tra muc_dich
        if wm.muc_dich not in valid_muc_dich:
            errors.append(
                f"muc_dich '{wm.muc_dich}' không hợp lệ. "
                f"Phải là một trong: {', '.join(sorted(valid_muc_dich))}. "
                f"Hệ thống sẽ tiếp tục với giá trị đã nhập — kết quả có thể không chính xác."  # [BUG3]
            )

        # Kiểm tra uu_tien (được phép để trống — hệ thống vẫn chạy được)
        if wm.uu_tien and wm.uu_tien not in valid_uu_tien:
            errors.append(
                f"uu_tien '{wm.uu_tien}' không hợp lệ. "
                f"Phải là một trong: {', '.join(sorted(valid_uu_tien))}. "
                f"Ưu tiên sẽ bị bỏ qua."
            )

        # Kiểm tra ngân sách
        if wm.ngan_sach <= 0:
            errors.append(
                f"ngan_sach phải lớn hơn 0 (hiện tại: {wm.ngan_sach:,} VNĐ). "
                f"Hệ thống sẽ tiếp tục với giá trị đã nhập — kết quả có thể không chính xác."  # [BUG3]
            )
        elif wm.ngan_sach < 5_000_000:
            # Cảnh báo nhưng vẫn tiếp tục — R01 có thể xử lý nếu muc_dich=office
            errors.append(
                f"⚠ Cảnh báo: ngân sách {wm.ngan_sach:,} VNĐ quá thấp "
                f"(< 5 triệu) — khó tư vấn được cấu hình thực tế."
            )

        return errors

    # ──────────────────────────────────────────────────────────────
    # BƯỚC 1 — Forward Chaining (fixpoint iteration)
    # ──────────────────────────────────────────────────────────────
    def run(self, wm: WorkingMemory) -> WorkingMemory:
        """
        Hàm chạy toàn bộ quá trình tư vấn từ thông tin người dùng.

        Tham số:
            wm: đối tượng chứa ngân sách, mục đích, ưu tiên của người dùng
        Trả về:
            wm đã được điền đầy đủ thông tin (tier linh kiện, ngân sách, ...)

        Các bước:
          - Bước 0: kiểm tra thông tin đầu vào
          - Bước 1: lặp liên tục cho đến khi không còn luật nào được áp dụng thêm
          - Bước 2: điền giá trị mặc định cho trường còn trống
          - Bước 3: tra bảng để điền công suất, TDP, dung lượng chi tiết
        """

        # ── Bước 0: Validate input ────────────────────────────────
        errors = self.validate_input(wm)
        if errors:
            # Ghi lỗi vào warnings và tiếp tục chạy với giá trị hiện có
            # _apply_defaults() sẽ đảm bảo output không bị trống
            for err in errors:
                wm.warnings.append(err)

        # ── Bước 1: Lặp qua các luật cho đến khi không còn luật nào khớp ──
        fired_ids: set[str] = set()
        changed = True

        while changed:
            changed = False
            for rule in self.rules:
                if rule.id in fired_ids:
                    continue

                # Kiểm tra điều kiện của luật, bắt lỗi để không làm sập chương trình
                try:
                    cond_met = rule.condition(wm)
                except Exception:
                    cond_met = False

                if cond_met:
                    # Lưu trạng thái trước khi áp dụng luật để biết có thay đổi không
                    before = {
                        k: v for k, v in wm.__dict__.items()
                        if k not in ("explanation", "fired_rules", "warnings")
                    }

                    # Áp dụng hành động của luật
                    try:
                        rule.action(wm)
                    except Exception as e:
                        wm.warnings.append(f"Rule {rule.id} lỗi action: {e}")
                        fired_ids.add(rule.id)
                        continue

                    # Ghi lại luật đã chạy và lý do (chỉ engine ghi, không phải action)
                    wm.fired_rules.append(rule.id)
                    wm.explanation.append(
                        f"[{rule.id}] {rule.name}: {rule.explanation}"
                    )
                    fired_ids.add(rule.id)

                    # Nếu có thay đổi thì cần chạy lại vòng lặp để kiểm tra luật khác
                    after = {
                        k: v for k, v in wm.__dict__.items()
                        if k not in ("explanation", "fired_rules", "warnings")
                    }
                    if before != after:
                        changed = True

        # ── Bước 2: Apply defaults ────────────────────────────────
        self._apply_defaults(wm)

        # ── Bước 3: Enrich output với thông tin chi tiết ──────────
        self._enrich_output(wm)

        return wm

    # ──────────────────────────────────────────────────────────────
    # BƯỚC 2 — Apply defaults
    # ──────────────────────────────────────────────────────────────
    def _apply_defaults(self, wm: WorkingMemory):
        """Điền giá trị mặc định vào các trường còn trống, tránh lỗi khi hiển thị kết quả."""
        if not wm.cpu_tier:       wm.cpu_tier       = "mid-range"
        if not wm.gpu_tier:       wm.gpu_tier        = "none"       # [BUG5] 'none' an toàn hơn vì CSP sẽ tự chọn GPU phù hợp theo muc_dich
        if not wm.ram_capacity:   wm.ram_capacity    = 16
        if not wm.ram_type:       wm.ram_type        = "DDR4"
        if not wm.storage_config: wm.storage_config  = "ssd-only"
        if not wm.form_factor:    wm.form_factor     = "ATX"
        if not wm.cooler_type:    wm.cooler_type     = "air-budget"
        if not wm.psu_tier:       wm.psu_tier        = "mid"

        # [U3] Enrich fields — _enrich_output() sẽ ghi đè ngay sau,
        # nhưng đặt default 0 ở đây đảm bảo không có AttributeError
        # nếu ai đó truy cập WM trước khi _enrich_output() chạy
        if not wm.psu_wattage_min:   wm.psu_wattage_min   = 0
        if not wm.cooler_tdp_max:    wm.cooler_tdp_max    = 0
        if not wm.nvme_capacity_gb:  wm.nvme_capacity_gb  = 0
        if not wm.hdd_capacity_gb:   wm.hdd_capacity_gb   = 0

        if not wm.fired_rules:
            wm.warnings.append("Không có rule nào khớp — dùng cấu hình mặc định.")

    # ──────────────────────────────────────────────────────────────
    # BƯỚC 3 — Enrich output
    # ──────────────────────────────────────────────────────────────
    def _enrich_output(self, wm: WorkingMemory):
        """
        Tra bảng để điền thêm thông tin chi tiết sau khi đã chạy xong các luật.
        Ví dụ: từ psu_tier="mid" → điền psu_wattage_min=650W.
        """
        # Tra công suất nguồn theo tier PSU
        wm.psu_wattage_min = PSU_TIER_WATTAGE.get(wm.psu_tier, 0)

        # Tra TDP tối đa cooler có thể tản
        wm.cooler_tdp_max = COOLER_TDP_SUPPORT.get(wm.cooler_type, 0)

        # Tra dung lượng storage theo kiểu cấu hình
        storage_info = STORAGE_CAPACITY.get(wm.storage_config, {})
        wm.nvme_capacity_gb = storage_info.get("nvme_gb", 0)
        wm.hdd_capacity_gb  = storage_info.get("hdd_gb", 0)
