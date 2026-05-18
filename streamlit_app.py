"""
streamlit_app.py — UI demo Hệ Chuyên Gia Tư Vấn Cấu Hình Máy Tính
====================================================================
Pipeline: Suy luận tự động → Kiểm tra ràng buộc → Tối ưu hóa lựa chọn
Chat AI: nhận diện ngân sách, mục đích, ưu tiên từ câu tự nhiên.
Chạy: streamlit run streamlit_app.py
"""

import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from knowledge_base import WorkingMemory
from forward_chaining import run_forward_chaining
from csp_checker import load_data, filter_with_fallback, csp_with_forward_checking
from astar_selector import astar_select, explain_score, performance_score, WEIGHTS
from predicate_extractor import extract_predicates, explain_predicates


# ══════════════════════════════════════════════════════════════════
# CONSTANTS — màu sắc và icon cho từng linh kiện
# ══════════════════════════════════════════════════════════════════

COMPONENT_COLORS = {
    "cpu":       "#3498db",
    "mainboard": "#27ae60",
    "ram":       "#e67e22",
    "vga":       "#e74c3c",
    "psu":       "#9b59b6",
    "storage":   "#f39c12",
    "case":      "#7f8c8d",
    "cooler":    "#1abc9c",
}
COMPONENT_ICONS = {
    "cpu":       "🔧", "mainboard": "🖥️", "ram": "💾",
    "vga":       "🎮", "psu":       "⚡", "storage": "💿",
    "case":      "📦", "cooler":    "❄️",
}
MUC_DICH_LABELS = {
    "office":    "💼 Văn phòng",
    "gaming":    "🎮 Gaming",
    "graphics":  "🎨 Đồ họa",
    "streaming": "📹 Streaming",
    "study":     "📚 Học tập / Lập trình",
    "editing":   "🎬 Dựng phim",
}
UU_TIEN_LABELS = {
    "":            "Không ưu tiên (cân bằng)",
    "performance": "⚡ Hiệu năng",
    "value":       "💰 Tiết kiệm",
    "quiet":       "🔇 Yên tĩnh",
    "compact":     "📦 Nhỏ gọn",
}
LINH_KIEN_LABELS = {
    "cpu":       "CPU",           "mainboard": "Mainboard",
    "ram":       "RAM",           "vga":       "Card đồ họa",
    "psu":       "Nguồn (PSU)",   "storage":   "Ổ cứng",
    "case":      "Vỏ máy (Case)", "cooler":    "Tản nhiệt",
}

RULE_GROUP_COLORS = {
    "R0": "#3498db", "R1": "#27ae60", "R2": "#e67e22",
    "R3": "#e74c3c", "R4": "#9b59b6",
}


# ══════════════════════════════════════════════════════════════════
# CSS INJECTION — custom styling toàn bộ app
# ══════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ── Tổng thể ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }

/* ── Metric cards ──────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px; padding: 12px 16px;
    border-left: 4px solid #3498db;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

/* ── Component table rows ─────────────────────────────── */
.comp-badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; color: white;
    font-size: 12px; font-weight: 600; white-space: nowrap;
}
.comp-table { width: 100%; border-collapse: collapse; }
.comp-table tr { transition: background 0.2s; }
.comp-table tr:hover { background: rgba(255,255,255,0.03); }
.comp-table td { padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.05); }
.comp-table .price-col { font-weight: 700; text-align: right; white-space: nowrap; }
.highlight-row { background: rgba(255,215,0,0.07) !important; }

/* ── Gold card (top config) ───────────────────────────── */
.gold-card {
    border: 2px solid #ffd700;
    border-radius: 16px; padding: 20px;
    background: linear-gradient(135deg, rgba(255,215,0,0.06) 0%, rgba(255,165,0,0.03) 100%);
    box-shadow: 0 4px 16px rgba(255,215,0,0.15);
}
.silver-card {
    border: 1px solid #a8a9ad;
    border-radius: 16px; padding: 20px;
    background: rgba(168,169,173,0.05);
}
.bronze-card {
    border: 1px solid #cd7f32;
    border-radius: 16px; padding: 20px;
    background: rgba(205,127,50,0.05);
}

/* ── Chat bubbles ─────────────────────────────────────── */
.chat-user {
    background: linear-gradient(135deg, #1e3a5f, #1a2e4a);
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px; margin: 8px 0;
    border-left: 3px solid #3498db;
}
.chat-result {
    background: linear-gradient(135deg, #1a2e1a, #152315);
    border-radius: 4px 12px 12px 12px;
    padding: 12px 16px; margin: 4px 0;
    border-left: 3px solid #27ae60;
    font-size: 13px;
}
.predicate-box {
    background: rgba(52, 152, 219, 0.1);
    border: 1px solid rgba(52, 152, 219, 0.3);
    border-radius: 8px; padding: 10px 14px; margin: 6px 0;
    font-family: monospace; font-size: 13px;
}

/* ── Sidebar header ───────────────────────────────────── */
.sidebar-logo {
    text-align: center; padding: 8px 0 16px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 16px;
}

/* ── Constraint check ─────────────────────────────────── */
.constraint-pass { color: #2ecc71; font-weight: 600; }
.constraint-fail { color: #e74c3c; font-weight: 600; }

/* ── Domain size bar ──────────────────────────────────── */
.domain-bar-wrap { background: rgba(255,255,255,0.05); border-radius: 4px; height: 8px; }
.domain-bar { border-radius: 4px; height: 8px; }

/* ── Footer ───────────────────────────────────────────── */
.footer {
    text-align: center; color: #666;
    font-size: 12px; margin-top: 48px;
    padding: 16px 0; border-top: 1px solid rgba(255,255,255,0.07);
}
</style>
"""


# ══════════════════════════════════════════════════════════════════
# PIPELINE — cached data + recommend()
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_data_cached():
    return load_data()


@st.cache_data(show_spinner=False)
def _total_product_count() -> int:
    data = _load_data_cached()
    return sum(len(v) for v in data.values())


def recommend(ngan_sach: int, muc_dich: str, uu_tien: str = "") -> dict | None:
    t1 = time.time()
    wm = WorkingMemory(ngan_sach=ngan_sach, muc_dich=muc_dich, uu_tien=uu_tien)
    wm, fired = run_forward_chaining(wm)
    t_fc = time.time() - t1

    data = _load_data_cached()
    t2 = time.time()
    domains, fb_log = filter_with_fallback(data, wm)
    valid_configs = csp_with_forward_checking(domains, wm.ngan_sach, max_results=50)

    # Nới lỏng ngân sách ±15% và thử lại nếu không tìm được cấu hình
    if not valid_configs:
        _relax_wm = WorkingMemory(
            ngan_sach=int(ngan_sach * 1.15),
            muc_dich=muc_dich, uu_tien=uu_tien,
        )
        _relax_wm, _ = run_forward_chaining(_relax_wm)
        _relax_domains, _relax_fb = filter_with_fallback(data, _relax_wm)
        valid_configs = csp_with_forward_checking(
            _relax_domains, _relax_wm.ngan_sach, max_results=50
        )
        if valid_configs:
            wm      = _relax_wm
            domains = _relax_domains
            fb_log  = _relax_fb + ["[Retry] Nới ngân sách ×1.15 để tìm được cấu hình"]

    t_csp = time.time() - t2

    if not valid_configs:
        return None

    t3 = time.time()
    best, top3, best_f = astar_select(valid_configs, wm)
    t_astar = time.time() - t3

    return {
        "best":           best,
        "top3":           top3,
        "f_score":        best_f,
        "perf_score":     performance_score(best, wm),
        "fired_rules":    fired,
        "working_memory": wm,
        "valid_count":    len(valid_configs),
        "domain_sizes":   {k: len(v) for k, v in domains.items()},
        "fallback_log":   fb_log,
        "explain":        explain_score(best, wm),
        "timing":         {"fc": t_fc, "csp": t_csp, "astar": t_astar},
    }


# ══════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════

def fmt_vnd(x) -> str:
    return f"{int(x):,}đ".replace(",", ".")


def config_to_table(config: dict) -> pd.DataFrame:
    rows = []
    for key in ["cpu", "mainboard", "ram", "vga", "psu", "storage", "case", "cooler"]:
        item = config.get(key, {})
        rows.append({
            "Linh kiện": LINH_KIEN_LABELS[key],
            "Tên sản phẩm": str(item.get("name", "—"))[:80],
            "Giá": fmt_vnd(item.get("price", 0)),
            "Link": item.get("link", ""),
        })
    rows.append({
        "Linh kiện": "TỔNG",
        "Tên sản phẩm": "—",
        "Giá": fmt_vnd(config.get("total", 0)),
        "Link": "",
    })
    return pd.DataFrame(rows)


def render_config_html(config: dict, ngan_sach: int) -> str:
    """Tạo HTML table với badge màu, link mua, highlight linh kiện đắt nhất."""
    COMP_KEYS = ["cpu", "mainboard", "ram", "vga", "psu", "storage", "case", "cooler"]
    most_expensive = max(COMP_KEYS, key=lambda k: config.get(k, {}).get("price", 0))
    has_stock_cooler = False   # để thêm ghi chú bên dưới nếu dùng box cooler

    rows_html = ""
    _seen_keys: set[str] = set()
    for key in COMP_KEYS:
        if key in _seen_keys:
            continue
        _seen_keys.add(key)
        item  = config.get(key, {})
        price = item.get("price", 0)
        name  = str(item.get("name", "—"))

        # FIX 3: bỏ qua dòng cooler nếu là box cooler 0đ
        if key == "cooler" and (price == 0 or "box cooler" in name.lower()):
            has_stock_cooler = True
            continue

        if len(name) > 70:
            name = name[:70] + "…"
        color    = COMPONENT_COLORS[key]
        icon     = COMPONENT_ICONS[key]
        label    = LINH_KIEN_LABELS[key]
        hl_style = "background:rgba(255,215,0,0.07);" if key == most_expensive else ""
        hl_badge = " ⭐" if key == most_expensive else ""

        # FIX 8: link mua từ CSV (chỉ URL thực, không tạo link giả)
        link_url = str(item.get("link", "") or "").strip()
        link_html = (
            f'<a href="{link_url}" target="_blank" style="color:#3498db;font-size:12px;'
            f'text-decoration:none;">🛒 Mua ngay</a>'
            if link_url else ""
        )

        rows_html += f"""
        <tr style="{hl_style}">
            <td style="width:140px">
                <span class="comp-badge" style="background:{color}">
                    {icon} {label}
                </span>{hl_badge}
            </td>
            <td style="font-size:13px;color:#ccc;">{name}</td>
            <td class="price-col" style="color:{color}">{fmt_vnd(price)}</td>
            <td style="text-align:center;white-space:nowrap">{link_html}</td>
        </tr>"""

    total = config.get("total", 0)
    pct   = total / ngan_sach * 100 if ngan_sach > 0 else 0
    rows_html += f"""
        <tr style="border-top:2px solid rgba(255,255,255,0.15);">
            <td><strong>TỔNG</strong></td>
            <td style="color:#aaa;font-size:13px;">{pct:.1f}% ngân sách</td>
            <td class="price-col" style="color:#ffd700;font-size:16px;">{fmt_vnd(total)}</td>
            <td></td>
        </tr>"""

    suffix = ""
    if has_stock_cooler:
        suffix = '<div style="font-size:12px;color:#888;margin-top:6px;">* CPU đi kèm tản nhiệt box</div>'
    return f"""<table class="comp-table">{rows_html}</table>{suffix}"""


def check_constraints(config: dict, wm: WorkingMemory) -> list[tuple[str, bool, str]]:
    """Kiểm tra C1-C5 cho cấu hình đã chọn, trả về (tên, pass, mô tả)."""
    def _f(x, d=0.0):
        try:
            v = float(x); return v if v == v else d
        except: return d

    FF_ORDER = {"EATX": 4, "ATX": 3, "mATX": 2, "ITX": 1}
    cpu = config.get("cpu", {}); mb = config.get("mainboard", {})
    ram = config.get("ram", {}); vga = config.get("vga", {}); psu = config.get("psu", {})
    case_ = config.get("case", {})

    c1 = cpu.get("socket") == mb.get("socket")
    c2 = ram.get("type") == mb.get("supported_ddr")
    cpu_tdp = _f(cpu.get("tdp_w"), 65)
    vga_tdp = min(_f(vga.get("tdp_w"), 0), 300)  # cap 300W (dữ liệu scraped có thể sai)
    psu_w   = _f(psu.get("wattage_w"), 500)
    c3 = (cpu_tdp + vga_tdp) <= psu_w * 0.8
    case_ff = case_.get("form_factor", "ATX"); mb_ff = mb.get("form_factor", "ATX")
    c4 = FF_ORDER.get(case_ff, 0) >= FF_ORDER.get(mb_ff, 0)
    total = config.get("total", 0)
    c5 = total <= wm.ngan_sach

    return [
        ("C1 Socket",      c1, f"CPU {cpu.get('socket','?')} == MB {mb.get('socket','?')}"),
        ("C2 DDR type",    c2, f"RAM {ram.get('type','?')} == MB {mb.get('supported_ddr','?')}"),
        ("C3 Power",       c3, f"({cpu_tdp:.0f}W + {vga_tdp:.0f}W) ≤ {psu_w:.0f}W × 0.8 = {psu_w*0.8:.0f}W"),
        ("C4 Form Factor", c4, f"Case {case_ff} ≥ MB {mb_ff}"),
        ("C5 Budget",      c5, f"{fmt_vnd(total)} ≤ {fmt_vnd(wm.ngan_sach)}"),
    ]


def rule_color(rule_id: str) -> str:
    try:
        n = int(rule_id.replace("R", ""))
        if n <= 10:   return "#3498db"
        if n <= 20:   return "#27ae60"
        if n <= 30:   return "#e67e22"
        if n <= 36:   return "#e74c3c"
        return "#9b59b6"
    except: return "#95a5a6"


def render_explanation(wm: WorkingMemory) -> None:
    for line in wm.explanation:
        rule_id = line.split("]")[0].replace("[", "").strip()
        color   = rule_color(rule_id)
        st.markdown(
            f'<div style="padding:6px 10px;margin:3px 0;border-radius:6px;'
            f'border-left:3px solid {color};background:rgba(0,0,0,0.2);font-size:13px;">'
            f'✅ {line}</div>',
            unsafe_allow_html=True,
        )


def render_top3_card(cfg: dict, wm: WorkingMemory, rank: int) -> None:
    ex         = explain_score(cfg, wm)
    rank_emoji = ["🥇", "🥈", "🥉"][rank]
    card_cls   = ["gold-card", "silver-card", "bronze-card"][rank]
    badge_html = '<span style="background:#ffd700;color:#000;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;">⭐ Tốt nhất</span>' if rank == 0 else ""

    st.markdown(f'<div class="{card_cls}">', unsafe_allow_html=True)
    st.markdown(f"#### {rank_emoji} Cấu hình #{rank+1} {badge_html if rank == 0 else ''}",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Tổng giá", fmt_vnd(cfg["total"]))
    c2.metric("Performance", f"{ex['perf']:.3f}")
    c3, c4 = st.columns(2)
    c3.metric("g(n)", f"{ex['g']:.3f}", help="Chi phí thực = total/ngân_sách")
    c4.metric("h(n)", f"{ex['h']:.3f}", help="Heuristic = 1 - performance_score")
    st.markdown(f"**f(n) = {ex['f']:.4f}**")
    st.markdown("---")
    for key in ["cpu", "vga", "ram", "storage"]:
        name  = str(cfg[key].get("name", "—"))[:45]
        color = COMPONENT_COLORS[key]
        st.markdown(
            f'<span class="comp-badge" style="background:{color}">'
            f'{COMPONENT_ICONS[key]} {LINH_KIEN_LABELS[key]}</span> '
            f'<span style="font-size:12px;color:#aaa;">{name}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_chat_entry(entry: dict) -> None:
    """Hiển thị 1 entry chat history — KHÔNG hiện predicates (đã hiện khi phân tích)."""
    st.markdown(
        f'<div class="chat-user">💬 <strong>Câu hỏi:</strong> {entry["text"]}</div>',
        unsafe_allow_html=True,
    )
    result = entry.get("result")
    if result is None:
        st.markdown(
            '<div class="chat-result">❌ Không tìm được cấu hình hợp lệ.</div>',
            unsafe_allow_html=True,
        )
    else:
        best  = result["best"]
        total = best["total"]
        cpu   = best["cpu"].get("name", "—")[:50]
        vga   = best["vga"].get("name", "—")[:50]
        perf  = result["perf_score"]
        st.markdown(
            f'<div class="chat-result">'
            f'✅ <strong>Cấu hình tìm thấy:</strong> {fmt_vnd(total)} '
            f'· Perf {perf:.3f}<br>'
            f'🔧 CPU: {cpu}<br>'
            f'🎮 VGA: {vga}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Hệ chuyên gia tư vấn cấu hình PC",
    page_icon="🖥️",
    layout="wide",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:36px;">🖥️</div>
        <div style="font-size:17px;font-weight:700;color:#3498db;">PC Expert System</div>
        <div style="font-size:11px;color:#888;margin-top:4px;">
            Hệ thống tư vấn thông minh
        </div>
    </div>
    """, unsafe_allow_html=True)

    total_products = _total_product_count()
    st.caption(f"📦 Cơ sở dữ liệu: **{total_products:,}** sản phẩm")
    st.markdown("---")

    st.header("⚙️ Yêu cầu của bạn")

    ngan_sach = st.slider(
        "💵 Ngân sách (VNĐ)",
        min_value=5_000_000, max_value=80_000_000,
        value=20_000_000, step=500_000, format="%d",
    )
    st.markdown(
        f'<div style="text-align:right;color:#3498db;font-weight:700;font-size:15px;">'
        f'{fmt_vnd(ngan_sach)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    muc_dich_label = st.selectbox(
        "🎯 Mục đích sử dụng",
        options=list(MUC_DICH_LABELS.values()), index=1,
    )
    muc_dich = next(k for k, v in MUC_DICH_LABELS.items() if v == muc_dich_label)

    uu_tien_label = st.radio(
        "⭐ Ưu tiên",
        options=list(UU_TIEN_LABELS.values()), index=0,
    )
    uu_tien = next(k for k, v in UU_TIEN_LABELS.items() if v == uu_tien_label)

    st.markdown("---")
    run_btn = st.button("🔍 TƯ VẤN NGAY", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════

if "sidebar_result" not in st.session_state:
    st.session_state.sidebar_result = None
if "show_balloons" not in st.session_state:
    st.session_state.show_balloons = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ══════════════════════════════════════════════════════════════════
# RUN SIDEBAR PIPELINE
# ══════════════════════════════════════════════════════════════════

if run_btn:
    with st.spinner("⏳ Đang chạy 3 thuật toán AI..."):
        r = recommend(ngan_sach, muc_dich, uu_tien)
    if r is not None:
        st.session_state.sidebar_result = r
        st.session_state.show_balloons  = True
    else:
        st.error(
            "❌ Không tìm được cấu hình hợp lệ. "
            "Thử tăng ngân sách hoặc đổi mục đích sử dụng."
        )
        st.session_state.sidebar_result = None

if st.session_state.show_balloons:
    st.balloons()
    st.session_state.show_balloons = False

result = st.session_state.sidebar_result


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════

st.title("🖥️ Hệ Chuyên Gia Tư Vấn Cấu Hình Máy Tính")
st.caption(
    "Hệ thống tư vấn thông minh giúp bạn chọn cấu hình máy tính "
    "phù hợp nhất với ngân sách và nhu cầu"
)


# ══════════════════════════════════════════════════════════════════
# 4 TABS — luôn hiển thị
# ══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🛒 Cấu hình đề xuất",
    "📊 So sánh Top 3",
    "🔧 Phân tích kỹ thuật",
    "💬 Chat AI",
])


# ─── TAB 1 — CẤU HÌNH ĐỀ XUẤT ────────────────────────────────────
with tab1:
    if result is None:
        st.info("👈 Chọn ngân sách, mục đích, ưu tiên ở thanh bên rồi bấm **TƯ VẤN NGAY**.")
        st.markdown("""
        ### Cách hoạt động
        1. **Suy luận tự động**: từ ngân sách + mục đích → suy ra tier linh kiện
           và phân bổ % ngân sách qua 40 rules IF-THEN.
        2. **Kiểm tra tương thích**: tìm cấu hình thỏa 5 ràng buộc kỹ thuật
           (socket, DDR, công suất, form factor, ngân sách).
        3. **Tối ưu hóa**: chọn cấu hình tốt nhất cân bằng giữa giá thực tế
           và điểm hiệu năng phù hợp với mục đích sử dụng.
        """)
    else:
        best = result["best"]
        wm   = result["working_memory"]

        # ── 4 metric cards ────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Tổng giá", fmt_vnd(best["total"]))
        pct_used = best["total"] / ngan_sach * 100
        c2.metric("📊 % ngân sách", f"{pct_used:.1f}%")
        c3.metric("🎯 Performance", f"{result['perf_score']:.3f}")
        c4.metric("⭐ A* f-score",  f"{result['f_score']:.4f}",
                  help="f(n) = g(n) + h(n). Càng nhỏ càng tốt.")

        # ── Progress bar ──────────────────────────────────────────
        prog_val = min(best["total"] / ngan_sach, 1.0)
        bar_color = "#27ae60" if prog_val < 0.9 else "#e67e22" if prog_val < 1.0 else "#e74c3c"
        st.markdown(
            f'<div style="margin:12px 0 4px"><strong>Ngân sách đã dùng: {pct_used:.1f}%</strong></div>'
            f'<div class="domain-bar-wrap"><div class="domain-bar" '
            f'style="width:{min(pct_used,100):.1f}%;background:{bar_color};"></div></div>',
            unsafe_allow_html=True,
        )

        # ── Component table ───────────────────────────────────────
        st.markdown("### Chi tiết cấu hình")
        st.markdown(render_config_html(best, ngan_sach), unsafe_allow_html=True)

        # ── Pie chart ─────────────────────────────────────────────
        st.markdown("### 🥧 Phân bổ ngân sách thực tế")
        pie_data = pd.DataFrame([
            {"Linh kiện": f"{COMPONENT_ICONS[k]} {LINH_KIEN_LABELS[k]}",
             "Giá": float(best[k].get("price", 0))}
            for k in ["cpu", "mainboard", "ram", "vga", "psu", "storage", "case", "cooler"]
            if best[k].get("price", 0) > 0
        ])
        if not pie_data.empty:
            fig = px.pie(
                pie_data, values="Giá", names="Linh kiện", hole=0.42,
                color_discrete_sequence=list(COMPONENT_COLORS.values()),
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(margin=dict(t=20, b=20, l=0, r=0), height=340)
            st.plotly_chart(fig, use_container_width=True)

        # ── Warnings ──────────────────────────────────────────────
        if wm.warnings:
            with st.expander(f"⚠ Cảnh báo ({len(wm.warnings)})", expanded=True):
                for w in wm.warnings:
                    st.warning(w)

        # ── Explanation Facility ──────────────────────────────────
        with st.expander(
            f"🧠 Giải thích AI — {len(result['fired_rules'])} quy tắc đã kích hoạt"
        ):
            st.markdown(
                "Hệ thống đã phân tích và đưa ra quyết định dựa trên các quy tắc sau:"
            )
            render_explanation(wm)


# ─── TAB 2 — SO SÁNH TOP 3 ──────────────────────────────────────
with tab2:
    if result is None:
        st.info("Chạy tư vấn từ sidebar để xem kết quả.")
    else:
        wm = result["working_memory"]
        st.markdown("### So sánh 3 cấu hình tốt nhất theo A\\*")

        cols = st.columns(3)
        for i, (cfg, col) in enumerate(zip(result["top3"], cols)):
            with col:
                render_top3_card(cfg, wm, i)

        # ── Bar chart so sánh ─────────────────────────────────────
        st.markdown("### 📊 So sánh Performance & f-score")
        _cfg_labels = ["#1 Tốt nhất", "#2 Tốt", "#3 Khá"][:len(result["top3"])]
        _colors_bar = ["#ffd700", "#a8a9ad", "#cd7f32"]
        _perf_vals  = [explain_score(c, wm)["perf"] for c in result["top3"]]
        _f_vals     = [explain_score(c, wm)["f"]    for c in result["top3"]]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=_cfg_labels, y=_perf_vals,
            name="Performance Score",
            marker_color=_colors_bar[:len(_cfg_labels)],
            text=[f"{v:.3f}" for v in _perf_vals], textposition="outside",
        ))
        fig2.add_trace(go.Bar(
            x=_cfg_labels, y=_f_vals,
            name="f-score (thấp = tốt)",
            marker_color=["rgba(255,200,0,0.4)", "rgba(168,169,173,0.4)", "rgba(205,127,50,0.4)"][:len(_cfg_labels)],
            text=[f"f={v:.3f}" for v in _f_vals], textposition="outside",
        ))
        fig2.update_layout(
            barmode="group", height=320,
            margin=dict(t=30, b=20, l=0, r=0),
            yaxis=dict(range=[0, max(max(_perf_vals), max(_f_vals)) * 1.25]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─── TAB 3 — KỸ THUẬT ────────────────────────────────────────────
with tab3:
    if result is None:
        st.info("Chạy tư vấn từ sidebar để xem chi tiết kỹ thuật.")
    else:
        best = result["best"]
        wm   = result["working_memory"]
        t    = result["timing"]
        total_t = t["fc"] + t["csp"] + t["astar"] + 1e-9

        # ── Timing ────────────────────────────────────────────────
        st.markdown("### ⏱ Thời gian xử lý")
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Forward Chaining",      f"{t['fc']*1000:.1f} ms",
                   delta=f"{t['fc']/total_t*100:.1f}% tổng thời gian", delta_color="off")
        tc2.metric("CSP + Forward Checking", f"{t['csp']*1000:.1f} ms",
                   delta=f"{t['csp']/total_t*100:.1f}% tổng thời gian", delta_color="off")
        tc3.metric("A* Search",             f"{t['astar']*1000:.1f} ms",
                   delta=f"{t['astar']/total_t*100:.1f}% tổng thời gian", delta_color="off")

        st.markdown(
            f'<div style="font-size:12px;color:#888;margin:-8px 0 12px">'
            f'Tổng: {total_t*1000:.1f} ms | Cấu hình hợp lệ: '
            f'<strong style="color:#27ae60">{result["valid_count"]}</strong></div>',
            unsafe_allow_html=True,
        )

        # ── Constraint Check ──────────────────────────────────────
        st.markdown("### ✅ Kiểm tra tính tương thích linh kiện")
        st.caption("Xác minh cấu hình tốt nhất thỏa mãn tất cả ràng buộc kỹ thuật:")
        constraints = check_constraints(best, wm)
        for name, passed, desc in constraints:
            icon  = "✅" if passed else "❌"
            color = "#2ecc71" if passed else "#e74c3c"
            st.markdown(
                f'<div style="padding:8px 12px;margin:4px 0;border-radius:8px;'
                f'background:rgba(0,0,0,0.2);border-left:3px solid {color};">'
                f'<strong style="color:{color}">{icon} {name}</strong>'
                f'<span style="color:#aaa;font-size:12px;margin-left:12px;">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Domain sizes ──────────────────────────────────────────
        st.markdown("### 📦 Số lượng sản phẩm phù hợp theo từng loại linh kiện")
        max_dom = max(result["domain_sizes"].values(), default=1)
        for var, count in result["domain_sizes"].items():
            color = COMPONENT_COLORS.get(var, "#888")
            pct   = int(count / max_dom * 100)
            st.markdown(
                f'<div style="display:flex;align-items:center;margin:5px 0;gap:10px;">'
                f'<span class="comp-badge" style="background:{color};min-width:90px;text-align:center">'
                f'{COMPONENT_ICONS.get(var,"")} {LINH_KIEN_LABELS.get(var, var)}</span>'
                f'<div class="domain-bar-wrap" style="flex:1">'
                f'<div class="domain-bar" style="width:{pct}%;background:{color};"></div></div>'
                f'<span style="min-width:30px;text-align:right;font-weight:700;">{count}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if result["fallback_log"]:
            with st.expander("⚠ Fallback domain (CSP đã nới ngân sách)"):
                for line in result["fallback_log"]:
                    st.code(line)

        # ── A* Breakdown ──────────────────────────────────────────
        st.markdown("### 🧮 A* Score breakdown (cấu hình tốt nhất)")
        ex = result["explain"]
        st.markdown(
            f'**f(n) = g(n) + h(n)** = `{ex["g"]:.4f}` + `{ex["h"]:.4f}` = '
            f'**`{ex["f"]:.4f}`**\n\n'
            f'- `g(n) = {best["total"]:,.0f} / {ngan_sach:,} = {ex["g"]:.4f}` *(chi phí thực)*\n'
            f'- `h(n) = 1 − {ex["perf"]:.4f} = {ex["h"]:.4f}` *(ước tính hiệu năng còn thiếu)*'
        )
        sub_df = pd.DataFrame([
            {"Sub-score": k.upper(),
             "Giá trị":   f"{ex['sub_scores'][k]:.3f}",
             "Trọng số":  f"{ex['weights'][k]:.2f}",
             "Đóng góp":  f"{ex['sub_scores'][k] * ex['weights'][k]:.3f}"}
            for k in ["cpu", "gpu", "ram", "storage"]
        ])
        st.dataframe(sub_df, use_container_width=True, hide_index=True)

        # ── Working Memory ─────────────────────────────────────────
        st.markdown("### 🧠 Trạng thái phân tích")
        wm_dict = {
            "Input": {"ngan_sach": wm.ngan_sach, "muc_dich": wm.muc_dich,
                      "uu_tien": wm.uu_tien or "(không)"},
            "Tier đã xác định": {
                "cpu_tier": wm.cpu_tier, "gpu_tier": wm.gpu_tier,
                "ram": f"{wm.ram_capacity}GB {wm.ram_type}",
                "storage_config": wm.storage_config,
                "form_factor": wm.form_factor, "cooler_type": wm.cooler_type,
                "psu_tier": f"{wm.psu_tier} ({wm.psu_wattage_min}W min)",
            },
            "Phân bổ ngân sách": {
                "cpu_budget":     fmt_vnd(wm.cpu_budget),
                "gpu_budget":     fmt_vnd(wm.gpu_budget),
                "ram_budget":     fmt_vnd(wm.ram_budget),
                "mb_budget":      fmt_vnd(wm.mb_budget),
                "psu_budget":     fmt_vnd(wm.psu_budget),
                "storage_budget": fmt_vnd(wm.storage_budget),
            },
            "Min specs": {
                "ram_min_gb": wm.ram_min_gb,
                "vram_min_gb": wm.vram_min_gb,
                "storage_min_gb": wm.storage_min_gb,
            },
        }
        st.json(wm_dict)

        st.markdown("### 🔥 Các quy tắc đã kích hoạt")
        _rule_docs = {line.split("]")[0].lstrip("["): line.split("] ", 1)[1]
                      for line in wm.explanation if "]" in line}
        for rid in wm.fired_rules:
            desc  = _rule_docs.get(rid, "")
            color = rule_color(rid)
            st.markdown(
                f'<div style="padding:4px 10px;margin:2px 0;border-radius:5px;'
                f'border-left:3px solid {color};background:rgba(0,0,0,0.15);font-size:13px;">'
                f'✅ <strong>[{rid}]</strong>'
                + (f' — {desc}' if desc else '')
                + '</div>',
                unsafe_allow_html=True,
            )


# ─── TAB 4 — CHAT AI ─────────────────────────────────────────────
with tab4:
    st.markdown("""
    ### 💬 Chat AI — Nhập câu tự nhiên
    <div style="color:#888;font-size:13px;margin-bottom:16px;">
        Hệ thống tự động nhận diện ngân sách, mục đích và ưu tiên từ câu bạn nhập.
    </div>
    """, unsafe_allow_html=True)

    # ── PHẦN A: Ô chat ─────────────────────────────────────────────
    chat_text = st.text_area(
        "📝 Mô tả yêu cầu của bạn",
        placeholder="Ví dụ: Tôi muốn build máy gaming 15 triệu, ưu tiên GPU mạnh",
        height=90, key="chat_input",
    )
    analyze_btn = st.button("🔍 Phân tích & Tư vấn", type="primary")

    if analyze_btn and chat_text.strip():
        facts  = extract_predicates(chat_text)
        exps   = explain_predicates(facts)

        # LỖI 4: hiển thị rõ nguồn ngân sách đang dùng
        if facts.get("ngan_sach"):
            st.info(f"💡 Sử dụng ngân sách từ câu chat: **{fmt_vnd(facts['ngan_sach'])}**")

        # Hiển thị predicates đã nhận diện (chỉ 1 lần ở đây, KHÔNG render lại trong lịch sử)
        st.markdown("#### 📌 Thông tin nhận diện được:")
        for e in exps:
            color = "#27ae60" if "?" not in e else "#e67e22"
            st.markdown(
                f'<div class="predicate-box" style="border-color:{color}55;">'
                f'<span style="color:{color}">◆</span> {e}</div>',
                unsafe_allow_html=True,
            )

        # Kiểm tra đủ thông tin
        if not facts.get("ngan_sach"):
            st.warning("⚠ Chưa nhận diện được ngân sách. "
                       "Vui lòng thêm số tiền vào câu (ví dụ: '15 triệu', '20tr').")
            chat_result = None
        elif not facts.get("muc_dich"):
            st.warning("⚠ Chưa nhận diện được mục đích. "
                       "Vui lòng đề cập đến mục đích sử dụng (gaming, văn phòng, ...).")
            chat_result = None
        else:
            # LỖI 4: LUÔN dùng ngân sách từ câu chat, KHÔNG dùng sidebar
            _chat_budget = facts["ngan_sach"]
            with st.spinner("⏳ Đang phân tích và tư vấn..."):
                chat_result = recommend(
                    ngan_sach=_chat_budget,
                    muc_dich=facts["muc_dich"],
                    uu_tien=facts.get("uu_tien", ""),
                )

            if chat_result is None:
                st.error("❌ Không tìm được cấu hình. Thử tăng ngân sách hoặc đổi mục đích.")
            else:
                st.success(f"✅ Tìm được **{chat_result['valid_count']}** cấu hình hợp lệ!")
                best_c = chat_result["best"]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("💰 Tổng giá", fmt_vnd(best_c["total"]))
                m2.metric("📊 % ngân sách", f"{best_c['total']/_chat_budget*100:.1f}%")
                m3.metric("🎯 Performance",  f"{chat_result['perf_score']:.3f}")
                m4.metric("⭐ A* f-score",   f"{chat_result['f_score']:.4f}")

                st.markdown(render_config_html(best_c, _chat_budget), unsafe_allow_html=True)

        # Lưu vào lịch sử (không lưu exps để tránh render lại trong lịch sử — LỖI 3)
        st.session_state.chat_history.append({
            "text":   chat_text,
            "facts":  facts,
            "result": chat_result if (facts.get("ngan_sach") and facts.get("muc_dich")) else None,
        })

    elif analyze_btn:
        st.warning("Vui lòng nhập câu yêu cầu trước khi phân tích.")

    # ── PHẦN B: Lịch sử chat ──────────────────────────────────────
    history = st.session_state.chat_history
    if history:
        st.markdown("---")
        st.markdown("#### 🕓 Lịch sử chat (3 câu gần nhất)")
        for entry in reversed(history[-3:]):
            render_chat_entry(entry)
            st.markdown("<hr style='border-color:rgba(255,255,255,0.05);'>",
                        unsafe_allow_html=True)

        if st.button("🗑 Xóa lịch sử", key="clear_history"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    🖥️ <strong>Hệ Chuyên Gia Tư Vấn Cấu Hình Máy Tính</strong>
</div>
""", unsafe_allow_html=True)
