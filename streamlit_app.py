"""
streamlit_app.py — UI demo Hệ Chuyên Gia Tư Vấn Cấu Hình Máy Tính
====================================================================
Tích hợp pipeline 3 thuật toán TTNT:
  Forward Chaining → CSP + Forward Checking → A* Search

Chạy:
  streamlit run streamlit_app.py

Pipeline recommend() được nhúng trực tiếp trong file này theo CLAUDE.md.
"""

import time
import pandas as pd
import streamlit as st
import plotly.express as px

from knowledge_base import WorkingMemory
from forward_chaining import run_forward_chaining
from csp_checker import load_data, filter_with_fallback, csp_with_forward_checking
from astar_selector import astar_select, explain_score, performance_score, WEIGHTS


# ══════════════════════════════════════════════════════════════════
# PIPELINE — recommend()
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load_data_cached():
    """Cache CSV loading vì đọc 8 file mỗi lần khá tốn thời gian."""
    return load_data()


def recommend(ngan_sach: int, muc_dich: str, uu_tien: str = "") -> dict | None:
    """
    Pipeline đầy đủ FC → CSP → A*.

    Returns: dict gồm best/top3/f_score/perf_score/fired_rules/working_memory/
             valid_count/timing/explain  hoặc None nếu không tìm được cấu hình.
    """
    # ── Bước 1: Forward Chaining (UNIT 10 + 11) ────────────────────
    t1 = time.time()
    wm = WorkingMemory(ngan_sach=ngan_sach, muc_dich=muc_dich, uu_tien=uu_tien)
    wm, fired = run_forward_chaining(wm)
    t_fc = time.time() - t1

    # ── Bước 2: Load data + CSP với Forward Checking ──────────────
    data = _load_data_cached()

    t2 = time.time()
    domains, fb_log = filter_with_fallback(data, wm)
    valid_configs = csp_with_forward_checking(domains, wm.ngan_sach, max_results=50)
    t_csp = time.time() - t2

    if not valid_configs:
        return None

    # ── Bước 3: A* Selection (Module 3 + lesson_34) ────────────────
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

# Vocab tiếng Việt cho user (KB dùng English internally)
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
    "cpu":       "CPU",
    "mainboard": "Mainboard",
    "ram":       "RAM",
    "vga":       "Card đồ họa (VGA)",
    "psu":       "Nguồn (PSU)",
    "storage":   "Ổ cứng (Storage)",
    "case":      "Vỏ máy (Case)",
    "cooler":    "Tản nhiệt (Cooler)",
}


def fmt_vnd(x: float | int) -> str:
    return f"{int(x):,}đ".replace(",", ".")


def config_to_table(config: dict) -> pd.DataFrame:
    """Chuyển 1 cấu hình thành DataFrame để st.dataframe hiển thị."""
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
        "Linh kiện": "**TỔNG**",
        "Tên sản phẩm": "—",
        "Giá": fmt_vnd(config.get("total", 0)),
        "Link": "",
    })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Hệ chuyên gia tư vấn cấu hình PC",
    page_icon="🖥️",
    layout="wide",
)

st.title("🖥️ Hệ Chuyên Gia Tư Vấn Cấu Hình Máy Tính")
st.caption("Pipeline 3 thuật toán TTNT: **Forward Chaining → CSP + Forward Checking → A\\* Search**")

# ── SIDEBAR — Input ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Yêu cầu của bạn")

    ngan_sach = st.slider(
        "💵 Ngân sách (VNĐ)",
        min_value=5_000_000,
        max_value=80_000_000,
        value=20_000_000,
        step=500_000,
        format="%d",
    )
    st.caption(f"→ {fmt_vnd(ngan_sach)}")

    muc_dich_label = st.selectbox(
        "🎯 Mục đích sử dụng",
        options=list(MUC_DICH_LABELS.values()),
        index=1,   # gaming default
    )
    muc_dich = next(k for k, v in MUC_DICH_LABELS.items() if v == muc_dich_label)

    uu_tien_label = st.radio(
        "⭐ Ưu tiên",
        options=list(UU_TIEN_LABELS.values()),
        index=0,
    )
    uu_tien = next(k for k, v in UU_TIEN_LABELS.items() if v == uu_tien_label)

    st.markdown("---")
    run_btn = st.button("🔍 TƯ VẤN NGAY", type="primary", use_container_width=True)


# ── MAIN — Output ──────────────────────────────────────────────────
if not run_btn:
    st.info(
        "👈 Chọn ngân sách, mục đích, ưu tiên ở thanh bên rồi bấm **TƯ VẤN NGAY**."
    )
    st.markdown("""
    ### Cách hoạt động
    1. **Forward Chaining** (UNIT 10 + 11): từ ngân sách + mục đích → suy ra
       tier linh kiện và phân bổ % ngân sách (39 rules).
    2. **CSP + Forward Checking** (Module 6): tìm cấu hình thỏa mãn 5 ràng buộc
       kỹ thuật (socket, DDR, công suất, form factor, ngân sách).
    3. **A\\* Search** (Module 3): chọn cấu hình tối ưu với f(n) = g(n) + h(n)
       — cân bằng giữa giá thực tế và điểm hiệu năng.
    """)
    st.stop()


with st.spinner("Đang chạy 3 thuật toán..."):
    result = recommend(ngan_sach, muc_dich, uu_tien)

if result is None:
    st.error(
        "❌ Không tìm được cấu hình hợp lệ với ngân sách hiện tại. "
        "Vui lòng tăng ngân sách hoặc đổi mục đích sử dụng."
    )
    st.stop()


# ── 3 TABS ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🛒 Cấu hình đề xuất",
    "📊 So sánh Top 3",
    "🔬 Kỹ thuật (cho thầy chấm)",
])

# ─── TAB 1 ────────────────────────────────────────────────────────
with tab1:
    best = result["best"]
    wm   = result["working_memory"]

    # 4 metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Tổng giá", fmt_vnd(best["total"]))
    pct_used = best["total"] / ngan_sach * 100
    c2.metric("📊 % ngân sách dùng", f"{pct_used:.1f}%")
    c3.metric("🎯 Performance score", f"{result['perf_score']:.3f}")
    c4.metric("⭐ A* f-score", f"{result['f_score']:.3f}",
              help="f(n) = g(n) + h(n). Càng nhỏ càng tốt.")

    st.markdown("### Chi tiết cấu hình")
    df = config_to_table(best)
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"Link": st.column_config.LinkColumn()})

    # Pie chart phân bổ giá thực tế
    st.markdown("### 🥧 Phân bổ giá thực tế")
    pie_data = pd.DataFrame([
        {"Linh kiện": LINH_KIEN_LABELS[k], "Giá": float(best[k].get("price", 0))}
        for k in ["cpu", "mainboard", "ram", "vga", "psu", "storage", "case", "cooler"]
        if best[k].get("price", 0) > 0
    ])
    if not pie_data.empty:
        fig = px.pie(pie_data, values="Giá", names="Linh kiện", hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # Warnings từ KB
    if wm.warnings:
        with st.expander(f"⚠ Cảnh báo ({len(wm.warnings)})", expanded=True):
            for w in wm.warnings:
                st.warning(w)

    # Explanation Facility
    with st.expander(f"🧠 Giải thích AI — Explanation Facility ({len(result['fired_rules'])} rules đã kích hoạt)"):
        st.markdown(
            "Hệ chuyên gia ghi lại chuỗi luật IF-THEN đã được kích hoạt "
            "(*data-driven reasoning* — UNIT 10) để giải thích quyết định:"
        )
        for line in wm.explanation:
            st.markdown(f"- {line}")


# ─── TAB 2 ────────────────────────────────────────────────────────
with tab2:
    st.markdown("### So sánh 3 cấu hình tốt nhất theo A*")

    cols = st.columns(3)
    for i, (cfg, col) in enumerate(zip(result["top3"], cols)):
        ex = explain_score(cfg, wm)
        rank_emoji = ["🥇", "🥈", "🥉"][i]
        with col:
            st.markdown(f"#### {rank_emoji} Cấu hình #{i+1}")
            st.metric("Tổng giá", fmt_vnd(cfg["total"]))
            st.metric("Performance", f"{ex['perf']:.3f}")
            st.metric("f-score", f"{ex['f']:.3f}",
                      delta=f"g={ex['g']:.3f}, h={ex['h']:.3f}",
                      delta_color="off")
            st.markdown("**Linh kiện chính:**")
            for k in ["cpu", "vga", "ram", "storage"]:
                name = str(cfg[k].get("name", "—"))[:40]
                st.caption(f"- **{LINH_KIEN_LABELS[k]}**: {name}")

    st.markdown("### 📊 Performance score so sánh")
    perf_df = pd.DataFrame([
        {"Cấu hình": f"#{i+1}", "Performance score": explain_score(c, wm)["perf"],
         "f-score": explain_score(c, wm)["f"], "Tổng giá (triệu)": c["total"]/1_000_000}
        for i, c in enumerate(result["top3"])
    ])
    st.bar_chart(perf_df.set_index("Cấu hình")[["Performance score", "f-score"]])


# ─── TAB 3 ────────────────────────────────────────────────────────
with tab3:
    st.markdown("### ⏱ Thời gian xử lý từng giai đoạn")
    t = result["timing"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Forward Chaining", f"{t['fc']*1000:.1f} ms")
    c2.metric("CSP + Forward Checking", f"{t['csp']*1000:.1f} ms")
    c3.metric("A* Search", f"{t['astar']*1000:.1f} ms")

    st.markdown(f"### 🎲 Số cấu hình hợp lệ sau CSP: **{result['valid_count']}**")
    if result["fallback_log"]:
        with st.expander("⚠ Fallback domain (CSP đã nới ngân sách)"):
            for line in result["fallback_log"]:
                st.code(line)

    st.markdown("### 📦 Domain size mỗi biến (sau filter tier+budget)")
    dom_df = pd.DataFrame([
        {"Biến": LINH_KIEN_LABELS[k], "Số sản phẩm còn lại": v}
        for k, v in result["domain_sizes"].items()
    ])
    st.dataframe(dom_df, use_container_width=True, hide_index=True)

    st.markdown("### 🧮 A* Score breakdown (best config)")
    ex = result["explain"]
    st.markdown(
        f"**f(n) = g(n) + h(n)** = `{ex['g']:.4f} + {ex['h']:.4f}` = **`{ex['f']:.4f}`**"
    )
    st.markdown(
        f"- `g(n) = total/ngan_sach = {best['total']:,.0f}/{ngan_sach:,} = {ex['g']:.4f}`\n"
        f"- `h(n) = 1 - performance_score = 1 - {ex['perf']:.4f} = {ex['h']:.4f}` "
        f"*(admissible vì h ≥ 0 và h* ≥ 0)*"
    )

    st.markdown("**Sub-scores (∈ [0, 1]):**")
    sub_df = pd.DataFrame([
        {"Sub-score": k.upper(),
         "Giá trị":   f"{ex['sub_scores'][k]:.3f}",
         "Trọng số":  f"{ex['weights'][k]:.2f}",
         "Đóng góp":  f"{ex['sub_scores'][k] * ex['weights'][k]:.3f}"}
        for k in ["cpu", "gpu", "ram", "storage"]
    ])
    st.dataframe(sub_df, use_container_width=True, hide_index=True)

    st.markdown("### 🧠 Working Memory (sau Forward Chaining)")
    wm_dict = {
        "Input": {
            "ngan_sach":  wm.ngan_sach,
            "muc_dich":   wm.muc_dich,
            "uu_tien":    wm.uu_tien or "(không)",
        },
        "Tier output (R01-R31)": {
            "cpu_tier":       wm.cpu_tier,
            "gpu_tier":       wm.gpu_tier,
            "ram":            f"{wm.ram_capacity}GB {wm.ram_type}",
            "storage_config": wm.storage_config,
            "form_factor":    wm.form_factor,
            "cooler_type":    wm.cooler_type,
            "psu_tier":       f"{wm.psu_tier} ({wm.psu_wattage_min}W min)",
        },
        "Budget allocation (R32-R39)": {
            "cpu_budget":     fmt_vnd(wm.cpu_budget),
            "gpu_budget":     fmt_vnd(wm.gpu_budget),
            "ram_budget":     fmt_vnd(wm.ram_budget),
            "mb_budget":      fmt_vnd(wm.mb_budget),
            "psu_budget":     fmt_vnd(wm.psu_budget),
            "storage_budget": fmt_vnd(wm.storage_budget),
        },
        "Min specs": {
            "ram_min_gb":     wm.ram_min_gb,
            "vram_min_gb":    wm.vram_min_gb,
            "storage_min_gb": wm.storage_min_gb,
        },
    }
    st.json(wm_dict)

    st.markdown("### 🔥 Fired rules (theo thứ tự kích hoạt)")
    st.code(" → ".join(wm.fired_rules), language="text")
