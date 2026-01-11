import re
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="业绩断层0.1", layout="wide")
st.title("业绩断层0.1")

# ====== 缺失值 ======
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-", "--", "—", "–"}

def normalize_col(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip().lower()

def to_number(x):
    if x is None:
        return pd.NA
    if isinstance(x, (int, float)):
        return x

    s = str(x).strip()
    if normalize_col(s) in MISSING_TOKENS:
        return pd.NA

    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1].strip()

    s = s.replace(",", "").replace(" ", "")

    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1]

    s = re.sub(r"[^0-9\.\-\+eE]", "", s)
    if s in {"", "+", "-", ".", "+.", "-."}:
        return pd.NA

    try:
        v = float(s)
        if neg:
            v = -v
        if is_percent:
            v = v / 100.0
        return v
    except Exception:
        return pd.NA

def num_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].map(to_number)
    s = s.mask(s == 0, pd.NA)
    return s

def pct_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = num_series(df, col)
    sample = s.dropna().abs()
    if len(sample) > 0 and sample.quantile(0.5) > 1.5:
        s = s / 100.0
    return s

def apply_rule(mask: pd.Series, s: pd.Series, op: str, v1: float, v2=None) -> pd.Series:
    m = mask & s.notna()
    if op == ">":
        return m & (s > v1)
    if op == ">=":
        return m & (s >= v1)
    if op == "<":
        return m & (s < v1)
    if op == "<=":
        return m & (s <= v1)
    if op == "between":
        lo, hi = v1, v2 if v2 is not None else v1
        if lo > hi:
            lo, hi = hi, lo
        return m & (s >= lo) & (s <= hi)
    return m

def find_numeric_like_columns(df: pd.DataFrame, sample_n=200, threshold=0.6):
    cols = []
    for c in df.columns:
        if c in {"证券代码", "证券简称"}:
            continue
        low = str(c).lower()
        if any(k in low for k in ["date", "time", "日期", "时间"]):
            continue
        nn = df[c].dropna()
        if nn.empty:
            continue
        ss = nn.sample(min(sample_n, len(nn)), random_state=7)
        if ss.map(to_number).notna().mean() >= threshold:
            cols.append(c)
    return cols

def is_yoy_qoq_col(col_name: str) -> bool:
    n = str(col_name).lower()
    return ("yoy" in n) or ("qoq" in n)

def format_percent_value(v, decimals=2):
    if pd.isna(v):
        return pd.NA
    p = v * 100 if abs(v) <= 1.5 else v
    if abs(p - round(p)) < 1e-9:
        return f"{int(round(p))}%"
    return f"{p:.{decimals}f}".rstrip("0").rstrip(".") + "%"

def make_display_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    out = df_raw.copy()
    for c in [c for c in out.columns if is_yoy_qoq_col(c)]:
        out[c] = out[c].map(to_number).map(lambda x: format_percent_value(x))
    return out

# =========================================================
# streamlit从路径从找b
# =========================================================
from pathlib import Path

DATA_ROOT = Path("data")

st.sidebar.header("业绩断层")
st.sidebar.subheader("选择数据集")

years = sorted([
    p.name for p in DATA_ROOT.iterdir()
    if p.is_dir() and re.fullmatch(r"20\d{2}", p.name)
]) if DATA_ROOT.exists() else []

if not years:
    st.sidebar.error("data/ 下未找到年份目录（例如 data/2025、data/2026...）")
    st.stop()

quarters = ["Q1", "Q2", "Q3", "Q4"]
kinds = ["预告", "实发"]

# 默认：2025 / Q4 / 预告
year_default = years.index("2025") if "2025" in years else 0
year_sel = st.sidebar.selectbox("年份", years, index=year_default)

quarter_sel = st.sidebar.radio("季度", quarters, index=3, horizontal=True)  # 默认 Q4
kind_sel = st.sidebar.radio("类型", kinds, index=0, horizontal=True)       # 默认 预告

b_path = DATA_ROOT / year_sel / f"{quarter_sel}{kind_sel}" / "B" / "B.xlsx"


if not b_path.exists():
    st.sidebar.error(f"未找到 B.xlsx：{b_path}")
    st.stop()

try:
    df_B = pd.read_excel(b_path)
except Exception as e:
    st.error(f"读取 B 表失败：{b_path}\n\n{e}")
    st.stop()



st.sidebar.header("数据处理工作台")

# ---- 日期 ----
st.sidebar.subheader("1）日期")

df_after_date = df_B.copy()
date_col_fixed = "日期"

if date_col_fixed not in df_B.columns:
    st.sidebar.info("未找到列：日期，将跳过日期筛选。")
else:
    tmp = df_after_date.copy()
    tmp[date_col_fixed] = pd.to_datetime(tmp[date_col_fixed], errors="coerce")
    tmp = tmp.dropna(subset=[date_col_fixed])

    if not tmp.empty:
        dmin, dmax = tmp[date_col_fixed].min().date(), tmp[date_col_fixed].max().date()
        date_mode = st.sidebar.radio("筛选方式", ["指定日期", "日期区间"], index=0)

        if date_mode == "指定日期":
            picked_day = st.sidebar.date_input("选择日期", value=dmax, min_value=dmin, max_value=dmax)
            df_after_date = tmp[tmp[date_col_fixed].dt.date == picked_day].copy()
        else:
            start, end = st.sidebar.date_input("选择日期区间", value=(dmin, dmax))
            df_after_date = tmp[(tmp[date_col_fixed].dt.date >= start) & (tmp[date_col_fixed].dt.date <= end)].copy()

st.sidebar.subheader("2）因子筛选")
numeric_like_cols = find_numeric_like_columns(df_B)

selected_filter_cols = st.sidebar.multiselect("因子（可多选）", numeric_like_cols)
OPS_UI = [">", ">=", "<", "<=", "介于"]
OP_MAP = {"介于": "between"}

mask = pd.Series(True, index=df_after_date.index)

for c in selected_filter_cols:
    with st.sidebar.expander(f"条件：{c}", expanded=True):
        op_ui = st.selectbox("操作符", OPS_UI, key=f"op_{c}")
        op = OP_MAP.get(op_ui, op_ui)

        if is_yoy_qoq_col(c):
            v1 = st.number_input("阈值1（%）", value=20.0, key=f"v1_{c}")
            v2 = st.number_input("阈值2（%）", value=50.0, key=f"v2_{c}") if op == "between" else None
            s = pct_series(df_after_date, c)
            mask &= apply_rule(mask, s, op, v1 / 100, None if v2 is None else v2 / 100)
        else:
            v1 = st.number_input("阈值1", value=0.0, key=f"v1_{c}")
            v2 = st.number_input("阈值2", value=100.0, key=f"v2_{c}") if op == "between" else None
            s = num_series(df_after_date, c)
            mask &= apply_rule(mask, s, op, v1, v2)

df_C = df_after_date.loc[mask].copy()

# =========================
# 结果预览与下载（以下全部保持不变）
# =========================
st.divider()
st.header("数据表")

preview_n = st.number_input("数据预览行数", min_value=1, max_value=5000, value=50)

tabB, tabC = st.tabs(["汇总（仅日期筛选）", "筛选（日期+因子筛选）"])

def show_block(df_show: pd.DataFrame, name: str):
    df_disp = make_display_df(df_show)
    st.dataframe(df_disp.head(int(preview_n)), use_container_width=True)
    st.download_button(
        f"下载 {name}.csv",
        data=df_disp.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{name}.csv",
        mime="text/csv",
    )

with tabB:
    show_block(df_B, "B")

with tabC:
    show_block(df_C, "C")

# =========================
# 可视化（保持不变）
# =========================
# =========================
# 可视化
# =========================
st.divider()
st.header("2D可视化")

use_c = len(selected_filter_cols) > 0
plot_df = df_C.copy() if use_c else df_B.copy()

if plot_df.empty:
    st.warning("当前选择的数据源为空，无法绘图。")
    st.stop()

all_cols_plot = list(plot_df.columns)

st.subheader("选择 X / Y / 点大小 / Color（自定义参数）")

default_x = all_cols_plot[0]
default_y = all_cols_plot[0]

x_col = st.selectbox("X轴（推荐使用PETTM）", all_cols_plot, index=all_cols_plot.index(default_x) if default_x in all_cols_plot else 0)
y_col = st.selectbox("Y轴（推荐使用QoQ）", all_cols_plot, index=all_cols_plot.index(default_y) if default_y in all_cols_plot else 0)

size_col = st.selectbox("点大小（推荐使用YoY，市值）", options=["(不使用)"] + all_cols_plot, index=0)
color_col = st.selectbox("颜色（推荐使用证券代码或证券简称）", options=["(不使用)"] + all_cols_plot, index=0)

hover_name_col = "证券简称" if "证券简称" in plot_df.columns else ("证券代码" if "证券代码" in plot_df.columns else None)

plot_df["_x_"] = plot_df[x_col].map(to_number)
plot_df["_y_"] = plot_df[y_col].map(to_number)

# =========================
# X和Y可视化的范围控制
# =========================
st.subheader("可视化数值范围控制")

col_x, col_y = st.columns(2)

# ---- X 轴 ----
with col_x:
    enable_x_range = st.checkbox(f"限制 X 轴（{x_col}）范围", value=False)

    if enable_x_range:
        xv = plot_df["_x_"].dropna()
        if not xv.empty:
            xmin, xmax = float(xv.min()), float(xv.max())
            pad = (xmax - xmin) * 0.05 if xmax > xmin else 1.0

            x_range = st.slider(
                f"{x_col} 区间",
                min_value=xmin - pad,
                max_value=xmax + pad,
                value=(xmin, xmax),
            )

            plot_df = plot_df[
                (plot_df["_x_"] >= x_range[0]) &
                (plot_df["_x_"] <= x_range[1])
            ]

# ---- Y 轴 ----
with col_y:
    enable_y_range = st.checkbox(f"限制 Y 轴（{y_col}）范围", value=False)

    if enable_y_range:
        yv = plot_df["_y_"].dropna()
        if not yv.empty:
            ymin, ymax = float(yv.min()), float(yv.max())
            pad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0

            y_range = st.slider(
                f"{y_col} 区间",
                min_value=ymin - pad,
                max_value=ymax + pad,
                value=(ymin, ymax),
            )

            plot_df = plot_df[
                (plot_df["_y_"] >= y_range[0]) &
                (plot_df["_y_"] <= y_range[1])
            ]

need = ["_x_", "_y_"]

if size_col != "(不使用)":
    plot_df["_size_raw_"] = plot_df[size_col].map(to_number)
    plot_df["_size_"] = plot_df["_size_raw_"].abs()
    need.append("_size_")

plot_df = plot_df.dropna(subset=need)

if plot_df.empty:
    st.warning("当前选择下没有可绘制的数据（X/Y 无法转成数值或缺失）。")
    st.stop()

# =========================
# Hover 内容
# =========================
# Hover Title：证券简称（证券代码）
if "证券简称" in plot_df.columns and "证券代码" in plot_df.columns:
    plot_df["_hover_title_"] = plot_df["证券简称"] + "（" + plot_df["证券代码"] + "）"
    hover_name_col = "_hover_title_"

HOVER_FIELDS = [
    "25Q4单季扣非",
    "QOQ",
    "YOY",
    "2025PE",
    "PETTM",
    "总市值（亿）",
]

hover_cols = [c for c in HOVER_FIELDS if c in plot_df.columns]


hover_data_dict = {c: True for c in hover_cols}

for internal_col in ["_x_", "_y_", "_size_", "_size_raw_"]:
    if internal_col in plot_df.columns:
        hover_data_dict[internal_col] = False

fig = px.scatter(
    plot_df,
    x="_x_",
    y="_y_",
    size=("_size_" if size_col != "(不使用)" else None),
    color=(None if color_col == "(不使用)" else color_col),
    hover_name=hover_name_col,
    hover_data=hover_data_dict,
)


fig.update_layout(
    height=700,
    xaxis_title=x_col,
    yaxis_title=y_col,
    margin=dict(l=10, r=10, t=40, b=10),
)
fig.update_traces(marker=dict(opacity=0.75), selector=dict(mode="markers"))

st.plotly_chart(fig, use_container_width=True)

with st.expander("数据预览（绘图数据）", expanded=False):
    st.dataframe(make_display_df(plot_df[hover_cols]).head(int(preview_n)), use_container_width=True)
