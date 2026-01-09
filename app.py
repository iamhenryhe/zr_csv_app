import re
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="ZR*CSV_1.0", layout="wide")
st.title("ZR*CSV_1.0")

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
# 直接读取 data/B/B.csv
# =========================================================
DATA_PATH = "data/B/B.csv"

try:
    df_B = pd.read_excel("data/B/B.xlsx")
except Exception as e:
    st.error(f"读取 B 表失败：{e}")
    st.stop()

# =========================================================
# Sidebar：数据处理工作台
# =========================================================
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
st.header("B / C 结果预览与下载")

preview_n = st.number_input("数据预览行数", min_value=1, max_value=5000, value=50)

tabB, tabC = st.tabs(["B（原始）", "C（处理后）"])

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
st.divider()
st.header("交互式可视化（散点2维图工作台）")

vis_source = st.radio("选择可视化数据源", ["B（原始）", "C（处理后）"])
plot_df = df_C if vis_source.startswith("C") else df_B

all_cols_plot = list(plot_df.columns)
x_col = st.selectbox("X轴", all_cols_plot)
y_col = st.selectbox("Y轴", all_cols_plot)

plot_df["_x_"] = plot_df[x_col].map(to_number)
plot_df["_y_"] = plot_df[y_col].map(to_number)
plot_df = plot_df.dropna(subset=["_x_", "_y_"])

fig = px.scatter(plot_df, x="_x_", y="_y_")
st.plotly_chart(fig, use_container_width=True)
