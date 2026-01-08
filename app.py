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

    # 括号负数：(3.2) -> -3.2
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1].strip()

    # 去逗号/空格
    s = s.replace(",", "").replace(" ", "")

    # 百分号：12.3% -> 0.123
    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1]

    # 只保留数字/小数点/科学计数/正负号
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

# 默认规则：把 0 当缺失
def num_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].map(to_number)
    s = s.mask(s == 0, pd.NA)
    return s

def pct_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    内部统一成“小数”(0.2=20%)，并默认去掉 0。
    兼容：
      - "20%" -> 0.2
      - "20"  (表示20%) -> 启发式除以100
    """
    s = num_series(df, col)
    sample = s.dropna().abs()
    if len(sample) > 0:
        q = sample.quantile(0.5)
        if q > 1.5:  # 更像 20、30 这种“百分数点”
            s = s / 100.0
    return s

def apply_rule(mask: pd.Series, s: pd.Series, op: str, v1: float, v2=None) -> pd.Series:
    # 启用筛选：默认要求该列非空（等同 dropna），且 0 已在 series 里转 NA
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
        lo = v1
        hi = v2 if v2 is not None else v1
        if lo > hi:
            lo, hi = hi, lo
        return m & (s >= lo) & (s <= hi)
    return m

def find_numeric_like_columns(df: pd.DataFrame, sample_n=200, threshold=0.6):
    """
    自动识别“可转成数字”的列（用于筛选候选）：
    - 抽样 sample_n 个非空值
    - to_number 后非空比例 >= threshold 则认为可筛选
    - 排除：证券代码/证券简称、日期/时间列
    """
    cols = []
    for c in df.columns:
        name = str(c)

        if name in {"证券代码", "证券简称"}:
            continue

        low = name.lower()
        if any(k in low for k in ["date", "time", "日期", "时间"]):
            continue

        s = df[c]
        nn = s.dropna()
        if nn.empty:
            continue

        ss = nn.sample(min(sample_n, len(nn)), random_state=7)
        conv = ss.map(to_number)
        ratio = conv.notna().mean()
        if ratio >= threshold:
            cols.append(c)
    return cols

def is_yoy_qoq_col(col_name: str) -> bool:
    n = str(col_name).lower()
    return ("yoy" in n) or ("qoq" in n)

def format_percent_value(v, decimals=2):
    if pd.isna(v):
        return pd.NA
    # v 可能是 0.2（小数）或 20（百分数点）
    if abs(v) <= 1.5:
        p = v * 100.0
    else:
        p = v

    # 格式：尽量不让用户看到 20.00%
    if abs(p - round(p)) < 1e-9:
        return f"{int(round(p))}%"
    s = f"{p:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{s}%"

def make_display_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    用于【预览 + 下载】的展示版：
    - YOY/QOQ 列统一显示成 20% / 50%（字符串）
    - 其他列保持原样
    """
    out = df_raw.copy()
    pct_cols = [c for c in out.columns if is_yoy_qoq_col(c)]
    for c in pct_cols:
        s = out[c].map(to_number)
        # 如果更像“百分数点”（20、30），保持；如果更像小数（0.2）就*100
        sample = s.dropna().abs()
        if len(sample) > 0:
            q = sample.quantile(0.5)
            if q > 1.5:
                # 20 -> 20%
                out[c] = s.map(lambda x: format_percent_value(x, decimals=2))
            else:
                # 0.2 -> 20%
                out[c] = s.map(lambda x: format_percent_value(x, decimals=2))
        else:
            out[c] = s.map(lambda x: format_percent_value(x, decimals=2))
    return out

# =========================
# 上传
# =========================
uploaded = st.file_uploader("CSV(xslx,xls)上传", type=["csv", "xlsx"])
if not uploaded:
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df_in = pd.read_csv(uploaded)
    else:
        df_in = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"读取文件失败：{e}")
    st.stop()

# 原始表：B（不改写）
df_B = df_in.copy()

# =========================================================
# Sidebar：C（处理后） = 日期筛选（固定列名“日期”） + 多指标筛选
# =========================================================
st.sidebar.header("数据处理工作台")

# ---- 日期筛选（固定：日期列名就是“日期”）----
st.sidebar.subheader("1）日期")

df_after_date = df_B.copy()
date_col_fixed = "日期"

if date_col_fixed not in df_B.columns:
    st.sidebar.info("未找到列：日期，将跳过日期筛选。")
else:
    tmp = df_after_date.copy()
    tmp[date_col_fixed] = pd.to_datetime(tmp[date_col_fixed], errors="coerce")
    tmp = tmp.dropna(subset=[date_col_fixed])

    if tmp.empty:
        st.sidebar.warning("日期列解析后全是空值，已跳过日期筛选。")
    else:
        dmin = tmp[date_col_fixed].min().date()
        dmax = tmp[date_col_fixed].max().date()

        date_mode = st.sidebar.radio(
            "筛选方式",
            options=["指定日期", "日期区间"],
            index=0,
            key="date_mode",
        )

        if date_mode == "指定某一天":
            picked_day = st.sidebar.date_input(
                "选择日期（单日）",
                value=dmax,
                min_value=dmin,
                max_value=dmax,
                key="single_day",
            )
            day_window = st.sidebar.number_input(
                "可选：±窗口天数",
                min_value=0, max_value=30, value=0, step=1,
                key="day_window"
            )
            start = picked_day - pd.Timedelta(days=int(day_window))
            end = picked_day + pd.Timedelta(days=int(day_window))
            df_after_date = tmp[(tmp[date_col_fixed].dt.date >= start) & (tmp[date_col_fixed].dt.date <= end)].copy()
        else:
            range_picked = st.sidebar.date_input(
                "选择日期范围",
                value=(dmin, dmax),
                min_value=dmin,
                max_value=dmax,
                key="range_day",
            )
            if isinstance(range_picked, tuple) and len(range_picked) == 2:
                start, end = range_picked
            else:
                start = end = range_picked
            df_after_date = tmp[(tmp[date_col_fixed].dt.date >= start) & (tmp[date_col_fixed].dt.date <= end)].copy()

# ---- 多指标数值筛选（可选）----
st.sidebar.subheader("2）因子筛选")
st.sidebar.caption("请选择需要筛选的因子。注：默认去掉缺失值和 0")

numeric_like_cols = find_numeric_like_columns(df_B)
if not numeric_like_cols:
    st.sidebar.warning("未识别到可筛选的数值列")

selected_filter_cols = st.sidebar.multiselect(
    "因子（可多选）",
    options=numeric_like_cols,
    default=[],
    key="filter_cols",
)

OPS_UI = [">", ">=", "<", "<=", "介于"]
OP_MAP = {"介于": "between"}

mask = pd.Series(True, index=df_after_date.index)

for c in selected_filter_cols:
    with st.sidebar.expander(f"条件：{c}", expanded=True):
        op_ui = st.selectbox("操作符", OPS_UI, index=0, key=f"op_{c}")
        op = OP_MAP.get(op_ui, op_ui)

        # YOY/QOQ：用百分比输入（20代表20%）
        if is_yoy_qoq_col(c):
            st.markdown("输入单位：**%**（例如：20 代表 20%）")
            v1 = st.number_input("阈值1（%）", value=20.0, step=1.0, key=f"v1_{c}")
            v2 = None
            if op == "between":
                v2 = st.number_input("阈值2（%）", value=50.0, step=1.0, key=f"v2_{c}")

            s = pct_series(df_after_date, c)
            mask = apply_rule(
                mask, s, op,
                float(v1) / 100.0,
                None if v2 is None else float(v2) / 100.0
            )
        else:
            v1 = st.number_input("阈值1", value=0.0, step=1.0, key=f"v1_{c}")
            v2 = None
            if op == "between":
                v2 = st.number_input("阈值2", value=100.0, step=1.0, key=f"v2_{c}")

            s = num_series(df_after_date, c)
            mask = apply_rule(mask, s, op, float(v1), None if v2 is None else float(v2))

# 处理后表：C
df_C = df_after_date.loc[mask].copy()

# =========================
# 结果预览与下载（预览行数在这里）
# =========================
st.divider()
st.header("B / C 结果预览与下载")

preview_n = st.number_input(
    "数据预览行数",
    min_value=1, max_value=5000, value=50, step=10
)

tabB, tabC = st.tabs(["B（原始）", "C（处理后）"])

def show_block(df_show: pd.DataFrame, name: str):
    df_disp = make_display_df(df_show)

    st.subheader(f"{name}：数据预览")
    st.dataframe(df_disp.head(int(preview_n)), use_container_width=True)

    st.download_button(
        f"下载 {name}.csv",
        data=df_disp.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{name}.csv",
        mime="text/csv",
        key=f"dl_{name}"
    )

with tabB:
    show_block(df_B, "B")

with tabC:
    if df_C.empty:
        st.warning("C 为空：当前筛选条件下没有数据（日期/数值条件可能太严）。")
    show_block(df_C, "C")

# =========================
# 可视化：只选 B 或 C
# =========================
st.divider()
st.header("交互式可视化（散点2维图工作台）")

vis_source = st.radio(
    "选择可视化数据源",
    options=["B（原始）", "C（处理后）"],
    index=0,
    key="vis_src",
)

plot_df = df_C.copy() if vis_source.startswith("C") else df_B.copy()

if plot_df.empty:
    st.warning("当前选择的数据源为空，无法绘图。")
    st.stop()

all_cols_plot = list(plot_df.columns)

st.subheader("选择 X / Y / 点大小 / Color（自定义参数）")

default_x = all_cols_plot[0]
default_y = all_cols_plot[0]

x_col = st.selectbox("X轴（推荐使用PETTM）", all_cols_plot, index=all_cols_plot.index(default_x) if default_x in all_cols_plot else 0)
y_col = st.selectbox("Y轴（推荐使用QoQ）", all_cols_plot, index=all_cols_plot.index(default_y) if default_y in all_cols_plot else 0)

size_col = st.selectbox(
    "点大小（推荐使用YoY，市值）",
    options=["(不使用)"] + all_cols_plot,
    index=0
)

color_col = st.selectbox("颜色（推荐使用证券代码或证券简称）", options=["(不使用)"] + all_cols_plot, index=0)

# ---- hover_name（默认证券简称）----
hover_name_col = None
if "证券简称" in plot_df.columns:
    hover_name_col = "证券简称"
elif "证券代码" in plot_df.columns:
    hover_name_col = "证券代码"

# --- 强制：X/Y 缺失丢弃 ---
plot_df["_x_"] = plot_df[x_col].map(to_number)
plot_df["_y_"] = plot_df[y_col].map(to_number)

need = ["_x_", "_y_"]

# --- size：强制 abs ---
if size_col != "(不使用)":
    plot_df["_size_raw_"] = plot_df[size_col].map(to_number)
    plot_df["_size_"] = plot_df["_size_raw_"].abs()
    need.append("_size_")

plot_df = plot_df.dropna(subset=need)

if plot_df.empty:
    st.warning("当前选择下没有可绘制的数据（X/Y 无法转成数值或缺失）。")
    st.stop()

hover_cols = [c for c in plot_df.columns if c not in {"_x_", "_y_", "_size_", "_size_raw_"}]

fig = px.scatter(
    plot_df,
    x="_x_",
    y="_y_",
    size=("_size_" if size_col != "(不使用)" else None),
    color=(None if color_col == "(不使用)" else color_col),
    hover_name=hover_name_col,
    hover_data=hover_cols,
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
