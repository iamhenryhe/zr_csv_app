import streamlit as st
from auth import require_login

#if "logged_in" not in st.session_state:
#    st.session_state.logged_in = False

#if not require_login():
#    st.stop()

import os
import pandas as pd
import re
import plotly.express as px
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher


st.set_page_config(page_title="中睿合银agent", layout="wide")

#yingwen bian zhong wen
def display_col_name(col):
    if col == "YOY":
        return "同比"
    if col == "QOQ":
        return "环比"
    return col

# =========================================================
# 模块激活状态 reference： aiagents-stock 的模块按钮：https://github.com/oficcejo/aiagents-stock）
# =========================================================
if "active_module" not in st.session_state:
    st.session_state.active_module = None

# 模块入口
st.sidebar.title("快速导航")

if st.sidebar.button("📊 业绩断层"):
    if st.session_state.active_module == "业绩断层":
        st.session_state.active_module = None
    else:
        st.session_state.active_module = "业绩断层"

if st.sidebar.button("🔥 传播度"):
    if st.session_state.active_module == "传播度":
        st.session_state.active_module = None
    else:
        st.session_state.active_module = "传播度"

if st.sidebar.button("📁 板块数据库"):
    if st.session_state.active_module == "板块数据库":
        st.session_state.active_module = None
    else:
        st.session_state.active_module = "板块数据库"



if st.session_state.active_module is None:
    st.info("👈 点击左侧项目以展开指定投研模块")
    st.stop()

# =========================================================
# 传播度模块（独立渲染，避免干扰业绩断层）
# =========================================================
if st.session_state.active_module == "传播度":
    from cbd import render as render_cbd

    #OSSSSSS
    cbd_base = os.getenv("CBD_BASE_DIR", "").strip()
    try:
        if cbd_base:
            render_cbd(base_dir=Path(cbd_base))
        else:
            render_cbd()
    except TypeError:
        render_cbd()

    st.stop()

# =========================================================
# 板块数据库模块（独立渲染，避免干扰前面两个）
# =========================================================
if st.session_state.active_module == "板块数据库":
    from database import render as render_db

    db_base = os.getenv("DB_BASE_DIR", "").strip()
    # 不配环境变量时，默认用本地目录：board-db
    render_db(base_dir=Path(db_base) if db_base else Path("板块数据库"))
    st.stop()


# =========================================================
# 业绩断层的module
# =========================================================
st.title("业绩断层0.1")
st.markdown("说明：此工作台负责将各个股的财报计算成技术因子，展示数据集均为财报计算清洗后表格，加以交互可视化分析。")

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
    n = str(col_name)
    return n in("YOY", "QOQ", "同比", "环比")

def format_percent_value(v, decimals=2):
    if pd.isna(v):
        return pd.NA
    p = v * 100 if abs(v) <= 1.5 else v
    if abs(p - round(p)) < 1e-9:
        return f"{int(round(p))}%"
    return f"{p:.{decimals}f}".rstrip("0").rstrip(".") + "%"

def make_display_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    out = df_raw.copy()

    for c in out.columns:
        if is_yoy_qoq_col(c):
            out[c] = out[c].map(to_number).map(
                lambda v: (f"{v*100:.1f}%"
                           if pd.notna(v) else pd.NA)
            )
        else:
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].map(lambda v: f"{v:.1f}" if pd.notna(v) else pd.NA)

    return out

# =========================================================
# streamlit从路径从找b + 自动 A->B
# =========================================================

#  关键：DATA_ROOT 支持 OSS（环境变量 DATA_ROOT）
# 本地默认 "data"
# 服务器 OSS 挂载后：export DATA_ROOT=/mnt/oss/xxx/data
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))

# a2b
from transform.a2b import ensure_b_up_to_date

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

year_default = years.index("2025") if "2025" in years else 0
year_sel = st.sidebar.selectbox("年份", years, index=year_default)

quarter_sel = st.sidebar.radio("季度", quarters, index=3, horizontal=True)
kind_sel = st.sidebar.radio("类型", kinds, index=0, horizontal=True)

a_path = DATA_ROOT / year_sel / f"{quarter_sel}{kind_sel}" / "A" / "A.xlsx"
b_path = DATA_ROOT / year_sel / f"{quarter_sel}{kind_sel}" / "B" / "B.xlsx"

try:
    did = ensure_b_up_to_date(a_path, b_path, force=False)
    if did:
        st.sidebar.success("已根据 A.xlsx 更新 B.xlsx")
    else:
        st.sidebar.caption("B.xlsx 已是最新（无需重算）")
except Exception as e:
    st.sidebar.error(f"A→B 失败：{e}")
    st.stop()

if not b_path.exists():
    st.sidebar.error(f"未找到 B.xlsx：{b_path}")
    st.stop()

try:
    df_B = pd.read_excel(b_path)
except Exception as e:
    st.error(f"读取 B 表失败：{b_path}\n\n{e}")
    st.stop()

st.sidebar.header("数据处理工作台")
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
            start, end = st.sidebar.date_input(
                "选择日期区间",
                value=(dmin, dmax),
                min_value=dmin,
                max_value=dmax,
            )

            df_after_date = tmp[
                (tmp[date_col_fixed].dt.date >= start) &
                (tmp[date_col_fixed].dt.date <= end)
            ].copy()


            df_after_date = tmp[
                (tmp[date_col_fixed].dt.date >= start) &
                (tmp[date_col_fixed].dt.date <= end)
            ].copy()


st.sidebar.subheader("2）因子筛选")
numeric_like_cols = find_numeric_like_columns(df_B)

selected_filter_cols = st.sidebar.multiselect("因子（可多选）", numeric_like_cols,format_func=display_col_name,)
OPS_UI = [">", ">=", "<", "<=", "介于"]
OP_MAP = {"介于": "between"}

mask = pd.Series(True, index=df_after_date.index)

for c in selected_filter_cols:
    with st.sidebar.expander(f"条件：{c}", expanded=True):
        if c == "2025PE":
            default_op_index = OPS_UI.index("介于")
        else:
            default_op_index = 0

        op_ui = st.selectbox(
            "操作符",
            OPS_UI,
            index=default_op_index,
            key=f"op_{c}"
        )
        op = OP_MAP.get(op_ui, op_ui)

        if is_yoy_qoq_col(c):
            v1 = st.number_input("阈值1（%）", value=0.0, key=f"v1_{c}")
            v2 = st.number_input("阈值2（%）", value=100.0, key=f"v2_{c}") if op == "between" else None
            s = pct_series(df_after_date, c)
            mask &= apply_rule(mask, s, op, v1 / 100, None if v2 is None else v2 / 100)
        else:
            v1 = st.number_input("阈值1", value=0.0, key=f"v1_{c}")
            v2 = st.number_input("阈值2", value=100.0, key=f"v2_{c}") if op == "between" else None
            s = num_series(df_after_date, c)
            mask &= apply_rule(mask, s, op, v1, v2)

df_C = df_after_date.loc[mask].copy()

st.divider()
st.header("数据表")

preview_n = st.number_input("数据预览行数", min_value=1, max_value=5000, value=50)

tabB, tabC = st.tabs(["汇总", "筛选（日期+因子筛选）"])

def show_block(df_show: pd.DataFrame, name: str):
    df_view = df_show.copy()

    st.dataframe(
        df_view.head(int(preview_n)),
        use_container_width=True,
        column_config={
            "同比": st.column_config.NumberColumn(label="同比", format="%.1f%%"),
            "环比": st.column_config.NumberColumn(label="环比", format="%.1f%%"),
        }
    )
    df_disp = make_display_df(df_show)
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

def yoy_to_size_bucket(v):
    if pd.isna(v):
        return 6
    if v < 0:
        return 6
    elif v <= 0.5:
        return 10
    elif v <= 1.0:
        return 14
    elif v <= 2.0:
        return 18
    else:
        return 22

st.divider()
st.header("可视化展示")
#jan30 。加了一个默认选好的作图选项
def index_of(options, value, default=0):
    try:
        return options.index(value)
    except ValueError:
        return default

use_c = len(selected_filter_cols) > 0
plot_df = df_C.copy() if use_c else df_after_date.copy()

if plot_df.empty:
    st.warning("当前选择的数据源为空，无法绘图。")
    st.stop()

all_cols_plot = list(plot_df.columns)

st.subheader("选择 X / Y / 点大小 / 颜色（自定义参数）")

default_x = all_cols_plot[0]
default_y = all_cols_plot[0]

x_col = st.selectbox(
    "X轴（推荐使用2025PE,PETTM）",
    all_cols_plot,
    index=index_of(all_cols_plot, "2025PE"),
    format_func=display_col_name,
)
y_col = st.selectbox(
    "Y轴（推荐使用环比）",
    all_cols_plot,
    index=index_of(all_cols_plot, "环比"),
    format_func=display_col_name,
)

size_options = ["(不使用)"] + all_cols_plot
size_col = st.selectbox(
    "点大小（推荐使用同比，市值）",
    options=size_options,
    index=index_of(size_options, "同比"),
    format_func=lambda x: "不使用" if x == "(不使用)" else display_col_name(x),
)

color_options = ["(不使用)"] + all_cols_plot
color_col = st.selectbox(
    "颜色（推荐使用证券代码或证券简称）",
    options=color_options,
    index=index_of(color_options, "证券代码"),
    format_func=lambda x: "不使用" if x == "(不使用)" else display_col_name(x),
)

plot_df["_x_"] = plot_df[x_col].map(to_number)
if is_yoy_qoq_col(x_col):
    plot_df["_x_"] = plot_df["_x_"] * 100

plot_df["_y_"] = plot_df[y_col].map(to_number)
if is_yoy_qoq_col(y_col):
    plot_df["_y_"] = plot_df["_y_"] * 100

st.subheader("指定代码（不选则默认符合筛选条件的全部标的）")

HAS_NAME = "证券简称" in plot_df.columns
HAS_CODE = "证券代码" in plot_df.columns

if HAS_NAME or HAS_CODE:
    def make_label(row):
        name = str(row["证券简称"]) if HAS_NAME else ""
        code = str(row["证券代码"]) if HAS_CODE else ""
        if name and code:
            return f"{name}（{code}）"
        return name or code

    plot_df["_sec_label_"] = plot_df.apply(make_label, axis=1)

    label_to_index = (
        plot_df[["_sec_label_"]]
        .reset_index()
        .set_index("_sec_label_")["index"]
        .to_dict()
    )
    all_labels = sorted(label_to_index.keys())
else:
    label_to_index = {}
    all_labels = []

col_keep, col_drop = st.columns(2)

with col_keep:
    keep_labels = st.multiselect(
        "添加（可多选，输入股票代码或简称即可）",
        options=all_labels,
        default=[],
        help="只显示你选中的证券",
        placeholder="请选择",
    )

with col_drop:
    drop_labels = st.multiselect(
        "删除（可多选）",
        options=all_labels,
        default=[],
        help="这些证券不会出现在图中",
        placeholder="请选择",
    )

if keep_labels:
    keep_idx = [label_to_index[l] for l in keep_labels]
    plot_df = plot_df.loc[keep_idx].copy()

if drop_labels:
    drop_idx = {label_to_index[l] for l in drop_labels}
    plot_df = plot_df.loc[~plot_df.index.isin(drop_idx)].copy()

col_x, col_y = st.columns(2)

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
            plot_df = plot_df[(plot_df["_x_"] >= x_range[0]) & (plot_df["_x_"] <= x_range[1])]

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
            plot_df = plot_df[(plot_df["_y_"] >= y_range[0]) & (plot_df["_y_"] <= y_range[1])]

need = ["_x_", "_y_"]
if size_col != "(不使用)":
    raw = plot_df[size_col].map(to_number)
    if is_yoy_qoq_col(size_col):
        plot_df["_size_"] = raw.apply(yoy_to_size_bucket)
    else:
        plot_df["_size_"] = pd.qcut(
            raw.abs(),
            q=5,
            labels=[8, 12, 16, 20, 24],
            duplicates="drop"
        )
    need.append("_size_")

plot_df = plot_df.dropna(subset=need)
if plot_df.empty:
    st.warning("当前选择下没有可绘制的数据（X/Y 无法转成数值或缺失）。")
    st.stop()

if "证券简称" in plot_df.columns and "证券代码" in plot_df.columns:
    plot_df["_hover_title_"] = plot_df["证券简称"] + "（" + plot_df["证券代码"] + "）"
    hover_name_col = "_hover_title_"
else:
    hover_name_col = None

#hover中文版
if "同比" in plot_df.columns:
    plot_df["_同比_PCT_"] = plot_df["同比"] * 100
if "环比" in plot_df.columns:
    plot_df["_环比_PCT_"] = plot_df["环比"] * 100

CUSTOM_FIELDS = [
    "_同比_PCT_",
    "_环比_PCT_",
    "25Q4单季扣非",
    "2025PE",
    "PETTM",
    "总市值（亿）",
]
custom_cols = [c for c in CUSTOM_FIELDS if c in plot_df.columns]

fig = px.scatter(
    plot_df,
    x="_x_",
    y="_y_",
    size=("_size_" if size_col != "(不使用)" else None),
    color=(None if color_col == "(不使用)" else color_col),
    hover_name=hover_name_col,
    custom_data=custom_cols,
)

hover_lines = []
hover_lines.append("%{hovertext}")

if is_yoy_qoq_col(x_col):
    hover_lines.append(f"{display_col_name(x_col)}: %{{x:.1f}}%")
else:
    hover_lines.append(f"{display_col_name(x_col)}: %{{x:.2f}}")

if is_yoy_qoq_col(y_col):
    hover_lines.append(f"{display_col_name(y_col)}: %{{y:.1f}}%")
else:
    hover_lines.append(f"{display_col_name(y_col)}: %{{y:.2f}}")

for i, c in enumerate(custom_cols):
    raw_name = c.replace("_同比_PCT_", "同比").replace("_环比_PCT_", "环比")
    display_name = display_col_name(raw_name)
    #防止重复显示
    if raw_name == x_col or raw_name == y_col:
        continue

    if c.endswith("_PCT_"):
        hover_lines.append(f"{display_name}: %{{customdata[{i}]:.1f}}%")
    else:
        hover_lines.append(f"{display_name}: %{{customdata[{i}]:.2f}}")

fig.update_traces(hovertemplate="<br>".join(hover_lines) + "<extra></extra>")

fig.update_layout(
    height=700,
    xaxis_title="同比" if x_col=="YOY" else "环比" if x_col=="QOQ" else x_col,
    yaxis_title="同比" if y_col=="YOY" else "环比" if y_col=="QOQ" else y_col,
    margin=dict(l=10, r=10, t=40, b=10),
)
fig.update_layout(hoverlabel=dict(font=dict(size=20)))

if is_yoy_qoq_col(x_col):
    fig.update_xaxes(ticksuffix="%")
if is_yoy_qoq_col(y_col):
    fig.update_yaxes(ticksuffix="%")

fig.update_traces(marker=dict(opacity=0.75), selector=dict(mode="markers"))

st.plotly_chart(fig, use_container_width=True)