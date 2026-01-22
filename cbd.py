from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

# 新结构：
# output/master-output/{sector|company}/total-score/plot/
# 里面同时包含：123 批次 csv + t 汇总 csv（同一层，不再有 t-files）
BASE_DIR = Path("output") / "master-output"

TYPE_OPTIONS = ["sector", "company"]
TYPE_LABELS = {"sector": "板块", "company": "个股"}

COL_TIME = "时间"
COL_SCORE = "得分"
COL_SECTOR = "板块"
COL_COMPANY = "个股"

SOURCE_OPTIONS = ["全部", "仅123", "仅T"]


def _read_csv(p: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(p)


def _ordered_unique(seq: list[str]) -> list[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _is_t_file(name: str) -> bool:
    # 规则：文件名以 t 开头就认为是 T 汇总（例如 t-2026-01-10.csv）
    # 你如果规则不同（比如包含 "_t_"），改这一行即可
    n = name.strip().lower()
    return n.startswith("t")


def render():
    st.title("传播度")

    # =========================
    # Sidebar：板块/个股 + 数据来源 + 时间段
    # =========================
    data_type = st.sidebar.radio(
        "数据类型",
        TYPE_OPTIONS,
        format_func=lambda x: TYPE_LABELS.get(x, x),
        horizontal=True,
        key="cbd_data_type",
    )

    source = st.sidebar.radio(
        "数据来源",
        SOURCE_OPTIONS,
        horizontal=True,
        key="cbd_source",
    )

    cn = TYPE_LABELS.get(data_type, data_type)
    group_col = COL_SECTOR if data_type == "sector" else COL_COMPANY

    base = BASE_DIR / data_type / "total-score" / "plot"
    st.sidebar.caption(f"当前路径：{base}")

    time_filter_on = st.sidebar.checkbox("按时间段筛选", value=False, key="cbd_time_filter_on")
    start_date = end_date = None
    if time_filter_on:
        c1, c2 = st.sidebar.columns(2)
        start_date = c1.date_input("开始", key="cbd_start_date")
        end_date = c2.date_input("结束", key="cbd_end_date")

    # =========================
    # 文件选择（同一层包含 123 + T）
    # =========================
    if not base.exists():
        st.error(f"目录不存在：{base}")
        st.stop()

    files = sorted(base.glob("*.csv"))
    if not files:
        st.warning(f"该目录下没有 CSV：{base}")
        st.stop()

    # 先按来源过滤文件列表
    files_123 = [f for f in files if not _is_t_file(f.name)]
    files_t = [f for f in files if _is_t_file(f.name)]

    if source == "仅123":
        files_use = files_123
    elif source == "仅T":
        files_use = files_t
    else:
        files_use = files

    if not files_use:
        st.warning("当前数据来源下没有可用 CSV。")
        st.stop()

    file_names = [f.name for f in files_use]
    chosen = st.multiselect(
        "选择要合并展示的 CSV（可多选）",
        options=file_names,
        default=file_names,
        key=f"cbd_choose_files_{data_type}_{source}",
    )
    if not chosen:
        st.stop()

    # =========================
    # 读取 + 校验
    # =========================
    required = {COL_TIME, COL_SCORE, group_col}
    dfs = []
    for name in chosen:
        df = _read_csv(base / name)

        if not required.issubset(df.columns):
            st.error(f"{name} 缺少列：{sorted(list(required - set(df.columns)))}")
            st.stop()

        df[COL_TIME] = df[COL_TIME].astype(str).str.strip()
        df[group_col] = df[group_col].astype(str).str.strip()
        df[COL_SCORE] = pd.to_numeric(df[COL_SCORE], errors="coerce")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # =========================
    # X轴顺序（按你选中的文件顺序）
    # =========================
    x_order = _ordered_unique([str(df[COL_TIME].iloc[0]).strip() for df in dfs if not df.empty])
    df_all[COL_TIME] = pd.Categorical(df_all[COL_TIME], categories=x_order, ordered=True)

    # =========================
    # 时间段过滤
    # =========================
    if time_filter_on:
        t = pd.to_datetime(df_all[COL_TIME], errors="coerce")
        df_all = df_all[t.notna()].copy()
        t = pd.to_datetime(df_all[COL_TIME], errors="coerce")

        if start_date and end_date:
            df_all = df_all[(t.dt.date >= start_date) & (t.dt.date <= end_date)].copy()

    # =========================
    # 添加 / 删除
    # =========================
    all_groups = sorted([s for s in df_all[group_col].dropna().unique().tolist() if str(s).strip() != ""])
    col_keep, col_drop = st.columns(2)

    with col_keep:
        keep_groups = st.multiselect(
            f"添加（只看这些{cn}）",
            options=all_groups,
            default=[],
            key=f"cbd_keep_{data_type}_{source}",
        )

    with col_drop:
        drop_groups = st.multiselect(
            f"删除（排除这些{cn}）",
            options=all_groups,
            default=[],
            key=f"cbd_drop_{data_type}_{source}",
        )

    if keep_groups:
        df_all = df_all[df_all[group_col].isin(keep_groups)].copy()
    if drop_groups:
        df_all = df_all[~df_all[group_col].isin(drop_groups)].copy()

    if df_all.empty:
        st.warning("筛选后数据为空。")
        st.stop()

    # =========================
    # 可视化
    # =========================
    fig = px.scatter(
        df_all,
        x=COL_TIME,
        y=COL_SCORE,
        color=group_col,
        title=f"{cn}打分（{source}）",
    )
    fig.update_layout(
        xaxis_title=COL_TIME,
        yaxis_title=COL_SCORE,
        height=650,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("数据预览", expanded=False):
        st.dataframe(df_all, use_container_width=True)
