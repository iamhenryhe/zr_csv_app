from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
import streamlit as st
import plotly.express as px

# 结构：
# output/master-output/{sector|company}/total-score/plot/
# 里面包含：
#   1/2/3-YYYY-MM-DD.csv   -> 时间分段
#   t-YYYY-MM-DD.csv       -> 汇总
BASE_DIR = Path("master-output") 

TYPE_OPTIONS = ["sector", "company"]
TYPE_LABELS = {"sector": "板块", "company": "个股"}

COL_TIME = "时间"
COL_SCORE = "得分"
COL_SECTOR = "板块"
COL_COMPANY = "个股"

SOURCE_OPTIONS = ["时间分段", "汇总"]
SOURCE_LABELS = {"时间分段": "时间分段（123）", "汇总": "汇总（T）"}

SLOT_OPTIONS = ["1", "2", "3"]


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
    return name.strip().lower().startswith("t-")


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _extract_date_from_name(name: str):
    m = _DATE_RE.search(name)
    if not m:
        return None
    try:
        return pd.to_datetime(m.group(1)).date()
    except Exception:
        return None


def _extract_slot_from_name(name: str):
    n = name.strip().lower()
    if _is_t_file(n):
        return None
    m = re.match(r"^([123])-", n)
    return m.group(1) if m else None


def _slot_rank(filename: str) -> int:
    if _is_t_file(filename):
        return 9
    s = _extract_slot_from_name(filename)
    return int(s) if s in {"1", "2", "3"} else 8


def _build_baseline_path(base: Path, source: str, baseline_date, baseline_slot: str | None):
    """根据 source + baseline_date(+slot) 生成基准文件路径"""
    d = pd.to_datetime(baseline_date).date()
    if source == "汇总":
        return base / f"t-{d}.csv"
    # 时间分段
    return base / f"{baseline_slot}-{d}.csv"


def render():
    st.title("传播度")

    # =========================
    # Sidebar：数据类型 / 数据来源 / 分段 / 时间范围
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
        format_func=lambda x: SOURCE_LABELS.get(x, x),
        horizontal=True,
        key="cbd_source",
    )

    cn = TYPE_LABELS.get(data_type, data_type)
    group_col = COL_SECTOR if data_type == "sector" else COL_COMPANY

    base = BASE_DIR / data_type / "total-score" / "plot"
    st.sidebar.caption(f"当前路径：{base}")

    slot_pick = None
    if source == "时间分段":
        slot_pick = st.sidebar.multiselect(
            "选择分段（可多选）",
            options=SLOT_OPTIONS,
            default=SLOT_OPTIONS,
            key="cbd_slot_pick",
        )
        if not slot_pick:
            st.sidebar.warning("未选择任何分段。")
            st.stop()

    st.sidebar.subheader("时间范围")
    c1, c2 = st.sidebar.columns(2)
    start_date = c1.date_input("开始", key="cbd_start_date")
    end_date = c2.date_input("结束", key="cbd_end_date")
    if start_date > end_date:
        st.sidebar.error("开始日期不能晚于结束日期。")
        st.stop()

    # =========================
    # TopN（sidebar）
    # =========================
    st.sidebar.subheader("TopN（基于某一天舆情生成近期趋势）")
    topn_on = st.sidebar.checkbox("启用 Top-N 筛选", value=False, key="cbd_topn_on")
    topn_n = None
    baseline_date = None
    baseline_slot = None

    if topn_on:
        topn_n = st.sidebar.number_input("N（默认10）", min_value=1, max_value=5000, value=10, step=5, key="cbd_topn_n")
        baseline_date = st.sidebar.date_input("基准日期（指定TopN基准）", key="cbd_topn_date")

        if source == "时间分段":
            baseline_slot = st.sidebar.selectbox("基准分段", options=SLOT_OPTIONS, index=0, key="cbd_topn_slot")

    # =========================
    # 列出目录下 CSV
    # =========================
    if not base.exists():
        st.error(f"目录不存在：{base}")
        st.stop()

    all_files = sorted(base.glob("*.csv"))
    if not all_files:
        st.warning(f"该目录下没有 CSV：{base}")
        st.stop()

    # 先按来源过滤
    if source == "汇总":
        files_use = [f for f in all_files if _is_t_file(f.name)]
    else:
        files_use = [f for f in all_files if not _is_t_file(f.name)]
        files_use = [f for f in files_use if (_extract_slot_from_name(f.name) in set(slot_pick))]

    # 再按文件名日期过滤
    filtered = []
    for f in files_use:
        d = _extract_date_from_name(f.name)
        if d is None:
            continue
        if start_date <= d <= end_date:
            filtered.append((d, f))

    filtered.sort(key=lambda x: (x[0], _slot_rank(x[1].name)))
    files_final = [f for _, f in filtered]

    st.caption(f"自动选中 CSV 数量：{len(files_final)}（{start_date} ~ {end_date}）")
    if not files_final:
        st.warning("该条件下没有匹配的 CSV。")
        st.stop()

    # =========================
    # 读取
    # =========================
    required = {COL_TIME, COL_SCORE, group_col}
    dfs = []
    for p in files_final:
        df = _read_csv(p)
        if not required.issubset(df.columns):
            st.error(f"{p.name} 缺少列：{sorted(list(required - set(df.columns)))}")
            st.stop()

        df[COL_TIME] = df[COL_TIME].astype(str).str.strip()
        df[group_col] = df[group_col].astype(str).str.strip()
        df[COL_SCORE] = pd.to_numeric(df[COL_SCORE], errors="coerce")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # 用文件名生成 x_order
    x_order = []
    for p in files_final:
        d = _extract_date_from_name(p.name)
        if d is None:
            continue
        if source == "汇总":
            x_order.append(f"{d}/t")
        else:
            slot = _extract_slot_from_name(p.name) or ""
            x_order.append(f"{d}/{slot}")


    df_all[COL_TIME] = pd.Categorical(df_all[COL_TIME], categories=_ordered_unique(x_order), ordered=True)

    # =========================
    # TopN 过滤：用“基准CSV”算名单，再过滤 df_all
    # =========================
    if topn_on:
        baseline_path = _build_baseline_path(base, source, baseline_date, baseline_slot)

        if not baseline_path.exists():
            st.error(f"TopN 基准文件不存在：{baseline_path.name}（请检查基准日期/分段是否有文件）")
            st.stop()

        df_base = _read_csv(baseline_path)
        if not required.issubset(df_base.columns):
            st.error(f"TopN 基准文件列不匹配：{baseline_path.name} 缺少 {sorted(list(required - set(df_base.columns)))}")
            st.stop()

        df_base[group_col] = df_base[group_col].astype(str).str.strip()
        df_base[COL_SCORE] = pd.to_numeric(df_base[COL_SCORE], errors="coerce")
        df_base = df_base.dropna(subset=[group_col, COL_SCORE])

        top_list = (
            df_base.sort_values(COL_SCORE, ascending=False)[group_col]
            .dropna()
            .astype(str)
            .head(int(topn_n))
            .tolist()
        )
        top_set = set(top_list)

        st.sidebar.caption(f"TopN 基准：{baseline_path.name}（Top{int(topn_n)}）")

        df_all = df_all[df_all[group_col].isin(top_set)].copy()

        if df_all.empty:
            st.warning("TopN 过滤后数据为空（说明基准TopN名单在该时间范围内没有对应记录）。")
            st.stop()

    # =========================
    # 添加 / 删除（板块/个股）
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
    title_suffix = SOURCE_LABELS[source]
    if source == "时间分段":
        title_suffix += f"｜分段{','.join(slot_pick)}"
    if topn_on:
        title_suffix += f"｜Top{int(topn_n)}@{baseline_date}"

    with st.expander("散点图（可以辅助TOP-K选择）", expanded=False):
        fig = px.scatter(
            df_all,
            x=COL_TIME,
            y=COL_SCORE,
            color=group_col,
            title=f"{cn}打分（{title_suffix}）",
        )
        fig.update_xaxes(categoryorder="array", categoryarray=_ordered_unique(x_order))
        fig.update_layout(
            xaxis_title=COL_TIME,
            yaxis_title=COL_SCORE,
            height=650,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


    st.subheader("趋势折线图")
    df_line = df_all.dropna(subset=[COL_TIME, COL_SCORE, group_col]).copy()

    fig_line = px.line(
        df_line,
        x=COL_TIME,
        y=COL_SCORE,
        color=group_col,
        markers=True,
        title=f"{cn}趋势变化（{title_suffix}）",
    )
    fig_line.update_xaxes(categoryorder="array", categoryarray=_ordered_unique(x_order))
    fig_line.update_layout(
        xaxis_title=COL_TIME,
        yaxis_title=COL_SCORE,
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    with st.expander("数据预览", expanded=False):
        st.dataframe(df_all, use_container_width=True)
