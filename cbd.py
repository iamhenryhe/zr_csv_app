from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

DEFAULT_DIR = Path("output") / "master-output" / "sector"
REQUIRED_COLS = {"time_frame", "total_score", "sector"}


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


def render(data_dir: str | Path | None = None):
    st.title("传播度")
    dir_str = st.text_input(
        "数据集路径",
        value=str(data_dir or DEFAULT_DIR),
        key="cbd_data_dir",
    )

    base = Path(dir_str)
    if not base.exists():
        st.error(f"目录不存在：{base}")
        st.stop()

    files = sorted(base.glob("*.csv"))
    if not files:
        st.warning(f"该目录下没有 CSV：{base}")
        st.stop()

    file_names = [f.name for f in files]

    chosen = st.multiselect(
        "选择要合并展示的 CSV（可多选）",
        options=file_names,
        default=file_names,
        key="cbd_choose_files",
    )
    if not chosen:
        st.stop()

    dfs = []
    for name in chosen:
        df = _read_csv(base / name)

        # 列名必须存在
        if not REQUIRED_COLS.issubset(df.columns):
            st.error(f"{name} 缺少列：{sorted(list(REQUIRED_COLS - set(df.columns)))}")
            st.stop()

        df["time_frame"] = df["time_frame"].astype(str).str.strip()
        df["sector"] = df["sector"].astype(str).str.strip()
        df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    x_order = _ordered_unique([str(df["time_frame"].iloc[0]).strip() for df in dfs if not df.empty])
    df_all["time_frame"] = pd.Categorical(df_all["time_frame"], categories=x_order, ordered=True)

    all_sectors = sorted([s for s in df_all["sector"].dropna().unique().tolist() if str(s).strip() != ""])
    pick_sectors = st.multiselect(
        "筛选板块（不选=全部）",
        options=all_sectors,
        default=[],
        key="cbd_sector_filter",
    )
    if pick_sectors:
        df_all = df_all[df_all["sector"].isin(pick_sectors)].copy()

    if df_all.empty:
        st.warning("筛选后数据为空。")
        st.stop()

    fig = px.scatter(
        df_all,
        x="time_frame",
        y="total_score",
        color="sector",
        title="板块打分",
    )
    fig.update_layout(
        xaxis_title="time_frame",
        yaxis_title="total_score",
        height=650,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("数据预览", expanded=False):
        st.dataframe(df_all, use_container_width=True)
