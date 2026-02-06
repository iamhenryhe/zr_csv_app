# database.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
import time
from pathlib import Path
import pandas as pd
import streamlit as st


SUPPORTED_EXT = {".csv", ".xlsx", ".xls"}


def _list_data_files(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    files = []
    # 允许：base_dir 下直接放文件；也允许多层子文件夹
    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            # 忽略临时/备份
            if p.name.startswith("~$"):
                continue
            files.append(p)
    return sorted(files, key=lambda x: str(x).lower())


def _read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        # 兼容中文 CSV
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")
    # excel
    return pd.read_excel(path)


def _safe_write_table(df: pd.DataFrame, path: Path) -> None:
    """
    安全写入：先备份 -> 写临时文件 -> replace 覆盖
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}.bak_{ts}{path.suffix}")
    # 只有原文件存在才备份
    if path.exists():
        shutil.copy2(path, backup_path)

    tmp_path = path.with_name(f".__tmp__{path.stem}_{ts}{path.suffix}")

    ext = path.suffix.lower()
    if ext == ".csv":
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(tmp_path, index=False)

    # 原子替换（同目录下最稳）
    tmp_path.replace(path)


def _filter_df(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if not q:
        return df
    q = q.strip().lower()
    if not q:
        return df

    # 在所有列里做包含匹配（字符串化）
    s = df.astype(str).apply(lambda col: col.str.lower().str.contains(q, na=False))
    mask = s.any(axis=1)
    return df.loc[mask].copy()


def render(base_dir: Path) -> None:
    st.title("板块数据库")
    files = _list_data_files(base_dir)

    if not files:
        st.warning(f"未在目录下找到数据文件：{base_dir}\n\n支持：.csv / .xlsx")
        return

    # Sidebar = 数据浏览器
    st.sidebar.header("数据浏览器（板块数据库）")

    # 选择“范围”（文件）——先用文件路径代表范围；后面你再升级成文件夹树也不影响主流程
    rel_paths = [str(p.relative_to(base_dir)) for p in files]
    pick = st.sidebar.selectbox("选择范围（文件）", rel_paths, index=0)
    file_path = base_dir / pick

    # 搜索框（放模块名下面）
    q = st.sidebar.text_input("搜索", value="", placeholder="输入代码/简称/板块/任意关键词")

    # 读取
    try:
        df_raw = _read_table(file_path)
    except Exception as e:
        st.error(f"读取失败：{file_path}\n\n{e}")
        return

    # 指标选择 = columns
    all_cols = list(df_raw.columns)
    default_cols = all_cols[: min(12, len(all_cols))]  # 默认先显示前 12 列，避免太宽
    show_cols = st.sidebar.multiselect("待选指标（展示列）", options=all_cols, default=default_cols)

    if not show_cols:
        st.sidebar.info("请至少选择一个展示列。")
        st.stop()

    # 过滤（搜索）
    df_view = df_raw.copy()
    df_view = _filter_df(df_view, q)

    # 为了“全列可编辑”——编辑器用原 df 的所有列更合理；
    # 但显示可以只显示 show_cols。这里做法：编辑器仍显示 show_cols，
    # 保存时把 show_cols 的修改回写到 df_raw 对应列。
    # 如果你想“表格里就能编辑所有列”，把 editor_df = df_view[show_cols] 改成 df_view 即可。
    editor_df = df_view[show_cols].copy()

    st.subheader("数据表（双击进行编辑）")
    state_key = f"db_editor::{str(file_path)}"
    if state_key not in st.session_state:
        st.session_state[state_key] = editor_df.copy()

    st.session_state[state_key] = editor_df.copy()

    edited = st.data_editor(
        st.session_state[state_key],
        use_container_width=True,
        num_rows="dynamic",    
        key=state_key + "::widget",
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        do_save = st.button("保存更改", type="primary", use_container_width=True)

    with col2:
        do_reload = st.button("放弃更改", use_container_width=True)

    with col3:
        st.download_button(
            "导出现在的CSV",
            data=edited.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{file_path.stem}_view.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if do_reload:
        st.session_state.pop(state_key, None)
        st.rerun()

    if do_save:
        try:
            df_new = df_raw.copy()
            for c in show_cols:
                df_new.loc[df_view.index, c] = edited[c].values

            _safe_write_table(df_new, file_path)
            st.success("已保存并写回本地文件（同时生成 .bak_时间戳 备份）")
        except Exception as e:
            st.error(f"保存失败：{e}")