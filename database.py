# database.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st

# =========================
# 目录约定（按你当前结构）
# DB_BASE_DIR/
#   太空光伏/
#     太空光伏.xlsx      (母表：三列 板块/简称/代码)  —— 不在主界面展示
#     公司打分.xlsx
#     动态.xlsx
#   商业航天/
#     商业航天.xlsx
#     公司打分.xlsx
#     动态.xlsx
# =========================
DYNAMIC_FILE = "动态.xlsx"
SCORE_FILE = "公司打分.xlsx"


# -------------------------
# IO: read / write
# -------------------------
@st.cache_data(show_spinner=False)
def _read_table_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    ext = path.suffix.lower()
    if ext == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")
    return pd.read_excel(path)


def _read_table(path: Path) -> pd.DataFrame:
    return _read_table_cached(str(path))


def _safe_write_table(df: pd.DataFrame, path: Path) -> None:
    """保存才写回：先备份 -> 写临时 -> replace 覆盖"""
    path.parent.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    if path.exists():
        backup_path = path.with_name(f"{path.stem}.bak_{ts}{path.suffix}")
        shutil.copy2(path, backup_path)

    tmp_path = path.with_name(f".__tmp__{path.stem}_{ts}{path.suffix}")
    ext = path.suffix.lower()
    if ext == ".csv":
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(tmp_path, index=False)

    tmp_path.replace(path)
    _read_table_cached.clear()


# -------------------------
# Utils
# -------------------------
def _list_sector_dirs(base_dir: Path) -> List[str]:
    if not base_dir.exists():
        return []
    dirs = []
    for p in base_dir.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            dirs.append(p.name)
    return sorted(dirs)


def _filter_df_any(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if not q:
        return df
    q = q.strip().lower()
    if not q:
        return df
    s = df.astype(str).apply(lambda col: col.str.lower().str.contains(q, na=False))
    return df.loc[s.any(axis=1)].copy()


def _load_or_empty(path: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    if not path.exists():
        return pd.DataFrame(), f"未找到文件：{path}"
    try:
        return _read_table(path), None
    except Exception as e:
        return pd.DataFrame(), f"读取失败：{path}\n\n{e}"


def _guess_master_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    """母表三列：板块/简称/代码（尽量自动识别）"""
    cols = list(df.columns)

    def pick(cands):
        for c in cols:
            cc = str(c).strip()
            if cc in cands:
                return c
        for c in cols:
            cc = str(c).strip()
            for k in cands:
                if k in cc:
                    return c
        return None

    col_sector = pick(["板块", "所属板块", "行业", "主题"])
    col_name = pick(["简称", "证券简称", "公司简称", "名称", "公司"])
    col_code = pick(["代码", "证券代码", "股票代码", "ticker", "secid"])

    if not (col_sector and col_name and col_code):
        raise ValueError(f"母表列识别失败。当前列：{cols}。需要包含：板块/简称/代码（或近似列名）")
    return str(col_sector), str(col_name), str(col_code)


def _normalize_code(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _dedup_keep_last(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp[key_col] = tmp[key_col].map(_normalize_code)
    tmp = tmp[tmp[key_col] != ""]
    return tmp.drop_duplicates(subset=[key_col], keep="last").reset_index(drop=True)


# -------------------------
# UI blocks
# -------------------------
def _editor_save_block(
    title: str,
    df_raw: pd.DataFrame,
    file_path: Path,
    search_q: str = "",
    height: int | None = None,
    allow_row_ops: bool = True,
    key_prefix: str = "",
) -> pd.DataFrame:
    """展示/编辑/保存一个表。默认全列可编辑。"""
    st.subheader(title)

    if df_raw.empty:
        st.info("（空表或文件缺失）")
        return df_raw

    df_view = _filter_df_any(df_raw, search_q) if search_q else df_raw.copy()

    state_key = f"{key_prefix}::editor::{str(file_path)}::{title}"
    st.session_state[state_key] = df_view.copy()

    edited = st.data_editor(
        st.session_state[state_key],
        use_container_width=True,
        num_rows="dynamic" if allow_row_ops else "fixed",
        height=height,
        key=state_key + "::widget",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        do_save = st.button("保存更改", type="primary", use_container_width=True, key=state_key + "::save")
    with c2:
        do_reload = st.button("放弃更改", use_container_width=True, key=state_key + "::reload")
    with c3:
        st.download_button(
            "导出当前视图 CSV",
            data=edited.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{file_path.stem}_{title}_view.csv",
            mime="text/csv",
            use_container_width=True,
            key=state_key + "::dl",
        )

    if do_reload:
        st.rerun()

    if do_save:
        try:
            # 保存策略：
            # - 无搜索：覆盖整表（允许增删行）
            # - 有搜索：不允许增删行，只做局部回写
            if not search_q:
                df_new = edited.copy()
            else:
                if len(edited) != len(df_view):
                    raise ValueError("搜索过滤状态下不允许增删行。请清空搜索后再增删行。")
                if list(edited.columns) != list(df_view.columns):
                    raise ValueError("过滤视图列结构发生变化，无法安全保存。")

                df_new = df_raw.copy()
                df_new.loc[df_view.index, :] = edited.values

            _safe_write_table(df_new, file_path)
            st.success(f"✅ 已保存：{file_path.name}（并生成 .bak_时间戳 备份）")
            return df_new
        except Exception as e:
            st.error(f"保存失败：{e}")
            return df_raw

    return edited


def _move_company_between_sectors(
    base_dir: Path,
    src_sector: str,
    dst_sector: str,
    code: str,
    name_hint: str | None = None,
    move_score_rows: bool = True,
) -> None:
    """
    迁移逻辑：
    -母表需要删掉该代码行
    -母表追加代码行
    """
    src_dir = base_dir / src_sector
    dst_dir = base_dir / dst_sector

    src_master = src_dir / f"{src_sector}.xlsx"
    dst_master = dst_dir / f"{dst_sector}.xlsx"

    df_src, err1 = _load_or_empty(src_master)
    if err1:
        raise FileNotFoundError(err1)

    df_dst, err2 = _load_or_empty(dst_master)
    if err2:
        df_dst = pd.DataFrame(columns=list(df_src.columns))

    col_sector, col_name, col_code = _guess_master_cols(df_src)

    # 对齐目标列
    if list(df_dst.columns) != list(df_src.columns):
        for c in df_src.columns:
            if c not in df_dst.columns:
                df_dst[c] = pd.NA
        df_dst = df_dst[df_src.columns.tolist()]

    code_n = _normalize_code(code)
    if not code_n:
        raise ValueError("代码为空，无法迁移。")

    src2 = df_src.copy()
    src2[col_code] = src2[col_code].map(_normalize_code)
    hit = src2[src2[col_code] == code_n]
    if hit.empty:
        raise ValueError(f"源板块【{src_sector}】母表未找到代码：{code_n}")

    row = hit.iloc[-1].copy()
    row[col_sector] = dst_sector
    if name_hint and (pd.isna(row[col_name]) or str(row[col_name]).strip() == ""):
        row[col_name] = name_hint

    # 源删、目标加（去重）
    df_src_new = src2[src2[col_code] != code_n].copy()

    dst2 = df_dst.copy()
    dst2[col_code] = dst2[col_code].map(_normalize_code)
    dst2 = pd.concat([dst2, pd.DataFrame([row])], ignore_index=True)
    df_dst_new = _dedup_keep_last(dst2, col_code)

    _safe_write_table(df_src_new, src_master)
    _safe_write_table(df_dst_new, dst_master)

    if not move_score_rows:
        return

    src_score = src_dir / SCORE_FILE
    dst_score = dst_dir / SCORE_FILE

    df_sc_src, e1 = _load_or_empty(src_score)
    if e1:
        return  # 源没有公司打分，跳过

    if dst_score.exists():
        df_sc_dst, _ = _load_or_empty(dst_score)
    else:
        df_sc_dst = pd.DataFrame(columns=df_sc_src.columns)

    # 找代码列（包含“代码”的那列）
    code_cols = [c for c in df_sc_src.columns if "代码" in str(c)]
    if not code_cols:
        raise ValueError(f"公司打分表找不到代码列。列名：{list(df_sc_src.columns)}")
    sc_code_col = code_cols[0]

    tmp = df_sc_src.copy()
    tmp[sc_code_col] = tmp[sc_code_col].map(_normalize_code)
    to_move = tmp[tmp[sc_code_col] == code_n].copy()
    remain = tmp[tmp[sc_code_col] != code_n].copy()

    df_sc_dst2 = pd.concat([df_sc_dst, to_move], ignore_index=True)

    _safe_write_table(remain, src_score)
    _safe_write_table(df_sc_dst2, dst_score)


def _build_code_candidates_from_master(master_path: Path) -> List[str]:
    """用于迁移下拉：优先从母表生成 '简称（代码）' 列表"""
    df_m, em = _load_or_empty(master_path)
    if em or df_m.empty:
        return []
    try:
        _, col_name, col_code = _guess_master_cols(df_m)
        tmp = df_m.copy()
        tmp[col_code] = tmp[col_code].map(_normalize_code)
        tmp[col_name] = tmp[col_name].astype(str)
        tmp = tmp.dropna(subset=[col_code])
        tmp = tmp[tmp[col_code].astype(str).str.strip() != ""]
        tmp = tmp.drop_duplicates(subset=[col_code], keep="last")
        return sorted([f"{r[col_name]}（{r[col_code]}）" for _, r in tmp.iterrows()])
    except Exception:
        return []


# -------------------------
# Main
# -------------------------
def render(base_dir: Path) -> None:
    st.title("板块数据库")

    if not base_dir.exists():
        st.error(f"目录不存在：{base_dir}")
        st.caption("可通过环境变量 DB_BASE_DIR 指向你的板块数据库根目录。")
        return

    sector_dirs = _list_sector_dirs(base_dir)
    if not sector_dirs:
        st.warning(f"目录下未发现板块文件夹：{base_dir}")
        return

    # ========== Sidebar ==========
    st.sidebar.header("范围选择（板块）")
    picked_sector = st.sidebar.selectbox("选择板块文件夹", sector_dirs, index=0)

    st.sidebar.subheader("公司打分搜索")
    q_company = st.sidebar.text_input(
        "搜索（代码/简称/任意字段）",
        value="",
        placeholder="只过滤公司打分表"
    )

    # 迁移：默认隐藏（expander + 勾选启用）
    enable_move = False
    dst_sector = None
    move_score_rows = True
    move_label = None
    manual_code = ""

    with st.sidebar.expander("动态调整个股的板块归类", expanded=False):
        enable_move = st.checkbox("启用迁移功能", value=False)
        st.caption("把某公司从当前板块迁移到目标板块")

        if enable_move:
            dst_sector = st.selectbox("迁移到（目标板块）", [s for s in sector_dirs if s != picked_sector], index=0)
            move_score_rows = st.checkbox("同时迁移公司打分记录", value=True)

            master_path = (base_dir / picked_sector / f"{picked_sector}.xlsx")
            candidates = _build_code_candidates_from_master(master_path)

            if candidates:
                move_label = st.selectbox("选择公司", candidates, index=0)
            else:
                manual_code = st.text_input("公司代码（必填）", value="", placeholder="例如：300751 或 300751.SZ")

    # ========== Paths ==========
    sector_dir = base_dir / picked_sector
    dyn_path = sector_dir / DYNAMIC_FILE
    score_path = sector_dir / SCORE_FILE
    master_path = sector_dir / f"{picked_sector}.xlsx"

    # ========== Main View ==========
    st.markdown(f"## 当前板块：{picked_sector}")

    df_dyn, err_dyn = _load_or_empty(dyn_path)
    if err_dyn:
        st.warning(err_dyn)
    else:
        _editor_save_block(
            title="动态（产业趋势XXXX）",
            df_raw=df_dyn,
            file_path=dyn_path,
            search_q="",
            height=260,
            allow_row_ops=True,
            key_prefix="dyn",
        )

    st.divider()

    df_score, err_score = _load_or_empty(score_path)
    if err_score:
        st.warning(err_score)
    else:
        _editor_save_block(
            title="公司打分（增量，XXX）",
            df_raw=df_score,
            file_path=score_path,
            search_q=q_company,
            height=420,
            allow_row_ops=True,
            key_prefix="score",
        )

    # ========== 主界面：迁移也折叠 ==========
    with st.expander("动态调整个股的板块归类", expanded=False):
        if not enable_move:
            st.info("使用工作台使用编辑功能")
        else:
            if move_label and "（" in move_label and move_label.endswith("）"):
                name_hint = move_label.split("（")[0].strip()
                code = move_label.split("（")[-1].rstrip("）").strip()
            else:
                name_hint = None
                code = (manual_code or "").strip()

            if not code:
                st.warning("请先选择公司或输入公司代码。")
            else:
                st.write(f"将迁移：**{name_hint or ''} {code}**")
                st.write(f"从：**{picked_sector}** → 到：**{dst_sector}**")
                st.write(f"同时迁移公司打分记录：**{move_score_rows}**")

                if st.button("执行迁移", type="primary", use_container_width=True):
                    try:
                        _move_company_between_sectors(
                            base_dir=base_dir,
                            src_sector=picked_sector,
                            dst_sector=dst_sector,
                            code=code,
                            name_hint=name_hint,
                            move_score_rows=move_score_rows,
                        )
                        st.success(f"✅ 已迁移：{code}  {picked_sector} → {dst_sector}")
                        st.caption("相关文件已自动备份（.bak_时间戳）。")
                        st.rerun()
                    except Exception as e:
                        st.error(f"迁移失败：{e}")
