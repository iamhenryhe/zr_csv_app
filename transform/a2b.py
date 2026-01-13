# transform/a2b.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-", "--", "—", "–"}


def _normalize_header(s: str) -> str:
    """
    统一清洗列名，解决 ifind / Excel 导出中的换行、_x000D_、箭头等噪声
    """
    s = str(s)

    # 处理Excel导出常见噪声
    s = s.replace("_x000D_", " ")
    s = s.replace("\r", " ").replace("\n", " ")
    s = s.replace("\u3000", " ")

    # 去引号/箭头等
    s = s.replace('"', "").replace("'", "")
    s = s.replace("↓", "")

    # 去所有空白
    s = re.sub(r"\s+", "", s)

    return s.lower()


def to_number(x):
    if x is None:
        return pd.NA
    if isinstance(x, (int, float)) and pd.notna(x):
        return float(x)

    s = str(x).strip()
    if _normalize_header(s) in MISSING_TOKENS:
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


def _build_col_map(df: pd.DataFrame) -> dict[str, str]:
    m = {}
    for c in df.columns:
        n = _normalize_header(c)
        if n not in m:
            m[n] = c
    return m


def _find_col(df: pd.DataFrame, must: list[str]) -> str:
    """
    在列名(规范化后)中寻找同时包含 must 所有关键词的列
    """
    col_map = _build_col_map(df)
    keys = list(col_map.keys())
    must_n = [_normalize_header(x) for x in must]

    for k in keys:
        if all(token in k for token in must_n):
            return col_map[k]

    raise KeyError(f"找不到列：必须包含 {must}。现有列：{list(df.columns)}")


def _find_mktcap_col(df: pd.DataFrame) -> str:
    """
    - 用总市值去找对应列
    """
    col_map = _build_col_map(df)
    keys = list(col_map.keys())

    # 优先：包含“总市值1”（老版本）
    for k in keys:
        if "总市值1" in k:
            return col_map[k]

    # 其次：包含“总市值”（你当前版本）
    for k in keys:
        if "总市值" in k:
            return col_map[k]

    raise KeyError(f"找不到市值列（包含“总市值”）。现有列：{list(df.columns)}")


def ensure_b_up_to_date(a_path: Path, b_path: Path, force: bool = False) -> bool:
    a_path = Path(a_path)
    b_path = Path(b_path)

    if not a_path.exists():
        raise FileNotFoundError(f"未找到 A.xlsx：{a_path}")

    if force or (not b_path.exists()):
        a2b(a_path, b_path)
        return True

    if a_path.stat().st_mtime > b_path.stat().st_mtime:
        a2b(a_path, b_path)
        return True

    return False


def a2b(a_path: Path, b_path: Path, sheet_name=0) -> Path:
    a_path = Path(a_path)
    b_path = Path(b_path)
    b_path.parent.mkdir(parents=True, exist_ok=True)

    dfA = pd.read_excel(a_path, sheet_name=sheet_name)

    # ====== 必要列 ======
    col_code = _find_col(dfA, ["证券代码"])
    col_name = _find_col(dfA, ["证券简称"])

    col_date = _find_col(dfA, ["业绩预告首次披露日期", "2025年报"])
    col_forecast = _find_col(dfA, ["预告扣非净利润下限", "2025年报"])
    col_q3_cum = _find_col(dfA, ["扣除非经常性损益后归属母公司股东的净利润", "2025三季"])
    col_2024q4 = _find_col(dfA, ["单季度", "扣除非经常性损益后归属母公司股东的净利润", "2024第四季度"])
    col_2025q3 = _find_col(dfA, ["单季度", "扣除非经常性损益后归属母公司股东的净利润", "2025第三季度"])
    col_mktcap = _find_mktcap_col(dfA) 

    out = pd.DataFrame({
        "证券代码": dfA[col_code].astype(str).str.strip(),
        "证券简称": dfA[col_name].astype(str).str.strip(),
        "日期": pd.to_datetime(dfA[col_date], errors="coerce"),
        "预告下限(亿）": dfA[col_forecast].map(to_number),
        "总市值（亿）": dfA[col_mktcap].map(to_number),
        "_q3_cum_": dfA[col_q3_cum].map(to_number),
        "_2024q4_": dfA[col_2024q4].map(to_number),
        "_2025q3_": dfA[col_2025q3].map(to_number),
    })

    # 25Q4
    out["25Q4单季扣非"] = out["预告下限(亿）"] - out["_q3_cum_"]

    # YOY / QOQ
    out["YOY"] = (out["25Q4单季扣非"] / out["_2024q4_"]) - 1
    out["QOQ"] = (out["25Q4单季扣非"] / out["_2025q3_"]) - 1

    # 2025PE / PETTM
    out["2025PE"] = out["总市值（亿）"] / out["预告下限(亿）"]
    denom_ttm = out["_2025q3_"] + 3 * out["25Q4单季扣非"]
    out["PETTM"] = out["总市值（亿）"] / denom_ttm

    need = [
        "证券代码", "证券简称", "日期", "预告下限(亿）", "总市值（亿）",
        "25Q4单季扣非", "YOY", "QOQ", "2025PE", "PETTM",
        "_2024q4_", "_2025q3_", "_q3_cum_"
    ]
    out = out.dropna(subset=need)

    # 分母不能为0
    out = out[
        (out["_2024q4_"] != 0) &
        (out["_2025q3_"] != 0) &
        (out["预告下限(亿）"] != 0) &
        (denom_ttm != 0)
    ].copy()

    out = out[[
        "证券代码",
        "证券简称",
        "日期",
        "预告下限(亿）",
        "总市值（亿）",
        "25Q4单季扣非",
        "YOY",
        "QOQ",
        "2025PE",
        "PETTM",
    ]].copy()

    out.to_excel(b_path, index=False)
    return b_path