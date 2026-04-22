"""
02_src/01_data/01_preprocessing/merger.py
3소스 outer join 병합 + product_db 추출
"""

import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.join(_HERE, "..", "..", "00_common")
if _COMMON not in sys.path:
    sys.path.insert(0, os.path.normpath(_COMMON))

import pandas as pd
from logger import get_logger

logger   = get_logger(__name__)
KEY_COLS = ["ingredient_ko", "ingredient_en"]


def merge_sources(df_pc, df_coos, df_hwahae, post_drop_cols):
    for df in [df_pc, df_coos, df_hwahae]:
        for col in KEY_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    df_merged = (
        df_pc
        .merge(df_coos,   on=KEY_COLS, how="outer")
        .merge(df_hwahae, on=KEY_COLS, how="outer")
    )
    df_merged = df_merged.drop(
        columns=[c for c in post_drop_cols if c in df_merged.columns],
        errors="ignore",
    )
    logger.info(f"[병합] 완료: {df_merged.shape}")
    return df_merged


def build_product_db(df_hwahae_raw, product_cfg):
    source_cols = [c for c in product_cfg["source_cols"] if c in df_hwahae_raw.columns]
    df = df_hwahae_raw[source_cols].copy()
    df = df.rename(columns=product_cfg["rename_cols"])
    logger.info(f"[product_db] 생성 완료: {df.shape}")
    return df
