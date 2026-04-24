"""
02_src/01_data/01_preprocessing/cleaner.py
각 소스 전처리 (드롭/리네임/결측치/스코어 변환)
"""

import os
import sys
import re

_HERE   = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.join(_HERE, "..", "..", "00_common")
if _COMMON not in sys.path:
    sys.path.insert(0, os.path.normpath(_COMMON))

import pandas as pd
from logger import get_logger

logger   = get_logger(__name__)
KEY_COLS = ["ingredient_ko", "ingredient_en"]


def clean_paulaschoice(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.drop(columns=[c for c in cfg["drop_cols"] if c in df.columns])
    df = df.dropna(how="any")
    df = df.rename(columns=cfg["rename_cols"])
    df = df.rename(columns=lambda c: f"pc_{c}" if c not in KEY_COLS else c)
    logger.info(f"[PaulasChoice] 전처리 완료: {df.shape}")
    return df


def clean_coos(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.drop(columns=[c for c in cfg["drop_cols"] if c in df.columns])
    for col, val in cfg["fillna_cols"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    df = df.dropna(how="any")
    df = df.rename(columns=cfg["rename_cols"])
    df = df.rename(columns=lambda c: f"coos_{c}" if c not in KEY_COLS else c)
    logger.info(f"[COOS] 전처리 완료: {df.shape}")
    return df


def clean_hwahae(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.drop(columns=[c for c in cfg["drop_cols"] if c in df.columns])
    for col, val in cfg["fillna_cols"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    df = df.dropna(how="any")
    df = df.rename(columns=cfg["rename_cols"])
    df = df.rename(columns=lambda c: f"hw_{c}" if c not in KEY_COLS else c)
    logger.info(f"[화해] 전처리 완료: {df.shape}")
    return df


def _map_coos_score(val, score_map):
    if pd.isna(val) or str(val).strip() == "":
        return 0
    for keyword, code in score_map.items():
        if keyword in str(val):
            return code
    return 0


def _map_pc_rating(val, rating_map):
    if pd.isna(val) or str(val).strip() == "":
        return 0
    return rating_map.get(str(val).strip(), 0)


def apply_score_mapping(df: pd.DataFrame, pre_cfg: dict) -> pd.DataFrame:
    if "coos_score" in df.columns:
        df["coos_score"] = df["coos_score"].apply(
            lambda v: _map_coos_score(v, pre_cfg["coos_score_map"])
        )
    if "pc_rating" in df.columns:
        df["pc_rating"] = df["pc_rating"].apply(
            lambda v: _map_pc_rating(v, pre_cfg["pc_rating_map"])
        )
    logger.info("[스코어 변환] 완료")
    return df


def parse_ewg_score(raw) -> int:
    """EWG 스코어 파싱 (범위 → 끝값, 없으면 0)"""
    if raw is None:
        return 0
    raw_str = str(raw).strip()
    if raw_str in ("", "nan", "None", "N/A", "-"):
        return 0
    cleaned = re.sub(r"[^\d\-–]", " ", raw_str).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    range_match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", cleaned)
    if range_match:
        return int(range_match.group(2))
    single_match = re.match(r"^(\d+)$", cleaned)
    if single_match:
        return int(single_match.group(1))
    numbers = re.findall(r"\d+", cleaned)
    if numbers:
        return int(numbers[-1])
    return 0


def clean_ewg(df: pd.DataFrame, ing_col: str, score_col: str) -> pd.DataFrame:
    """EWG 원본 → score_parsed + ingredient_key 컬럼 생성, 빈 성분명 제거"""
    df = df.copy()
    df["score_parsed"] = df[score_col].apply(parse_ewg_score)
    df = df[
        df[ing_col].notna() &
        (df[ing_col].astype(str).str.strip() != "")
    ].copy()
    df["ingredient_key"] = df[ing_col].astype(str).str.strip().str.lower()
    logger.info(f"[EWG cleaner] 완료: {df.shape}")
    return df