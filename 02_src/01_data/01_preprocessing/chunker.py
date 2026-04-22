"""
02_src/01_data/01_preprocessing/chunker.py
4가지 가중치 프리셋 청크 생성 + 검증
"""

import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.join(_HERE, "..", "..", "00_common")
if _COMMON not in sys.path:
    sys.path.insert(0, os.path.normpath(_COMMON))

import numpy as np
from collections import Counter, defaultdict
from logger import get_logger

logger = get_logger(__name__)


def is_valid(val) -> bool:
    if val is None:
        return False
    s = str(val).strip()
    if s in ("", "nan", "NaN", "None", "없음", "0"):
        return False
    try:
        if np.isnan(float(s)):
            return False
    except (ValueError, TypeError):
        pass
    return True


def select_best_rows(data: list, priority_cols: list) -> dict:
    groups = defaultdict(list)
    for row in data:
        groups[row.get("ingredient_ko", "")].append(row)

    best = {
        ing: max(rows, key=lambda r: sum(1 for c in priority_cols if is_valid(r.get(c))))
        for ing, rows in groups.items()
    }
    dup = {k: v for k, v in Counter(r.get("ingredient_ko", "") for r in data).items() if v > 1}
    logger.info(f"[선별] {len(data)}행 → {len(best)}개 성분 | 중복: {len(dup)}개")
    return best


def build_chunks(best_rows: dict, weights: dict, score_label_map: dict) -> list:
    chunks = []
    for ing, row in best_rows.items():
        ingredient    = str(row.get("ingredient_ko") or "")
        ingredient_en = str(row.get("ingredient_en") or "")
        base_meta = {
            "ingredient_ko": ingredient, "ingredient_en": ingredient_en,
            "coos_score": row.get("coos_score"), "hw_ewg": row.get("hw_ewg"),
            "pc_rating": row.get("pc_rating"),
        }

        # EWG 청크
        ewg_parts = []
        if is_valid(row.get("coos_score")):
            label = score_label_map.get(str(row["coos_score"]), str(row["coos_score"]))
            ewg_parts.append(f"EWG 스코어: {label} ({row['coos_score']}등급)")
        if is_valid(row.get("hw_ewg")):
            ewg_parts.append(f"화해 EWG: {row['hw_ewg']}")
        if ewg_parts:
            chunks.append({"page_content": f"[{ingredient}] " + " / ".join(ewg_parts),
                           "metadata": {**base_meta, "chunk_type": "ewg",
                                        "chunk_weight": weights["ewg"]}})

        # Basic Info 청크
        basic_parts = []
        for col, label in [("coos_function", "기능"), ("pc_effect", "효과"), ("hw_purpose", "목적")]:
            if is_valid(row.get(col)):
                basic_parts.append(f"{label}: {row[col]}")
        if basic_parts:
            chunks.append({"page_content": f"[{ingredient}] " + " / ".join(basic_parts),
                           "metadata": {**base_meta, "chunk_type": "basic_info",
                                        "chunk_weight": weights["basic_info"]}})

        # Expert 청크
        expert_parts = []
        if is_valid(row.get("pc_description")):
            expert_parts.append(f"설명: {row['pc_description']}")
        if is_valid(row.get("coos_kr_restricted")):
            expert_parts.append(f"규제: {row['coos_kr_restricted']}")
        if expert_parts:
            chunks.append({"page_content": f"[{ingredient}] " + " | ".join(expert_parts),
                           "metadata": {**base_meta, "chunk_type": "expert",
                                        "chunk_weight": weights["expert"]}})

    logger.info(f"[청크 생성] {len(best_rows)}개 성분 → {len(chunks)}개 청크")
    return chunks


def validate_chunks(chunks: list, preset_id: int) -> None:
    type_map = defaultdict(list)
    for c in chunks:
        m = c.get("metadata", {})
        type_map[m.get("chunk_type", "unknown")].append(m.get("chunk_weight", 0))

    for t in ("ewg", "basic_info", "expert"):
        ws   = type_map.get(t, [])
        avg  = sum(ws) / len(ws) if ws else 0
        ings = [c["metadata"]["ingredient_ko"] for c in chunks
                if c["metadata"].get("chunk_type") == t]
        dups   = {k: v for k, v in Counter(ings).items() if v > 1}
        status = "✅ 중복 없음" if not dups else f"❌ 중복 {len(dups)}건"
        logger.info(f"  프리셋{preset_id} | {t:12}: {len(ws):5}개 weight={avg:.2f} | {status}")
