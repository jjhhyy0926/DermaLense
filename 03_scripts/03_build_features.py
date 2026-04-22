"""
03_scripts/03_build_features.py
ingredient_merged2.json → ingredient_chunks_preset{1~4}.json 생성

실행:
    python 03_scripts/03_build_features.py
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(ROOT, "02_src", "00_common"))
sys.path.insert(0, os.path.join(ROOT, "02_src", "01_data", "01_preprocessing"))
sys.path.insert(0, os.path.join(ROOT, "02_src", "01_data", "02_io"))

from config_loader import load_config, resolve_output
from logger import get_logger
from reader import load_json
from writer import save_json
from chunker import select_best_rows, build_chunks, validate_chunks

logger = get_logger(__name__)


def main():
    logger.info("====== [03] 피처(청크) 생성 시작 ======")
    cfg    = load_config()
    ch_cfg = cfg["chunking"]

    # 병합 JSON 로드
    data = load_json(resolve_output(cfg, "merged_json"))

    # 최적 행 선별 (1회)
    best_rows = select_best_rows(data, ch_cfg["priority_cols"])

    # 프리셋별 청크 생성 & 저장
    score_labels = ch_cfg["coos_score_label"]

    for preset_id, weights in ch_cfg["weight_presets"].items():
        logger.info(f"--- 프리셋 {preset_id}: 가중치={weights} ---")
        chunks   = build_chunks(best_rows, weights, score_labels)
        out_path = resolve_output(cfg, "chunk_prefix", f"{preset_id}.json")
        save_json(chunks, out_path)
        validate_chunks(chunks, preset_id)

    logger.info("====== [03] 피처 생성 완료 ✅ ======")


if __name__ == "__main__":
    main()
