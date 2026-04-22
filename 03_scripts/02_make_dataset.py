"""
03_scripts/02_make_dataset.py
전처리 파이프라인 전체를 실행합니다.
  raw → ingredient_merged2.json + product_db.csv

실행:
    python 03_scripts/02_make_dataset.py
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(ROOT, "02_src", "00_common"))
sys.path.insert(0, os.path.join(ROOT, "02_src", "01_data", "00_ingestion"))
sys.path.insert(0, os.path.join(ROOT, "02_src", "01_data", "01_preprocessing"))
sys.path.insert(0, os.path.join(ROOT, "02_src", "01_data", "02_io"))

from config_loader import load_config, resolve_output
from logger import get_logger
from loader import load_all_raw
from cleaner import clean_paulaschoice, clean_coos, clean_hwahae, apply_score_mapping
from merger import merge_sources, build_product_db
from writer import save_df_as_json, save_csv

logger = get_logger(__name__)


def main():
    logger.info("====== [02] 데이터셋 생성 시작 ======")
    cfg     = load_config()
    pre_cfg = cfg["preprocessing"]

    # Step 1. 원본 로드
    logger.info("--- Step 1: 원본 로드 ---")
    df_pc_raw, df_coos_raw, df_hwahae_raw = load_all_raw(cfg)

    # Step 2. product_db 생성 (원본 화해 기준)
    logger.info("--- Step 2: product_db 생성 ---")
    df_product = build_product_db(df_hwahae_raw, cfg["product_db"])
    save_csv(df_product, resolve_output(cfg, "product_db"))

    # Step 3. 각 소스 전처리
    logger.info("--- Step 3: 각 소스 전처리 ---")
    df_pc     = clean_paulaschoice(df_pc_raw.copy(),   pre_cfg["paulaschoice"])
    df_coos   = clean_coos(df_coos_raw.copy(),         pre_cfg["coos"])
    df_hwahae = clean_hwahae(df_hwahae_raw.copy(),     pre_cfg["hwahae"])

    # Step 4. 병합
    logger.info("--- Step 4: 병합 ---")
    df_merged = merge_sources(
        df_pc, df_coos, df_hwahae,
        pre_cfg["post_merge_drop_cols"],
    )

    # Step 5. 스코어 수치화
    logger.info("--- Step 5: 스코어 수치화 ---")
    df_merged = apply_score_mapping(df_merged, pre_cfg)

    # Step 6. JSON 저장
    logger.info("--- Step 6: JSON 저장 ---")
    save_df_as_json(df_merged, resolve_output(cfg, "merged_json"))

    logger.info("====== [02] 데이터셋 생성 완료 ✅ ======")


if __name__ == "__main__":
    main()
