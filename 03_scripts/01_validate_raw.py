"""
03_scripts/01_validate_raw.py
원본 파일 3개의 존재 여부 + 필수 컬럼 스키마를 검증합니다.

실행:
    python 03_scripts/01_validate_raw.py
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(ROOT, "02_src", "00_common"))
sys.path.insert(0, os.path.join(ROOT, "02_src", "01_data", "00_ingestion"))

from config_loader import load_config
from logger import get_logger
from loader import load_all_raw

logger = get_logger(__name__)


def main():
    logger.info("====== [01] raw 데이터 검증 시작 ======")
    cfg = load_config()
    df_pc, df_coos, df_hwahae = load_all_raw(cfg)
    logger.info(f"PaulasChoice : {df_pc.shape}")
    logger.info(f"COOS         : {df_coos.shape}")
    logger.info(f"화해          : {df_hwahae.shape}")
    logger.info("====== [01] 검증 완료 ✅ ======")


if __name__ == "__main__":
    main()
