"""
03_scripts/05_build_ewg_db.py
EWG 성분 DB 빌드 스크립트
실행: python 03_scripts/05_build_ewg_db.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_src", "00_common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_src", "01_data", "00_ingestion"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_src", "01_data", "01_preprocessing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_src", "01_data", "02_io"))

from config_loader import load_config, resolve_path, resolve_output
from loader  import load_ewg
from cleaner import clean_ewg
from merger  import merge_ewg_scores
from writer  import save_csv, save_df_as_json

def main():
    cfg      = load_config()
    raw_dir  = resolve_path(cfg, "raw_dir")
    ewg_cfg  = cfg["preprocessing"]["ewg"]
    val_cfg  = cfg["validation"]["ewg"]

    # 1. 로드
    df_raw = load_ewg(raw_dir, cfg["paths"]["raw_files"]["ewg"], val_cfg["required_cols"])

    # 2. 정제
    df_clean = clean_ewg(df_raw, ewg_cfg["ing_col"], ewg_cfg["score_col"])

    # 3. 병합
    df_merged = merge_ewg_scores(df_clean, ewg_cfg["ing_col"])

    # 4. 저장
    save_csv(df_merged, resolve_output(cfg, "ewg_csv"))


if __name__ == "__main__":
    main()