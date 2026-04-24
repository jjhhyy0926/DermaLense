"""
api_server.py  —  FastAPI 진입점

실행 방법:
    uvicorn api_server:app --reload
"""

import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))

# 04_api 및 의존 패키지 경로 추가
sys.path.insert(0, os.path.join(_ROOT, "02_src", "04_api"))
sys.path.insert(0, os.path.join(_ROOT, "02_src", "02_model", "01_rag_chain"))
sys.path.insert(0, os.path.join(_ROOT, "02_src", "02_model", "02_inference"))

from main import app  # noqa: F401 — uvicorn이 이 app 객체를 사용
