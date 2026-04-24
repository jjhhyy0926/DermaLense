"""
streamlit_app.py  —  DermaLens 진입점

실행 방법:
    uvicorn api_server:app --reload        # 백엔드 먼저 실행
    streamlit run streamlit_app.py         # 프론트엔드 실행
"""

import sys
import os

# 03_front 패키지 경로 추가
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "02_src", "03_front"))

import streamlit as st

st.set_page_config(
    page_title="DermaLens | AI 스킨케어",
    page_icon="\U0001f33f",
    layout="wide",
    initial_sidebar_state="expanded",
)

from state import session as sess
from ui import styles, navbar
from views import home, analysis, scanner, recommendation

sess.init()
styles.inject()

page = st.query_params.get("page", "home")
navbar.render(page)

_ROUTES: dict = {
    "home":           home.render,
    "analysis":       analysis.render,
    "scanner":        scanner.render,
    "recommendation": recommendation.render,
}

renderer = _ROUTES.get(page, home.render)
renderer()
