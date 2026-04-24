"""
state/session.py
Streamlit 세션 상태 초기화 및 기본값 관리
"""
import streamlit as st
import uuid

_DEFAULTS: dict = {
    # 성분 분석 (Analysis)
    "qa_messages":  [],
    "qa_prefill":   None,
    "skin_type":    None,
    # 성분 스캐너 (Scanner)
    "scan_done":      False,
    "scan_image":     None,
    "scan_results":   None,
    "scan_upload_key": 0,
    # 제품 추천 (Recommendation)
    "rec_messages":   [],
    "rec_session_id": None,
    "rec_prefill":    None,
}


def init() -> None:
    """앱 최초 로드 시 세션 상태 기본값을 설정한다."""
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis() -> None:
    st.session_state.qa_messages = []
    st.session_state.qa_prefill  = None


def reset_scanner() -> None:
    st.session_state.scan_image    = None
    st.session_state.scan_done     = False
    st.session_state.scan_results  = None
    st.session_state.scan_upload_key += 1


def reset_recommendation() -> None:
    st.session_state.rec_messages   = []
    st.session_state.rec_session_id = str(uuid.uuid4())
