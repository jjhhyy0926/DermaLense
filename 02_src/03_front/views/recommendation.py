"""
views/recommendation.py
제품 추천 페이지 — product_db.csv 기반 자연어 Q&A
"""
import uuid
import streamlit as st
from services import api
from ui import components
from state import session as sess

_CHIPS = [
    "3만원 이하 토너 추천해줘",
    "트러블 피부에 좋은 제품 있어?",
    "평점 4.5 이상 보습 제품 알려줘",
    "위험 성분 없는 클렌징 추천해줘",
]


def render() -> None:
    st.markdown('<div class="d-page">', unsafe_allow_html=True)
    components.page_header(
        "✨ 맞춤형 제품 추천",
        "카테고리, 가격, 성분 안전성 조건을 자유롭게 말씀해주세요",
    )

    # 세션 ID 초기화
    if not st.session_state.rec_session_id:
        st.session_state.rec_session_id = str(uuid.uuid4())

    _render_empty_state()
    _render_history()
    _render_input()
    _render_reset_button()
    st.markdown('</div>', unsafe_allow_html=True)


def _render_empty_state() -> None:
    if st.session_state.rec_messages:
        return

    st.markdown(
        '''<div style="text-align:center; padding:48px 0 20px;">
          <div style="
            display:inline-flex; align-items:center; justify-content:center;
            width:72px; height:72px; border-radius:50%;
            background:#faf5ff; border:1px solid #e9d5ff;
            font-size:2rem; margin-bottom:16px;">✨</div>
          <div style="font-size:1.1rem; font-weight:700; color:#111827; margin-bottom:6px;">
            어떤 제품을 찾고 계신가요?
          </div>
          <div style="font-size:.875rem; color:#9ca3af; margin-bottom:24px;">
            가격, 카테고리, 피부 타입을 말씀해주시면 안전한 제품을 찾아드립니다
          </div>
        </div>''',
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, chip in enumerate(_CHIPS):
        if cols[i % 2].button(chip, key=f"rec_chip_{i}", use_container_width=True):
            st.session_state.rec_prefill = chip
            st.rerun()


def _render_history() -> None:
    for msg in st.session_state.rec_messages:
        avatar = "✨" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])


def _render_input() -> None:
    prefill    = st.session_state.pop("rec_prefill", None)
    user_input = st.chat_input("예: 3만원 이하 토너 중 안전한 거 추천해줘", key="rec_chat")
    if prefill and not user_input:
        user_input = prefill
    if not user_input:
        return

    st.session_state.rec_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("제품 데이터베이스 검색 중..."):
            try:
                data   = api.recommend_chat(user_input, st.session_state.rec_session_id)
                answer = data["answer"]
                st.markdown(answer)
                st.session_state.rec_messages.append({"role": "assistant", "content": answer})
            except api.APIError as err:
                st.error(str(err))
                st.session_state.rec_messages.append({"role": "assistant", "content": str(err)})
    st.rerun()


def _render_reset_button() -> None:
    if not st.session_state.rec_messages:
        return
    if st.button("🗑️ 대화 초기화", key="reset_rec"):
        sess.reset_recommendation()
        st.rerun()
