import sys
import os
import traceback
import logging

# 02_inference 경로 추가
_INF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "02_model", "02_inference"))
sys.path.insert(0, _INF_DIR)

from fastapi import APIRouter, HTTPException
from schemas import RecommendChatRequest, RecommendChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_sessions: dict[str, list] = {}

try:
    from product_ai import chat as _chat
    logger.info("product_ai import 성공")
except Exception as e:
    logger.error(f"product_ai import 실패: {e}\n{traceback.format_exc()}")
    _chat = None


@router.post("/recommend/chat", response_model=RecommendChatResponse)
async def recommend_chat(req: RecommendChatRequest):
    if _chat is None:
        raise HTTPException(status_code=500, detail="product_ai 모듈 로드 실패 — 서버 로그 확인")

    history = _sessions.get(req.session_id, [])
    try:
        answer, updated = _chat(req.message, history)
    except Exception as e:
        logger.error(f"[recommend/chat] 오류:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    _sessions[req.session_id] = updated
    return RecommendChatResponse(answer=answer, session_id=req.session_id)


@router.delete("/recommend/chat/{session_id}")
async def clear_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
