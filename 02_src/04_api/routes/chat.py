import sys
import os
import traceback
import logging

# 01_rag_chain 패키지 경로 추가
_RAG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "02_model", "01_rag_chain"))
sys.path.insert(0, _RAG_DIR)

from fastapi import APIRouter, HTTPException
from schemas import ChatRequest, ChatResponse, SourceChunk
from graph import run_graph

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        history = [{"role": m.role, "content": m.content} for m in (req.history or [])]
        result = run_graph(
            query=req.question,
            history=history,
        )
        sources = [
            SourceChunk(
                product_name=s.get("ingredient", s.get("source", "?")),
                content=s.get("content", ""),
            )
            for s in result.get("sources", [])
        ]
        return ChatResponse(answer=result["answer"], sources=sources)
    except Exception as e:
        logger.error(f"[/api/chat] 오류:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
