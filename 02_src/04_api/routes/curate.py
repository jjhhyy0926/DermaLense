from fastapi import APIRouter
from schemas import CurateRequest, CurateResponse
from rag.curator import curate

router = APIRouter()


@router.post("/curate", response_model=CurateResponse)
async def curate_endpoint(req: CurateRequest):
    result = curate(message=req.message, session=req.session)
    return CurateResponse(**result)
