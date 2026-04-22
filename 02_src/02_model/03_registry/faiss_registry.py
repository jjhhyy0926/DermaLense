"""
02_src/02_model/03_registry/faiss_registry.py
LangChain Document 리스트로 FAISS 인덱스를 구축하고
지정 경로에 저장·로딩하는 함수 모음입니다.
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.common.logger import get_logger

logger = get_logger(__name__)


def build_faiss(docs: list[Document],
                embedding_model) -> FAISS:
    """Document 리스트로 FAISS 인덱스를 구축합니다."""
    logger.info(f"[FAISS] 인덱스 구축 시작 🚀 ({len(docs)}개 벡터)")
    vs = FAISS.from_documents(documents=docs, embedding=embedding_model)
    logger.info(f"[FAISS] 구축 완료 | 저장된 벡터 수: {vs.index.ntotal}")
    return vs


def save_faiss(vs: FAISS, save_path: str) -> None:
    """FAISS 인덱스를 로컬(또는 Google Drive)에 저장합니다."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vs.save_local(save_path)
    logger.info(f"[FAISS] 저장 완료: {save_path}")


def load_faiss(load_path: str, embedding_model) -> FAISS:
    """저장된 FAISS 인덱스를 불러옵니다."""
    vs = FAISS.load_local(
        load_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    logger.info(f"[FAISS] 로드 완료: {load_path} | 벡터 수: {vs.index.ntotal}")
    return vs
