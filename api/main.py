"""FastAPI service: POST /ask, GET /health.

    uvicorn api.main:app --reload --port 8000   ->  http://localhost:8000/docs
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from sellerpolicy import __version__
from sellerpolicy.pipeline import IndexNotBuiltError, SellerPolicyRAG
from sellerpolicy.vector_store import EmbedderMismatchError

app = FastAPI(title="SellerPolicy Assistant API", version=__version__,
              description="Hybrid-search RAG over Seller Support policies, with citations.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["What is the ODR target?"])
    mode: Literal["hybrid", "bm25", "vector"] = "hybrid"
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool | None = None
    use_llm: bool | None = None


class SourceOut(BaseModel):
    number: int
    chunk_id: str
    doc_id: str
    citation: str
    section: str
    text: str
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rerank_score: float | None = None
    cited: bool


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: str
    retrieval_mode: str
    found: bool
    note: str | None = None
    sources: list[SourceOut]


@lru_cache(maxsize=1)
def get_rag() -> SellerPolicyRAG:
    return SellerPolicyRAG()


@app.get("/health")
def health(rag: SellerPolicyRAG = Depends(get_rag)) -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "index_ready": rag.is_ready(),
        "embedding_provider": rag.settings.embedding_provider,
        "llm_enabled": rag.generator.llm_enabled,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, rag: SellerPolicyRAG = Depends(get_rag)) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be blank")
    if not rag.is_ready():
        raise HTTPException(status_code=503, detail="Index not built. Run: python -m scripts.build_index")
    try:
        answer = rag.answer(request.question, mode=request.mode, top_k=request.top_k,
                            rerank=request.rerank, use_llm=request.use_llm)
    except (IndexNotBuiltError, EmbedderMismatchError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    sources = [
        SourceOut(number=i, cited=i in answer.cited, **{k: v for k, v in s.to_dict().items() if k in SourceOut.model_fields})
        for i, s in enumerate(answer.sources, start=1)
    ]
    return AskResponse(question=answer.question, answer=answer.text, mode=answer.mode,
                       retrieval_mode=answer.retrieval_mode, found=answer.found, note=answer.note, sources=sources)
