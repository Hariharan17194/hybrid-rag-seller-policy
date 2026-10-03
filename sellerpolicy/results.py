"""Data classes that flow out of retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from sellerpolicy.chunker import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float                     # fused RRF score, or the single method's score
    bm25_rank: int | None = None
    vector_rank: int | None = None
    bm25_score: float | None = None
    vector_score: float | None = None
    rerank_score: float | None = None

    @property
    def citation_label(self) -> str:
        return self.chunk.citation_label

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "section": self.chunk.breadcrumb,
            "citation": self.citation_label,
            "text": self.chunk.text,
            "score": self.score,
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
            "rerank_score": self.rerank_score,
        }


@dataclass
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)   # 1-based source numbers used in the text
    mode: str = "extractive"                         # "llm" | "extractive" | "not_found"
    retrieval_mode: str = "hybrid"
    note: str | None = None

    @property
    def found(self) -> bool:
        return self.mode != "not_found"
