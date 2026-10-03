"""SellerPolicyRAG: build_index / retrieve / answer.

Seller Support analogy: the whole shift in one object. Build the library once,
then for each email: search two ways, fuse, optionally re-read the shortlist, and
write the cited reply.

The embedder, vector store and reranker are created lazily, so a BM25-only
question never loads a 90 MB model.

chunks.json stores the full chunk objects (title, breadcrumb, text). BM25 and Chroma
only return ids, and this file turns ids back into citable chunks.
"""

from __future__ import annotations

import json
from pathlib import Path

from sellerpolicy.bm25_index import BM25Index
from sellerpolicy.chunker import Chunk, chunk_documents
from sellerpolicy.config import Settings, settings as default_settings
from sellerpolicy.embeddings import Embedder, get_embedder
from sellerpolicy.generator import Generator
from sellerpolicy.hybrid import reciprocal_rank_fusion
from sellerpolicy.loader import load_directory
from sellerpolicy.reranker import Reranker
from sellerpolicy.results import Answer, RetrievedChunk
from sellerpolicy.vector_store import VectorStore

MODES = ("hybrid", "bm25", "vector")


class IndexNotBuiltError(RuntimeError):
    pass


class SellerPolicyRAG:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self._embedder = embedder
        self._reranker = reranker
        self._generator = generator
        self._chunks: dict[str, Chunk] | None = None
        self._bm25: BM25Index | None = None
        self._vector_store: VectorStore | None = None

    # ---------- paths ----------
    @property
    def storage_dir(self) -> Path:
        return Path(self.settings.storage_dir)

    @property
    def chunks_path(self) -> Path:
        return self.storage_dir / "chunks.json"

    @property
    def bm25_dir(self) -> Path:
        return self.storage_dir / "bm25"

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    def is_ready(self) -> bool:
        return self.chunks_path.exists() and (self.bm25_dir / "ids.json").exists()

    # ---------- lazy components ----------
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.settings)
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self.settings.reranker_model)
        return self._reranker

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(self.settings)
        return self._generator

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(self.chroma_dir, self.settings.collection_name)
        return self._vector_store

    @property
    def chunks(self) -> dict[str, Chunk]:
        if self._chunks is None:
            if not self.chunks_path.exists():
                raise IndexNotBuiltError("Index not built. Run: python -m scripts.build_index")
            data = json.loads(self.chunks_path.read_text(encoding="utf-8"))
            self._chunks = {d["chunk_id"]: Chunk.from_dict(d) for d in data["chunks"]}
        return self._chunks

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            if not (self.bm25_dir / "ids.json").exists():
                raise IndexNotBuiltError("Index not built. Run: python -m scripts.build_index")
            self._bm25 = BM25Index.load(self.bm25_dir)
        return self._bm25

    # ---------- build ----------
    def build_index(self, data_dir: str | Path | None = None) -> dict:
        data_dir = Path(data_dir) if data_dir else Path(self.settings.data_dir)
        docs = load_directory(data_dir)
        if not docs:
            raise ValueError(f"No .md/.txt/.pdf documents found in {data_dir}")
        chunks = chunk_documents(docs, self.settings.chunk_max_chars, self.settings.chunk_overlap_chars)

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_dir": str(data_dir),
            "embedding_model": self.embedder.name,
            "chunks": [c.to_dict() for c in chunks],
        }
        self.chunks_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        bm25 = BM25Index()
        bm25.build([c.chunk_id for c in chunks], [c.text_for_search for c in chunks])
        bm25.save(self.bm25_dir)

        embeddings = self.embedder.embed_documents([c.text_for_search for c in chunks])
        self.vector_store.build(chunks, embeddings, self.embedder.name)

        self._chunks = {c.chunk_id: c for c in chunks}
        self._bm25 = bm25
        return {
            "documents": len(docs),
            "chunks": len(chunks),
            "embedding_model": self.embedder.name,
            "data_dir": str(data_dir),
            "storage_dir": str(self.storage_dir),
        }

    # ---------- retrieve ----------
    def _bm25_search(self, question: str, k: int) -> list[tuple[str, float]]:
        return self.bm25.search(question, k=k)

    def _vector_search(self, question: str, k: int) -> list[tuple[str, float]]:
        self.vector_store.check_embedder(self.embedder.name)
        return self.vector_store.search(self.embedder.embed_query(question), k=k)

    def retrieve(
        self, question: str, mode: str = "hybrid", top_k: int | None = None, rerank: bool | None = None
    ) -> list[RetrievedChunk]:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        top_k = top_k or self.settings.top_k
        rerank = self.settings.use_reranker if rerank is None else rerank
        candidate_k = max(self.settings.candidate_k, top_k)
        chunks = self.chunks

        bm25_hits = self._bm25_search(question, candidate_k) if mode in ("hybrid", "bm25") else []
        vector_hits = self._vector_search(question, candidate_k) if mode in ("hybrid", "vector") else []
        bm25_rank = {cid: i for i, (cid, _) in enumerate(bm25_hits, start=1)}
        vector_rank = {cid: i for i, (cid, _) in enumerate(vector_hits, start=1)}
        bm25_score = dict(bm25_hits)
        vector_score = dict(vector_hits)

        if mode == "hybrid":
            fused = reciprocal_rank_fusion(
                {"bm25": [c for c, _ in bm25_hits], "vector": [c for c, _ in vector_hits]}, k=self.settings.rrf_k
            )
            ordered = [(r.chunk_id, r.score) for r in fused]
        else:
            ordered = bm25_hits if mode == "bm25" else vector_hits

        candidates = [
            RetrievedChunk(
                chunk=chunks[cid],
                score=score,
                bm25_rank=bm25_rank.get(cid),
                vector_rank=vector_rank.get(cid),
                bm25_score=bm25_score.get(cid),
                vector_score=vector_score.get(cid),
            )
            for cid, score in ordered
            if cid in chunks
        ]
        if rerank:
            return self.reranker.rerank(question, candidates[:candidate_k], top_k)
        return candidates[:top_k]

    # ---------- answer ----------
    def answer(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        rerank: bool | None = None,
        use_llm: bool | None = None,
    ) -> Answer:
        sources = self.retrieve(question, mode=mode, top_k=top_k, rerank=rerank)
        result = self.generator.generate(question, sources, use_llm=use_llm)
        result.retrieval_mode = mode
        return result
