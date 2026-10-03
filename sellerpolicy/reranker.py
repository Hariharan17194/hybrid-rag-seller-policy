"""Optional cross-encoder reranker.

Seller Support analogy: retrieval is skimming headlines; reranking is actually
reading the shortlisted paragraphs next to the seller's email.

A bi-encoder (our embedder) encodes question and chunk separately, which is fast and
works on the whole library. A cross-encoder reads question + chunk *together*, which
is more precise but slow, so we only run it on the top ~20 fused candidates.

The model is loaded lazily on first use. If it cannot load (no internet, no
sentence-transformers), we keep the fused order instead of crashing.
"""

from __future__ import annotations

import logging

from sellerpolicy.results import RetrievedChunk

logger = logging.getLogger(__name__)


class _TransformersCrossEncoder:
    """Minimal CrossEncoder.predict() with plain `transformers`, for when
    sentence-transformers cannot be imported."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        batch = self._tokenizer([q for q, _ in pairs], [t for _, t in pairs], padding=True,
                                truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**batch).logits
        return logits.view(-1).tolist() if logits.shape[-1] == 1 else logits[:, -1].tolist()


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", model=None) -> None:
        self.model_name = model_name
        self._model = model
        self._failed = False

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._model is not None or self._failed:
            return self._model
        try:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except ImportError:
                self._model = _TransformersCrossEncoder(self.model_name)
        except Exception as exc:
            logger.warning("Reranker unavailable (%s); keeping fused order.", exc)
            self._failed = True
        return self._model

    def rerank(self, question: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        model = self._load()
        if model is None:
            return candidates[:top_k]
        try:
            scores = model.predict([(question, c.chunk.text_for_search) for c in candidates])
        except Exception as exc:
            logger.warning("Reranking failed (%s); keeping fused order.", exc)
            return candidates[:top_k]
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        ordered = sorted(candidates, key=lambda c: -c.rerank_score)
        return ordered[:top_k]
