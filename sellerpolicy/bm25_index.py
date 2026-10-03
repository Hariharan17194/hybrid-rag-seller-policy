"""BM25 keyword search with a domain-aware tokenizer.

Seller Support analogy: Ctrl+F on steroids. A seller who writes "SAFE-T" or
"A-to-z" wants exactly that policy, and BM25 rewards rare, exact words (high IDF)
far more than common ones like "seller".

The tokenizer keeps policy codes in one piece (`a-to-z`, `safe-t`, `2.5`, `60-day`)
and normalises the ways sellers write A-to-z (`A to Z`, `A2Z`, `AtoZ`) to `a-to-z`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its me my of on or our so than that the their them then there these they this to
    was we were what when where which who why will with would you your"""
    .split()
)

_A_TO_Z_RE = re.compile(r"\ba(?:\s*-\s*|\s+)?(?:to|2)(?:\s*-\s*|\s+)?z\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def _light_stem(token: str) -> str:
    """Drop a plural 's' so 'claims' matches 'claim'. Applied identically to docs and queries."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss") and token.isalpha():
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    text = _A_TO_Z_RE.sub("a-to-z", text.lower())
    tokens = []
    for token in _TOKEN_RE.findall(text):
        if token in STOPWORDS:
            continue
        tokens.append(_light_stem(token))
    return tokens


class BM25Index:
    """Thin wrapper over `bm25s` that remembers which chunk id each row is."""

    def __init__(self) -> None:
        self._retriever = None
        self.ids: list[str] = []

    @property
    def is_built(self) -> bool:
        return self._retriever is not None

    def build(self, ids: list[str], texts: list[str]) -> None:
        import bm25s

        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        if not ids:
            raise ValueError("Cannot build a BM25 index with no documents")
        self.ids = list(ids)
        corpus_tokens = [tokenize(t) or ["__empty__"] for t in texts]
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        self._retriever = retriever

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return up to k (chunk_id, score) pairs. Chunks with score 0 share no words with the query."""
        if not self.is_built:
            raise RuntimeError("BM25 index is not built or loaded")
        vocab = self._retriever.vocab_dict
        query_tokens = [t for t in tokenize(query) if t in vocab]
        if not query_tokens:
            return []
        scores = np.asarray(self._retriever.get_scores(query_tokens), dtype=float)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, directory: str | Path) -> None:
        if not self.is_built:
            raise RuntimeError("Nothing to save: BM25 index is not built")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(directory))
        (directory / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "BM25Index":
        import bm25s

        directory = Path(directory)
        if not (directory / "ids.json").exists():
            raise FileNotFoundError(f"No BM25 index in {directory}. Run: python -m scripts.build_index")
        index = cls()
        index._retriever = bm25s.BM25.load(str(directory))
        index.ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        return index
