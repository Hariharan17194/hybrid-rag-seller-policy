"""Pluggable embedders: sentence-transformers (default), OpenAI, or offline hashing.

Seller Support analogy: an embedding is a "meaning fingerprint". Two emails that mean
the same thing ("parcel never showed up" / "item did not arrive") get fingerprints
that point the same way, so cosine similarity finds them even with no shared words.

`HashingEmbedder` is NOT semantic. It hashes words into a fixed-size vector, so it
runs offline with no downloads. Use it for tests and demos only.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

from sellerpolicy.bm25_index import tokenize


class Embedder(ABC):
    name: str = "base"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


class HashingEmbedder(Embedder):
    """Bag-of-words feature hashing (unigrams + bigrams), L2-normalised."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            index, sign = self._bucket(feature)
            vector[index] += sign
        return _normalise(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class _TransformersMeanPooler:
    """Same vectors as SentenceTransformer for MiniLM-style models (mean pooling + L2 norm),
    using plain `transformers`. Used when sentence-transformers cannot be imported
    (for example when Windows Application Control blocks one of its scikit-learn DLLs)."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).eval()

    def encode(self, texts: list[str], batch_size: int = 32, **_) -> list:
        import torch
        import torch.nn.functional as F

        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = self._tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                    max_length=256, return_tensors="pt")
            with torch.no_grad():
                hidden = self._model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9), dim=1)
            vectors.extend(pooled.numpy())
        return vectors


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.name = f"sentence-transformers:{model_name}"
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except ImportError:
            self._model = _TransformersMeanPooler(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        return [v.tolist() for v in vectors]


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small", client=None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self._client = client
        self.model_name = model_name
        self.name = f"openai:{model_name}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 100):
            response = self._client.embeddings.create(model=self.model_name, input=texts[start:start + 100])
            vectors.extend(item.embedding for item in response.data)
        return vectors


def get_embedder(settings) -> Embedder:
    provider = settings.embedding_provider
    if provider == "hashing":
        return HashingEmbedder(settings.hashing_dim)
    if provider == "openai":
        if not settings.has_openai_key:
            raise ValueError("EMBEDDING_PROVIDER=openai needs OPENAI_API_KEY in .env")
        return OpenAIEmbedder(settings.openai_api_key.get_secret_value(), settings.openai_embedding_model)
    return SentenceTransformerEmbedder(settings.st_model_name)
