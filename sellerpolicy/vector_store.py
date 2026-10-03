"""ChromaDB persistent vector index.

Seller Support analogy: a filing cabinet sorted by meaning instead of by title.

We compute embeddings ourselves (not Chroma's built-in function) so the same
embedder is used at build time and at question time. The collection records which
model built it; if you switch EMBEDDING_PROVIDER without rebuilding, the vectors
would be meaningless, so `check_embedder` refuses to search.

Chroma returns cosine *distance*; we report similarity = 1 - distance.
"""

from __future__ import annotations

from pathlib import Path

from sellerpolicy.chunker import Chunk


class EmbedderMismatchError(RuntimeError):
    pass


class VectorStore:
    def __init__(self, path: str | Path, collection_name: str = "seller_policies") -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(self.path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            try:
                self._collection = self._client.get_collection(self.collection_name)
            except Exception as exc:  # chroma raises different types across versions
                raise FileNotFoundError(
                    f"No vector collection '{self.collection_name}' in {self.path}. "
                    "Run: python -m scripts.build_index"
                ) from exc
        return self._collection

    @property
    def built_with(self) -> str | None:
        return (self.collection.metadata or {}).get("embedding_model")

    def count(self) -> int:
        return self.collection.count()

    def build(self, chunks: list[Chunk], embeddings: list[list[float]], embedder_name: str) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedder_name},
        )
        for start in range(0, len(chunks), 500):
            batch = chunks[start:start + 500]
            self._collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[start:start + 500],
                documents=[c.text for c in batch],
                metadatas=[{"doc_id": c.doc_id, "title": c.title, "section": c.breadcrumb} for c in batch],
            )

    def check_embedder(self, embedder_name: str) -> None:
        built = self.built_with
        if built and built != embedder_name:
            raise EmbedderMismatchError(
                f"The vector index was built with '{built}' but the current embedder is "
                f"'{embedder_name}'. Rebuild it: python -m scripts.build_index"
            )

    def search(self, query_embedding: list[float], k: int = 10, where: dict | None = None) -> list[tuple[str, float]]:
        total = self.count()
        if total == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding], n_results=min(k, total), where=where, include=["distances"]
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [(chunk_id, 1.0 - float(dist)) for chunk_id, dist in zip(ids, distances)]
