import math

import pytest

from sellerpolicy.chunker import Chunk
from sellerpolicy.config import Settings
from sellerpolicy.embeddings import HashingEmbedder, OpenAIEmbedder, get_embedder
from sellerpolicy.vector_store import EmbedderMismatchError, VectorStore


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder(128)
    v1, v2 = e.embed_query("lost inventory"), e.embed_query("lost inventory")
    assert v1 == v2
    assert len(v1) == 128
    assert math.isclose(math.sqrt(dot(v1, v1)), 1.0, rel_tol=1e-9)


def test_hashing_similar_texts_are_closer():
    e = HashingEmbedder(1024)
    a, b, c = e.embed_documents(["lost inventory reimbursement", "inventory lost, reimbursed", "ODR target"])
    assert dot(a, b) > dot(a, c)


def test_hashing_empty_text_is_zero_vector():
    assert set(HashingEmbedder(16).embed_query("")) == {0.0}


def test_get_embedder_hashing():
    s = Settings(_env_file=None, embedding_provider="hashing", hashing_dim=64)
    assert get_embedder(s).name == "hashing:64"


def test_get_embedder_openai_requires_key():
    s = Settings(_env_file=None, embedding_provider="openai", openai_api_key=None)
    with pytest.raises(ValueError):
        get_embedder(s)


def test_invalid_provider_rejected():
    with pytest.raises(ValueError):
        Settings(_env_file=None, embedding_provider="magic")


def test_openai_embedder_with_fake_client():
    class Item:
        def __init__(self, i):
            self.embedding = [float(i)]

    class FakeEmbeddings:
        def create(self, model, input):
            return type("R", (), {"data": [Item(i) for i, _ in enumerate(input)]})()

    fake = type("C", (), {"embeddings": FakeEmbeddings()})()
    emb = OpenAIEmbedder(api_key="x", client=fake)
    assert emb.embed_documents(["a", "b"]) == [[0.0], [1.0]]
    assert emb.name == "openai:text-embedding-3-small"


def _chunks():
    return [
        Chunk("d::000", "d", "T", "s", ["Refunds"], "refund within 2 business days", 0),
        Chunk("d::001", "d", "T", "s", ["ODR"], "order defect rate under 1%", 1),
    ]


def test_vector_store_build_and_search(tmp_path):
    e = HashingEmbedder(256)
    store = VectorStore(tmp_path / "chroma", "test")
    chunks = _chunks()
    store.build(chunks, e.embed_documents([c.text for c in chunks]), e.name)
    assert store.count() == 2
    assert store.built_with == "hashing:256"
    hits = store.search(e.embed_query("order defect rate"), k=2)
    assert hits[0][0] == "d::001"
    assert -1.0 <= hits[0][1] <= 1.0001


def test_vector_store_rebuild_replaces(tmp_path):
    e = HashingEmbedder(64)
    store = VectorStore(tmp_path / "chroma", "test")
    chunks = _chunks()
    store.build(chunks, e.embed_documents([c.text for c in chunks]), e.name)
    store.build(chunks[:1], e.embed_documents([chunks[0].text]), e.name)
    assert store.count() == 1


def test_vector_store_embedder_mismatch(tmp_path):
    e = HashingEmbedder(64)
    store = VectorStore(tmp_path / "chroma", "test")
    chunks = _chunks()
    store.build(chunks, e.embed_documents([c.text for c in chunks]), e.name)
    store.check_embedder("hashing:64")
    with pytest.raises(EmbedderMismatchError):
        store.check_embedder("sentence-transformers:other")


def test_vector_store_missing_collection(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore(tmp_path / "chroma", "absent").count()
