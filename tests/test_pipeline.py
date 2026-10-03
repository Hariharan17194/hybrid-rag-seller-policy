import json

import pytest

from sellerpolicy.config import Settings
from sellerpolicy.embeddings import HashingEmbedder
from sellerpolicy.pipeline import IndexNotBuiltError, SellerPolicyRAG
from sellerpolicy.vector_store import EmbedderMismatchError


def test_build_index_writes_storage(rag):
    assert rag.is_ready()
    data = json.loads(rag.chunks_path.read_text(encoding="utf-8"))
    assert data["embedding_model"] == "hashing:256"
    assert len(data["chunks"]) == len(rag.chunks) > 0


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_retrieve_finds_safe_t(rag, mode):
    results = rag.retrieve("SAFE-T claim deadline", mode=mode)
    assert results[0].chunk.doc_id == "returns"


def test_hybrid_keeps_both_ranks(rag):
    top = rag.retrieve("A-to-z claim response 72 hours", mode="hybrid")[0]
    assert top.bm25_rank is not None and top.vector_rank is not None


def test_retrieve_respects_top_k(rag):
    assert len(rag.retrieve("claim", mode="hybrid", top_k=2)) <= 2


def test_retrieve_validates_input(rag):
    with pytest.raises(ValueError):
        rag.retrieve("  ")
    with pytest.raises(ValueError):
        rag.retrieve("x", mode="magic")


def test_answer_extractive_with_citation(rag):
    answer = rag.answer("How many days to appeal a deactivation?")
    assert answer.mode == "extractive"
    assert "17 days" in answer.text
    assert answer.retrieval_mode == "hybrid"


def test_answer_bm25_off_topic_not_found(rag):
    answer = rag.answer("What is the capital of France?", mode="bm25")
    assert answer.mode == "not_found"


def test_reload_from_disk(rag, test_settings):
    fresh = SellerPolicyRAG(settings=test_settings, embedder=HashingEmbedder(test_settings.hashing_dim))
    assert fresh.retrieve("SAFE-T", mode="bm25")[0].chunk.doc_id == "returns"


def test_bm25_mode_does_not_load_embedder(rag, test_settings):
    fresh = SellerPolicyRAG(settings=test_settings)
    fresh.retrieve("SAFE-T", mode="bm25")
    assert fresh._embedder is None


def test_mismatched_embedder_refuses(rag, test_settings):
    other = SellerPolicyRAG(settings=test_settings, embedder=HashingEmbedder(128))
    with pytest.raises(EmbedderMismatchError):
        other.retrieve("SAFE-T", mode="vector")


def test_not_built_raises(tmp_path):
    s = Settings(_env_file=None, embedding_provider="hashing", storage_dir=tmp_path / "empty")
    rag = SellerPolicyRAG(settings=s)
    assert not rag.is_ready()
    with pytest.raises(IndexNotBuiltError):
        rag.retrieve("x", mode="bm25")


def test_build_from_empty_dir_fails(tmp_path, test_settings):
    empty = tmp_path / "nothing"
    empty.mkdir()
    rag = SellerPolicyRAG(settings=test_settings, embedder=HashingEmbedder(64))
    with pytest.raises(ValueError):
        rag.build_index(empty)
