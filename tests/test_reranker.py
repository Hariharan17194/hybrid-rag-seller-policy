from sellerpolicy.chunker import Chunk
from sellerpolicy.reranker import Reranker
from sellerpolicy.results import RetrievedChunk


def candidates():
    return [
        RetrievedChunk(Chunk(f"d::{i:03d}", "d", "T", "s", ["S"], text, i), score=1.0 - i / 10)
        for i, text in enumerate(["about payments", "about appeals within 17 days", "about refunds"])
    ]


class FakeCrossEncoder:
    def predict(self, pairs):
        return [10.0 if "appeal" in text else 0.0 for _, text in pairs]


class BrokenCrossEncoder:
    def predict(self, pairs):
        raise RuntimeError("boom")


def test_rerank_reorders_by_model_score():
    result = Reranker(model=FakeCrossEncoder()).rerank("appeal deadline", candidates(), top_k=2)
    assert result[0].chunk.chunk_id == "d::001"
    assert result[0].rerank_score == 10.0
    assert len(result) == 2


def test_rerank_falls_back_when_model_cannot_load():
    reranker = Reranker(model_name="definitely/not-a-real-model-xyz")
    reranker._failed = True  # simulate a failed load without touching the network
    result = reranker.rerank("q", candidates(), top_k=2)
    assert [r.chunk.chunk_id for r in result] == ["d::000", "d::001"]
    assert not reranker.available


def test_rerank_falls_back_when_predict_fails():
    result = Reranker(model=BrokenCrossEncoder()).rerank("q", candidates(), top_k=3)
    assert [r.chunk.chunk_id for r in result] == ["d::000", "d::001", "d::002"]


def test_rerank_empty():
    assert Reranker(model=FakeCrossEncoder()).rerank("q", [], top_k=3) == []
