import math

import pytest

from sellerpolicy.hybrid import reciprocal_rank_fusion


def test_docstring_example_k60():
    fused = reciprocal_rank_fusion({"bm25": ["A", "B", "C"], "vector": ["B", "D", "A"]}, k=60)
    assert [r.chunk_id for r in fused] == ["B", "A", "D", "C"]
    scores = {r.chunk_id: r.score for r in fused}
    assert math.isclose(scores["A"], 1 / 61 + 1 / 63)
    assert math.isclose(scores["B"], 1 / 62 + 1 / 61)


def test_small_k_favours_single_first_place():
    fused = reciprocal_rank_fusion({"bm25": ["A", "B", "C"], "vector": ["B", "D", "A"]}, k=1)
    # A: 1/2 + 1/4 = 0.75 ; B: 1/3 + 1/2 = 0.833 -> B still wins here, D (1/3) beats C (1/4)
    assert [r.chunk_id for r in fused][:2] == ["B", "A"]
    fused = reciprocal_rank_fusion({"bm25": ["A", "B"], "vector": ["C", "B"]}, k=0)
    # k=0: A = 1, C = 1, B = 1/2 + 1/2 = 1 -> tie broken by best single rank, then id
    assert [r.chunk_id for r in fused] == ["A", "C", "B"]


def test_ranks_are_kept_per_method():
    fused = reciprocal_rank_fusion({"bm25": ["A"], "vector": ["B", "A"]})
    by_id = {r.chunk_id: r for r in fused}
    assert by_id["A"].ranks == {"bm25": 1, "vector": 2}
    assert by_id["B"].ranks == {"vector": 1}


def test_chunk_in_one_list_only():
    fused = reciprocal_rank_fusion({"bm25": ["A"], "vector": []})
    assert len(fused) == 1 and math.isclose(fused[0].score, 1 / 61)


def test_duplicates_in_a_list_are_ignored():
    fused = reciprocal_rank_fusion({"bm25": ["A", "A", "B"]})
    by_id = {r.chunk_id: r for r in fused}
    assert by_id["B"].ranks["bm25"] == 2


def test_empty_input():
    assert reciprocal_rank_fusion({}) == []


def test_negative_k_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"bm25": ["A"]}, k=-1)
