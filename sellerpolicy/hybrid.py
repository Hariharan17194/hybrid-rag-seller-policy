"""Reciprocal Rank Fusion (RRF), hand-written.

Seller Support analogy: two senior associates each give you their top-5 policy
pages. One searched by keyword, one by meaning. A page that BOTH put near the top
is the safest bet, even if neither put it first.

Why ranks, not scores? BM25 scores look like 7.3 and cosine similarities like 0.62.
They are on different scales, so adding them is meaningless. Ranks are comparable.

    RRF(chunk) = sum over methods of 1 / (k + rank)      (rank starts at 1, k = 60)

Worked example (k = 60):

    BM25 ranking:   A, B, C
    Vector ranking: B, D, A

    A: 1/(60+1) + 1/(60+3) = 0.016393 + 0.015873 = 0.032266
    B: 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522   <- winner
    C: 1/(60+3)            = 0.015873
    D: 1/(60+2)            = 0.016129

    Fused order: B, A, D, C

B wins: ranked well by both methods. With a small k (say k = 1) a single first place
counts for much more, and A (1st in BM25) would beat B.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusedResult:
    chunk_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # method -> 1-based rank


def reciprocal_rank_fusion(ranked_lists: dict[str, list[str]], k: int = 60) -> list[FusedResult]:
    """Fuse several ranked lists of chunk ids. Keeps each method's rank for transparency."""
    if k < 0:
        raise ValueError("k must be >= 0")
    results: dict[str, FusedResult] = {}
    for method, ids in ranked_lists.items():
        seen: set[str] = set()
        rank = 0
        for chunk_id in ids:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            entry = results.setdefault(chunk_id, FusedResult(chunk_id=chunk_id, score=0.0))
            entry.score += 1.0 / (k + rank)
            entry.ranks[method] = rank
    # Tie-break: best single rank, then id, so the order is deterministic.
    return sorted(results.values(), key=lambda r: (-r.score, min(r.ranks.values()), r.chunk_id))
