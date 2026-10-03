"""Retrieval evaluation: Hit@1, Hit@k and MRR@k for BM25 vs vector vs hybrid.

    python -m scripts.evaluate [--k 5] [--rerank] [--out evals/results.md]

A hit = a chunk from the expected document whose breadcrumb contains the expected
section keyword appears in the top k.

Hit@k says "was it on the first page at all?"; MRR (mean of 1/rank) also rewards
putting it near the top. Rank 1 and rank 5 are both Hit@5 = 1, but MRR 1.0 vs 0.2.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sellerpolicy.config import PROJECT_ROOT
from sellerpolicy.pipeline import SellerPolicyRAG
from sellerpolicy.results import RetrievedChunk

MODES = ("bm25", "vector", "hybrid")


def is_relevant(item: RetrievedChunk, question: dict) -> bool:
    return item.chunk.doc_id == question["expected_doc_id"] and (
        question["section_keyword"].lower() in item.chunk.breadcrumb.lower()
    )


def first_relevant_rank(results: list[RetrievedChunk], question: dict) -> int | None:
    for rank, item in enumerate(results, start=1):
        if is_relevant(item, question):
            return rank
    return None


def summarise(ranks: list[int | None], k: int) -> dict:
    n = len(ranks) or 1
    return {
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n,
        f"mrr@{k}": sum(1.0 / r for r in ranks if r is not None and r <= k) / n,
    }


def load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(rag: SellerPolicyRAG, questions: list[dict], k: int, rerank: bool) -> dict[str, list[int | None]]:
    return {
        mode: [first_relevant_rank(rag.retrieve(q["question"], mode=mode, top_k=k, rerank=rerank), q) for q in questions]
        for mode in MODES
    }


def render_markdown(questions: list[dict], ranks: dict[str, list[int | None]], k: int, embedder: str, rerank: bool) -> str:
    fmt = lambda r: "-" if r is None else str(r)  # noqa: E731
    lines = [
        "# Retrieval evaluation",
        "",
        f"- Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        f"- Embedder: `{embedder}`",
        f"- Reranker: {'on' if rerank else 'off'}",
        f"- Questions: {len(questions)} (`evals/questions.json`)",
        "",
        f"| Mode | Hit@1 | Hit@{k} | MRR@{k} |",
        "|---|---|---|---|",
    ]
    for mode in MODES:
        s = summarise(ranks[mode], k)
        lines.append(f"| {mode} | {s['hit@1']:.2f} | {s[f'hit@{k}']:.2f} | {s[f'mrr@{k}']:.3f} |")
    lines += ["", f"## Hit@{k} by question style", "", "| Style | Questions | bm25 | vector | hybrid |", "|---|---|---|---|---|"]
    for style in sorted({q["style"] for q in questions}):
        idx = [i for i, q in enumerate(questions) if q["style"] == style]
        cells = [f"{summarise([ranks[m][i] for i in idx], k)[f'hit@{k}']:.2f}" for m in MODES]
        lines.append(f"| {style} | {len(idx)} | " + " | ".join(cells) + " |")
    lines += ["", f"## Per question (rank of first correct chunk, - = not in top {k})", "",
              "| id | style | bm25 | vector | hybrid | question |", "|---|---|---|---|---|---|"]
    for i, q in enumerate(questions):
        lines.append(
            f"| {q['id']} | {q['style']} | {fmt(ranks['bm25'][i])} | {fmt(ranks['vector'][i])} | {fmt(ranks['hybrid'][i])} | {q['question']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--questions", default=str(PROJECT_ROOT / "evals" / "questions.json"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "evals" / "results.md"))
    args = parser.parse_args()

    rag = SellerPolicyRAG()
    questions = load_questions(Path(args.questions))
    ranks = run(rag, questions, args.k, args.rerank)
    report = render_markdown(questions, ranks, args.k, rag.embedder.name, args.rerank)
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
