"""BM25 vs vector vs hybrid, side by side: python -m scripts.compare_search ["question"] [--top-k 5]"""

from __future__ import annotations

import argparse
import sys

from sellerpolicy.pipeline import SellerPolicyRAG

DEFAULT_QUESTIONS = [
    "buyer says the parcel never arrived",
    "What is the SAFE-T claim deadline?",
    "can I charge a fee if the customer opened the box?",
]


def show(rag: SellerPolicyRAG, question: str, top_k: int) -> None:
    print("=" * 100)
    print(f"Q: {question}")
    for mode in ("bm25", "vector", "hybrid"):
        print(f"\n  {mode.upper()}")
        results = rag.retrieve(question, mode=mode, top_k=top_k, rerank=False)
        if not results:
            print("    (no results)")
        for i, r in enumerate(results, start=1):
            extra = f"bm25#{r.bm25_rank or '-'} vec#{r.vector_rank or '-'}" if mode == "hybrid" else ""
            print(f"    {i}. {r.score:8.4f}  {r.citation_label:<70} {extra}")
    print()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Compare BM25, vector and hybrid retrieval.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    rag = SellerPolicyRAG()
    for question in [args.question] if args.question else DEFAULT_QUESTIONS:
        show(rag, question, args.top_k)


if __name__ == "__main__":
    main()
