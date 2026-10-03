"""Ask one question: python -m scripts.ask "question" [--mode hybrid|bm25|vector] [--top-k 5] [--rerank] [--no-llm]"""

from __future__ import annotations

import argparse
import sys

from sellerpolicy.pipeline import MODES, SellerPolicyRAG


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask the policy assistant a question.")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--no-llm", action="store_true", help="force extractive mode")
    args = parser.parse_args()

    rag = SellerPolicyRAG()
    answer = rag.answer(
        args.question, mode=args.mode, top_k=args.top_k, rerank=args.rerank or None, use_llm=False if args.no_llm else None
    )
    print(f"\nQ: {answer.question}")
    print(f"[retrieval: {answer.retrieval_mode} | answer mode: {answer.mode}]\n")
    print(answer.text)
    if answer.note:
        print(f"\n({answer.note})")
    if answer.sources:
        print("\nSources:")
        for number, src in enumerate(answer.sources, start=1):
            marker = "*" if number in answer.cited else " "
            ranks = f"bm25={src.bm25_rank or '-'} vector={src.vector_rank or '-'} score={src.score:.4f}"
            print(f" {marker}[{number}] {src.citation_label}  ({ranks})")


if __name__ == "__main__":
    main()
