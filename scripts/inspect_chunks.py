"""Look at your chunks: python -m scripts.inspect_chunks [--max-chars 400] [--all]"""

from __future__ import annotations

import argparse
from collections import Counter

from sellerpolicy.chunker import chunk_documents
from sellerpolicy.config import settings
from sellerpolicy.loader import load_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Show how the policy documents are chunked.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--max-chars", type=int, default=settings.chunk_max_chars)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap_chars)
    parser.add_argument("--all", action="store_true", help="print every chunk (default: first 5)")
    args = parser.parse_args()

    docs = load_directory(args.data_dir or settings.data_dir)
    chunks = chunk_documents(docs, args.max_chars, args.overlap)
    sizes = [len(c.text) for c in chunks]
    print(f"{len(docs)} documents -> {len(chunks)} chunks (max_chars={args.max_chars}, overlap={args.overlap})")
    print(f"chunk size: min {min(sizes)}, avg {sum(sizes) // len(sizes)}, max {max(sizes)} chars\n")
    for doc_id, count in Counter(c.doc_id for c in chunks).items():
        print(f"  {doc_id}: {count}")
    print()
    for chunk in chunks if args.all else chunks[:5]:
        print("=" * 80)
        print(f"{chunk.chunk_id}  |  {chunk.citation_label}")
        print(f"breadcrumb: {chunk.breadcrumb or '(introduction)'}")
        print("-" * 80)
        print(chunk.text)
    if not args.all:
        print("\n(use --all to see every chunk)")


if __name__ == "__main__":
    main()
