"""Build BM25 + vector indexes: python -m scripts.build_index [--data-dir data/banking_sample]"""

from __future__ import annotations

import argparse
import time

from sellerpolicy.pipeline import SellerPolicyRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the BM25 and vector indexes.")
    parser.add_argument("--data-dir", default=None, help="folder of .md/.txt/.pdf files (default: DATA_DIR)")
    args = parser.parse_args()

    rag = SellerPolicyRAG()
    print(f"Embedding provider: {rag.settings.embedding_provider}")
    start = time.perf_counter()
    stats = rag.build_index(args.data_dir)
    elapsed = time.perf_counter() - start
    print(f"Indexed {stats['documents']} documents -> {stats['chunks']} chunks in {elapsed:.1f}s")
    print(f"  data:     {stats['data_dir']}")
    print(f"  storage:  {stats['storage_dir']}")
    print(f"  embedder: {stats['embedding_model']}")


if __name__ == "__main__":
    main()
