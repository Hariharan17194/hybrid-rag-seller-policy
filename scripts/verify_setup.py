"""Environment checklist: python -m scripts.verify_setup"""

from __future__ import annotations

import importlib
import sys

from sellerpolicy.config import settings

REQUIRED = ["pydantic_settings", "dotenv", "pypdf", "bm25s", "chromadb", "numpy", "openai"]
OPTIONAL = ["sentence_transformers", "transformers", "torch", "streamlit", "fastapi", "uvicorn", "pytest"]


def check(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def main() -> int:
    ok = True
    print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info >= (3, 10) else "(3.10+ recommended)")
    print("\nRequired packages:")
    for module in REQUIRED:
        found = check(module)
        ok &= found
        print(f"  [{'x' if found else ' '}] {module}")
    print("\nOptional packages:")
    for module in OPTIONAL:
        print(f"  [{'x' if check(module) else ' '}] {module}")

    docs = sorted(p.name for p in settings.data_dir.glob("*") if p.suffix.lower() in (".md", ".txt", ".pdf"))
    print(f"\nData dir: {settings.data_dir} ({len(docs)} documents)")
    ok &= bool(docs)
    print(f"Storage dir: {settings.storage_dir}")
    print(f"Embedding provider: {settings.embedding_provider}")
    if settings.embedding_provider == "sentence-transformers" and not check("sentence_transformers"):
        if check("transformers") and check("torch"):
            print("  (sentence-transformers unavailable; using the plain-transformers fallback, same vectors)")
        else:
            print("  ! needs sentence-transformers (or transformers + torch): set EMBEDDING_PROVIDER=hashing or pip install it")
            ok = False
    print(f"OpenAI key: {'set (LLM answers)' if settings.has_openai_key else 'not set (extractive mode)'}")
    print(f"Reranker: {'on' if settings.use_reranker else 'off'}")
    print("\nAll good!" if ok else "\nSome checks failed (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
