"""Read policy files (.md / .txt / .pdf) into clean `Document` objects.

Seller Support analogy: before you can quote a policy, you open the page and ignore
the clutter around it. `clean_text` is that "ignore the clutter" step.

PDFs have no headings we can trust, so each page becomes a `## Page N` section.
That way citations read "Policy title § Page 4".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = (".md", ".txt", ".pdf")


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    text: str


def clean_text(text: str) -> str:
    """Normalise whitespace without touching meaning."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("﻿", "")
    text = re.sub(r"[ \t]+\n", "\n", text)      # trailing spaces
    text = re.sub(r"[ \t]{2,}", " ", text)      # repeated spaces
    text = re.sub(r"\n{3,}", "\n\n", text)      # blank-line runs
    return text.strip()


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        match = re.match(r"^#\s+(.+)$", line.strip())
        if match:
            return match.group(1).strip()
    return fallback


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = [f"# {path.stem.replace('_', ' ').title()}"]
    for number, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")
        if page_text:
            parts.append(f"## Page {number}\n\n{page_text}")
    return "\n\n".join(parts)


def load_file(path: str | Path) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.name}")
    if suffix == ".pdf":
        raw = _read_pdf(path)
    else:
        raw = path.read_text(encoding="utf-8")
    text = clean_text(raw)
    fallback = path.stem.replace("_", " ").title()
    return Document(doc_id=path.stem, title=_title_from_text(text, fallback), source=str(path), text=text)


def load_directory(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory not found: {directory}")
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    return [doc for doc in (load_file(p) for p in files) if doc.text]
