"""Section-aware chunking with breadcrumbs.

Seller Support analogy: you never quote "17 days" without knowing it belongs to
"Deactivation and Appeals > Appeal timelines". A fixed 1000-character cut could
split that sentence from its heading. So we split on headings first, and only cut
long sections into smaller pieces (with a little overlap) when we must.

Every chunk carries its breadcrumb, and `text_for_search` puts the document title
and breadcrumb in front of the body, so a chunk makes sense on its own.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from sellerpolicy.loader import Document

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    section_path: list[str] = field(default_factory=list)
    text: str = ""
    ordinal: int = 0

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.section_path)

    @property
    def section(self) -> str:
        return self.section_path[-1] if self.section_path else "Introduction"

    @property
    def citation_label(self) -> str:
        return f"{self.title} § {self.section}"

    @property
    def text_for_search(self) -> str:
        header = self.title if not self.section_path else f"{self.title}\n{self.breadcrumb}"
        return f"{header}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def split_into_sections(text: str) -> list[tuple[list[str], str]]:
    """Return (section_path, body) pairs. The level-1 heading is the doc title, not a section."""
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append(([name for _, name in stack], content))
        body.clear()

    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if not match:
            body.append(line)
            continue
        flush()
        level, name = len(match.group(1)), match.group(2).strip()
        if level == 1:
            stack = []
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, name))
    flush()
    return sections


def _tail(text: str, overlap: int) -> str:
    """Last `overlap` characters, starting at a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def _split_long_piece(piece: str, max_chars: int) -> list[str]:
    """Split one oversized paragraph on sentence ends, then on words."""
    sentences = re.split(r"(?<=[.!?])\s+", piece)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_chars and current:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def split_text(text: str, max_chars: int = 1000, overlap: int = 150) -> list[str]:
    """Pack paragraphs into pieces of at most ~max_chars, with overlap between pieces."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        paragraphs.extend(_split_long_piece(para, max_chars) if len(para) > max_chars else [para])

    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        pieces.append(current)
        carry = _tail(current, overlap)
        current = f"{carry}\n\n{para}" if carry and len(carry) + len(para) + 2 <= max_chars else para
    if current:
        pieces.append(current)
    return pieces


def chunk_document(doc: Document, max_chars: int = 1000, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section_path, body in split_into_sections(doc.text):
        for piece in split_text(body, max_chars=max_chars, overlap=overlap):
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{ordinal:03d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source=doc.source,
                    section_path=list(section_path),
                    text=piece,
                    ordinal=ordinal,
                )
            )
    return chunks


def chunk_documents(docs: list[Document], max_chars: int = 1000, overlap: int = 150) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, max_chars=max_chars, overlap=overlap)]
