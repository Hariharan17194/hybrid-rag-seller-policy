"""Prompts for grounded, cited answers.

Each rule mirrors a Seller Support QA rule:
- "Use ONLY the sources"          -> answer only from approved policy pages
- "Cite every fact with [n]"      -> link the policy page you relied on
- "If not covered, say exactly…"  -> escalate instead of improvising; never guess timelines
"""

from __future__ import annotations

from sellerpolicy.results import RetrievedChunk

NOT_FOUND_MESSAGE = "I couldn't find this in the policy documents."

SYSTEM_PROMPT = f"""You are SellerPolicy Assistant, helping Seller Support associates answer seller questions.

Rules:
1. Use ONLY the numbered policy sources provided. Never use outside knowledge.
2. Cite every fact with its source number in square brackets, e.g. [1] or [2][3].
3. Quote numbers, timelines and amounts exactly as written in the sources. Never estimate.
4. If the sources do not answer the question, reply exactly: "{NOT_FOUND_MESSAGE}"
5. Be concise: a direct answer first, then short supporting points if useful.
6. Never promise an outcome (e.g. "your account will be reinstated")."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for number, item in enumerate(chunks, start=1):
        blocks.append(f"[{number}] {item.citation_label}\nSection: {item.chunk.breadcrumb or 'Introduction'}\n{item.chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"Policy sources:\n\n{build_context(chunks)}\n\nQuestion: {question}\n\nAnswer with [n] citations."
