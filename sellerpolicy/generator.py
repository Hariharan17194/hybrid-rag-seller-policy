"""Answer generation: OpenAI LLM with citations, or an extractive fallback.

Seller Support analogy: the LLM is an associate writing a reply from the policy
pages you hand them. Extractive mode is pasting the most relevant policy sentences
with their links: less fluent, but it cannot invent anything and needs no API key.

If the LLM call fails (network, quota), we fall back to extractive mode instead of
showing the associate an error.
"""

from __future__ import annotations

import logging
import re

from sellerpolicy.bm25_index import tokenize
from sellerpolicy.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from sellerpolicy.results import Answer, RetrievedChunk

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_citations(text: str, n_sources: int) -> list[int]:
    """Unique [n] numbers in order of appearance; numbers outside 1..n_sources are hallucinated and dropped."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        number = int(match.group(1))
        if 1 <= number <= n_sources and number not in seen:
            seen.append(number)
    return seen


def _sentences(text: str) -> list[str]:
    lines = [ln.strip(" -\t") for ln in text.splitlines() if ln.strip() and not ln.strip().startswith(">")]
    joined = " ".join(lines)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", joined) if len(s.strip()) > 20]


def extractive_answer(question: str, chunks: list[RetrievedChunk], max_sentences: int = 3, max_sources: int = 3) -> tuple[str, list[int]]:
    """Pick the sentences from the top sources that share the most words with the question."""
    query_terms = set(tokenize(question))
    candidates: list[tuple[float, int, int, str]] = []
    for number, item in enumerate(chunks[:max_sources], start=1):
        for position, sentence in enumerate(_sentences(item.chunk.text)):
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap:
                # Prefer more overlap, then higher-ranked sources, then earlier sentences.
                candidates.append((overlap - 0.1 * (number - 1), number, position, sentence))
    if not candidates:
        return NOT_FOUND_MESSAGE, []
    best = sorted(candidates, key=lambda c: (-c[0], c[1], c[2]))[:max_sentences]
    best.sort(key=lambda c: (c[1], c[2]))
    lines = [f"- {sentence} [{number}]" for _, number, _, sentence in best]
    cited = []
    for _, number, _, _ in best:
        if number not in cited:
            cited.append(number)
    return "Most relevant policy excerpts:\n\n" + "\n".join(lines), cited


class Generator:
    def __init__(self, settings, client=None) -> None:
        self.settings = settings
        self._client = client

    @property
    def llm_enabled(self) -> bool:
        return self._client is not None or self.settings.has_openai_key

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.settings.openai_api_key.get_secret_value())
        return self._client

    def _extractive(self, question: str, chunks: list[RetrievedChunk], note: str | None = None) -> Answer:
        text, cited = extractive_answer(question, chunks)
        mode = "extractive" if cited else "not_found"
        return Answer(question=question, text=text, sources=chunks, cited=cited, mode=mode, note=note)

    def generate(self, question: str, chunks: list[RetrievedChunk], use_llm: bool | None = None) -> Answer:
        if not chunks:
            return Answer(question=question, text=NOT_FOUND_MESSAGE, sources=[], mode="not_found")
        if use_llm is None:
            use_llm = self.llm_enabled
        if not use_llm:
            return self._extractive(question, chunks)
        try:
            response = self._get_client().chat.completions.create(
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, chunks)},
                ],
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("LLM call failed (%s); using extractive mode.", exc)
            return self._extractive(question, chunks, note=f"LLM unavailable ({type(exc).__name__}); showing excerpts.")
        if not text or NOT_FOUND_MESSAGE.lower().rstrip(".") in text.lower():
            return Answer(question=question, text=NOT_FOUND_MESSAGE, sources=chunks, mode="not_found")
        return Answer(question=question, text=text, sources=chunks, cited=extract_citations(text, len(chunks)), mode="llm")
