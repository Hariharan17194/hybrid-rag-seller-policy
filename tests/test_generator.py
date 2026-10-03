from sellerpolicy.chunker import Chunk
from sellerpolicy.config import Settings
from sellerpolicy.generator import Generator, extract_citations, extractive_answer
from sellerpolicy.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_context
from sellerpolicy.results import RetrievedChunk


def sources():
    return [
        RetrievedChunk(Chunk("a::000", "a", "Account Health", "s", ["Deactivation and Appeals", "Appeal timelines"],
                             "A seller may submit an appeal within 17 days of the deactivation notice.", 0), 0.03),
        RetrievedChunk(Chunk("b::000", "b", "Payments", "s", ["Settlement cycle"],
                             "Seller payments are calculated in a settlement period of 14 days.", 0), 0.02),
    ]


class FakeOpenAI:
    """Mimics client.chat.completions.create without the internet."""

    def __init__(self, reply: str | None = None, error: Exception | None = None):
        self.reply, self.error, self.calls = reply, error, []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        message = type("M", (), {"content": self.reply})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


def settings():
    return Settings(_env_file=None, openai_api_key=None, embedding_provider="hashing")


def test_extract_citations_unique_and_in_range():
    assert extract_citations("Yes [2]. Also [1][2] and [7].", n_sources=3) == [2, 1]


def test_extract_citations_none():
    assert extract_citations("No citations here.", 3) == []


def test_build_context_numbers_sources():
    ctx = build_context(sources())
    assert ctx.startswith("[1] Account Health § Appeal timelines")
    assert "[2] Payments § Settlement cycle" in ctx


def test_system_prompt_contains_rules():
    assert NOT_FOUND_MESSAGE in SYSTEM_PROMPT
    assert "ONLY" in SYSTEM_PROMPT


def test_extractive_answer_picks_relevant_sentence():
    text, cited = extractive_answer("How long to appeal a deactivation?", sources())
    assert "17 days" in text and "[1]" in text
    assert cited == [1]


def test_extractive_answer_not_found_without_overlap():
    text, cited = extractive_answer("capital of France", sources())
    assert text == NOT_FOUND_MESSAGE and cited == []


def test_generate_extractive_when_no_key():
    gen = Generator(settings())
    assert not gen.llm_enabled
    answer = gen.generate("appeal deactivation days", sources())
    assert answer.mode == "extractive" and answer.found


def test_generate_no_sources_is_not_found():
    answer = Generator(settings()).generate("anything", [])
    assert answer.mode == "not_found" and answer.text == NOT_FOUND_MESSAGE


def test_generate_llm_path_with_fake_client():
    fake = FakeOpenAI(reply="You have 17 days to appeal [1].")
    answer = Generator(settings(), client=fake).generate("appeal?", sources())
    assert answer.mode == "llm"
    assert answer.cited == [1]
    sent = fake.calls[0]["messages"]
    assert sent[0]["role"] == "system" and "[1] Account Health" in sent[1]["content"]


def test_generate_llm_not_found_reply():
    fake = FakeOpenAI(reply=NOT_FOUND_MESSAGE)
    answer = Generator(settings(), client=fake).generate("capital of France?", sources())
    assert answer.mode == "not_found"


def test_generate_llm_error_falls_back_to_extractive():
    fake = FakeOpenAI(error=ConnectionError("offline"))
    answer = Generator(settings(), client=fake).generate("appeal deactivation days", sources())
    assert answer.mode == "extractive"
    assert "ConnectionError" in answer.note


def test_use_llm_false_forces_extractive():
    fake = FakeOpenAI(reply="x [1]")
    answer = Generator(settings(), client=fake).generate("appeal deactivation", sources(), use_llm=False)
    assert answer.mode == "extractive" and not fake.calls
