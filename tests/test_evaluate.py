import json

from scripts.evaluate import first_relevant_rank, render_markdown, summarise
from sellerpolicy.chunker import Chunk
from sellerpolicy.config import PROJECT_ROOT
from sellerpolicy.results import RetrievedChunk


def item(doc_id, path):
    return RetrievedChunk(Chunk(f"{doc_id}::000", doc_id, "T", "s", path, "x", 0), 0.0)


def test_first_relevant_rank_matches_doc_and_section():
    q = {"expected_doc_id": "a", "section_keyword": "appeal timelines"}
    results = [item("b", ["Appeal timelines"]), item("a", ["Other"]), item("a", ["Deactivation", "Appeal timelines"])]
    assert first_relevant_rank(results, q) == 3
    assert first_relevant_rank(results[:2], q) is None


def test_summarise_hit_and_mrr():
    s = summarise([1, 5, None, 2], k=5)
    assert s["hit@1"] == 0.25
    assert s["hit@5"] == 0.75
    assert abs(s["mrr@5"] - (1 + 0.2 + 0.5) / 4) < 1e-9


def test_render_markdown_has_tables():
    questions = [{"id": "q1", "style": "keyword", "question": "Q?"}]
    md = render_markdown(questions, {"bm25": [1], "vector": [None], "hybrid": [2]}, 5, "hashing:1024", False)
    assert "| bm25 | 1.00 | 1.00 | 1.000 |" in md
    assert "| q1 | keyword | 1 | - | 2 | Q? |" in md


def test_questions_file_is_valid():
    questions = json.loads((PROJECT_ROOT / "evals" / "questions.json").read_text(encoding="utf-8"))
    assert len(questions) == 16
    assert {q["style"] for q in questions} == {"keyword", "paraphrase"}
    for q in questions:
        assert (PROJECT_ROOT / "data" / "policies" / f"{q['expected_doc_id']}.md").exists()
