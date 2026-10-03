from sellerpolicy.chunker import Chunk, chunk_document, chunk_documents, split_into_sections, split_text
from sellerpolicy.loader import Document, load_directory


def make_doc(text: str) -> Document:
    return Document(doc_id="doc", title="Doc Title", source="doc.md", text=text)


def test_sections_build_breadcrumbs():
    text = "# T\n\nintro\n\n## A\n\n### A1\n\nbody a1\n\n## B\n\nbody b"
    sections = split_into_sections(text)
    assert sections == [([], "intro"), (["A", "A1"], "body a1"), (["B"], "body b")]


def test_heading_without_body_is_skipped():
    sections = split_into_sections("# T\n\n## Parent\n\n### Child\n\ntext")
    assert [path for path, _ in sections] == [["Parent", "Child"]]


def test_sibling_heading_pops_stack():
    sections = split_into_sections("## A\n### A1\nx\n### A2\ny")
    assert [path for path, _ in sections] == [["A", "A1"], ["A", "A2"]]


def test_chunk_metadata_and_citation():
    chunks = chunk_document(make_doc("# Doc Title\n\n## Appeals\n\n### Appeal timelines\n\n17 days."))
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "doc::000"
    assert chunk.breadcrumb == "Appeals > Appeal timelines"
    assert chunk.citation_label == "Doc Title § Appeal timelines"
    assert chunk.text_for_search.startswith("Doc Title\nAppeals > Appeal timelines\n")


def test_intro_chunk_citation_says_introduction():
    chunk = chunk_document(make_doc("# Doc Title\n\nSome intro."))[0]
    assert chunk.section == "Introduction"
    assert chunk.text_for_search == "Doc Title\nSome intro."


def test_short_text_is_not_split():
    assert split_text("short text", max_chars=100) == ["short text"]


def test_empty_text_gives_no_pieces():
    assert split_text("   ", max_chars=100) == []


def test_long_text_split_respects_max_and_overlaps():
    paragraphs = [f"Paragraph {i} " + "word " * 30 for i in range(10)]
    pieces = split_text("\n\n".join(paragraphs), max_chars=300, overlap=60)
    assert len(pieces) > 1
    assert all(len(p) <= 300 for p in pieces)
    # overlap: the start of piece 2 repeats the end of piece 1
    assert pieces[1].split("\n\n")[0] in pieces[0]


def test_single_huge_paragraph_is_split_by_words():
    pieces = split_text("x" * 50 + " " + "word " * 400, max_chars=200, overlap=0)
    assert all(len(p) <= 200 for p in pieces)
    assert "".join(pieces).replace(" ", "").count("word") == 400


def test_chunk_roundtrip_dict():
    chunk = Chunk("d::000", "d", "T", "s", ["A"], "text", 0)
    assert Chunk.from_dict(chunk.to_dict()) == chunk


def test_real_policies_have_unique_ids_and_breadcrumbs(real_policy_dir):
    chunks = chunk_documents(load_directory(real_policy_dir))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    appeal = [c for c in chunks if c.breadcrumb == "Deactivation and Appeals > Appeal timelines"]
    assert appeal and "17 days" in appeal[0].text
