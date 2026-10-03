import pytest

from sellerpolicy.loader import clean_text, load_directory, load_file


def test_clean_text_normalises_whitespace():
    raw = "Line one   \r\n\r\n\r\n\r\nLine two\t\tend  "
    assert clean_text(raw) == "Line one\n\nLine two end"


def test_clean_text_removes_bom():
    assert clean_text("﻿# Title") == "# Title"


def test_load_file_uses_h1_as_title(sample_dir):
    doc = load_file(sample_dir / "claims.md")
    assert doc.title == "A-to-z Guarantee Claims"
    assert doc.doc_id == "claims"


def test_load_file_title_fallback(tmp_path):
    path = tmp_path / "no_heading_here.txt"
    path.write_text("Just text.", encoding="utf-8")
    assert load_file(path).title == "No Heading Here"


def test_load_file_rejects_unknown_extension(tmp_path):
    path = tmp_path / "file.docx"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load_file(path)


def test_load_directory_reads_md_and_txt_sorted(sample_dir):
    docs = load_directory(sample_dir)
    assert [d.doc_id for d in docs] == ["appeals", "claims", "returns"]


def test_load_directory_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path / "nope")


def test_load_directory_skips_empty_files(tmp_path):
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")
    (tmp_path / "full.md").write_text("# Full\n\nBody", encoding="utf-8")
    assert [d.doc_id for d in load_directory(tmp_path)] == ["full"]


def test_real_policies_load(real_policy_dir):
    docs = load_directory(real_policy_dir)
    assert len(docs) == 7
    assert all(d.text.startswith("# ") for d in docs)
