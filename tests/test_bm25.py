import pytest

from sellerpolicy.bm25_index import BM25Index, tokenize


@pytest.mark.parametrize("text", ["A-to-z claim", "A to Z claim", "A2Z claim", "a-to-z claim", "AtoZ claim"])
def test_tokenize_normalises_a_to_z(text):
    assert tokenize(text) == ["a-to-z", "claim"]


def test_tokenize_keeps_codes_and_numbers():
    assert tokenize("SAFE-T within 60-day, rate 2.5%") == ["safe-t", "within", "60-day", "rate", "2.5"]


def test_tokenize_drops_stopwords_and_plurals():
    assert tokenize("What are the claims for the orders") == ["claim", "order"]


def test_tokenize_keeps_double_s():
    assert tokenize("business address") == ["business", "address"]


@pytest.fixture
def index():
    idx = BM25Index()
    idx.build(
        ["a", "b", "c"],
        [
            "SAFE-T claim within 30 days of the refund",
            "A-to-z Guarantee claim response window 72 hours",
            "Order defect rate target under 1%",
        ],
    )
    return idx


def test_search_ranks_rare_term_first(index):
    assert index.search("SAFE-T claim", k=3)[0][0] == "a"


def test_search_a2z_alias(index):
    assert index.search("my A2Z thing", k=1)[0][0] == "b"


def test_search_drops_zero_scores(index):
    ids = [cid for cid, _ in index.search("defect", k=3)]
    assert ids == ["c"]


def test_search_unknown_words_returns_empty(index):
    assert index.search("capital of France", k=3) == []


def test_save_and_load_roundtrip(index, tmp_path):
    index.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25")
    assert loaded.search("SAFE-T", k=1) == index.search("SAFE-T", k=1)


def test_search_before_build_raises():
    with pytest.raises(RuntimeError):
        BM25Index().search("x")


def test_build_validates_input():
    with pytest.raises(ValueError):
        BM25Index().build(["a"], [])
    with pytest.raises(ValueError):
        BM25Index().build([], [])


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BM25Index.load(tmp_path / "missing")
