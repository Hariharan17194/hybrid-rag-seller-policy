# LEARN.md: build-order learning path

This guide walks through the project **in the order it was built**. Each phase lists what to read, what to run, and 2-3 self-check questions. Answer them out loud or in a notebook *before* moving on. If you can explain a phase to a colleague, you own it.

**Before you start:** activate your venv in the project folder (`.\.venv\Scripts\Activate.ps1`) and run commands from the project root. Every file starts with a docstring that explains the idea with a Seller Support analogy, so read that first.

Tip: while learning, set `EMBEDDING_PROVIDER=hashing` in `.env` so everything runs instantly and offline. Switch back to `sentence-transformers` in Phase 7 to see real semantic search.

## Phase 1: Setup, loading and chunking

**Read (in order)**

1. `sellerpolicy/config.py`: one settings object; `.env` overrides; `SecretStr` for keys.
2. `sellerpolicy/loader.py`: `Document`, `clean_text`, PDF pages become `## Page N` headings.
3. `sellerpolicy/chunker.py`: section-aware split, breadcrumbs, overlap, `text_for_search`.
4. One file in `data/policies/` to see the heading structure the chunker relies on.

**Run**

```powershell
python -m scripts.verify_setup
python -m scripts.inspect_chunks
python -m scripts.inspect_chunks --max-chars 400 --all
pytest tests/test_chunker.py -q
```

**Self-check**

1. Why do we split on headings first instead of every 1000 characters? What could go wrong with a fixed cut?
2. What is a breadcrumb (`Deactivation and Appeals > Appeal timelines`) and why does a chunk need it?
3. In what order does pydantic-settings pick a value: default, `.env`, environment variable?

## Phase 2: BM25 keyword search

**Read**

- `sellerpolicy/bm25_index.py`: start with `tokenize()`, then `BM25Index.build / save / load / search`.
- `tests/test_bm25.py`: the tests double as usage examples.

**Run**

```powershell
python -c "from sellerpolicy.bm25_index import tokenize; print(tokenize('My A2Z claim for 60-day orders'))"
pytest tests/test_bm25.py -q
```

Then try it in a Python shell:

```python
from sellerpolicy.loader import load_directory
from sellerpolicy.chunker import chunk_documents
from sellerpolicy.bm25_index import BM25Index
from sellerpolicy.config import settings

chunks = chunk_documents(load_directory(settings.data_dir))
idx = BM25Index(); idx.build([c.chunk_id for c in chunks], [c.text_for_search for c in chunks])
idx.search("SAFE-T claim", k=3)
idx.search("parcel never showed up", k=3)   # notice how weak this is
```

**Self-check**

1. Why is a rare word like "safe-t" worth more than "seller" in BM25 (what is IDF)?
2. Why do we keep `a-to-z` as one token and also normalise `A2Z` → `a-to-z`?
3. Why does `search()` drop chunks with score 0?

## Phase 3: Embeddings and the vector store

**Read**

- `sellerpolicy/embeddings.py`: the `Embedder` interface, the three providers and `get_embedder()`.
- `sellerpolicy/vector_store.py`: Chroma `PersistentClient`, cosine space, `similarity = 1 - distance`, the "built with" check.
- `tests/test_embeddings_and_vectors.py`

**Run**

```powershell
pytest tests/test_embeddings_and_vectors.py -q
```

```python
from sellerpolicy.embeddings import HashingEmbedder
e = HashingEmbedder()
a, b, c = (e.embed_query(t) for t in ["lost inventory reimbursement", "inventory lost, reimbursed", "ODR target"])
sum(x*y for x, y in zip(a, b)), sum(x*y for x, y in zip(a, c))   # which pair is closer?
```

**Self-check**

1. In one sentence, what does an embedding represent? Why does cosine similarity work for comparing them?
2. Why do we compute embeddings ourselves instead of letting Chroma do it?
3. Why must you rebuild the index after changing `EMBEDDING_PROVIDER`? What does the code do if you forget?

## Phase 4: Hybrid fusion (RRF) and reranking

**Read**

- `sellerpolicy/hybrid.py`: work through the numeric example in the docstring **by hand** with a calculator.
- `tests/test_hybrid.py`: the same example as a test, and how `k` changes the outcome.
- `sellerpolicy/reranker.py`: bi-encoder vs cross-encoder, lazy loading, fallback.

**Run**

```powershell
pytest tests/test_hybrid.py tests/test_reranker.py -q
python -m scripts.build_index
python -m scripts.compare_search
python -m scripts.compare_search "can I charge a fee if the customer opened the box?"
```

**Self-check**

1. Why can't we just add the BM25 score and the cosine score together?
2. Compute the RRF score for a chunk ranked 2nd by BM25 and 5th by vector search (k = 60).
3. Why do we rerank only the top ~20 candidates and not the whole library?

## Phase 5: Prompting and answer generation

**Read**

- `sellerpolicy/results.py`: `RetrievedChunk` and `Answer`, the data that flows out of retrieval.
- `sellerpolicy/prompts.py`: the rules, `NOT_FOUND_MESSAGE`, and how context is numbered.
- `sellerpolicy/generator.py`: LLM mode, extractive mode, `extract_citations`, error fallback.
- `tests/test_generator.py`: note the `FakeOpenAI` class, which tests the LLM path without the internet.

**Run**

```powershell
python -m scripts.ask "How long does a seller have to appeal a deactivation?"
python -m scripts.ask "What is the capital of France?"     # how does it behave off-topic?
pytest tests/test_generator.py -q
```

With an OpenAI key in `.env`, run the same questions again and compare the two modes.

**Self-check**

1. Which three prompt rules stop the model from "making up" policy? Map each to a Seller Support QA rule.
2. Why do we parse `[n]` citations out of the answer, and why ignore numbers larger than the source count?
3. When does the app use extractive mode, and why is it useful even after you add a key?

## Phase 6: The pipeline, the UI and the API

**Read**

- `sellerpolicy/pipeline.py`: `build_index()`, lazy properties, `retrieve()` (search → fuse → rerank), `answer()`.
- `scripts/build_index.py`, `scripts/ask.py`
- `app/streamlit_app.py`: `st.cache_resource`, `st.session_state`, the sidebar controls.
- `api/main.py`: pydantic request/response models, `Depends(get_rag)`, 422 and 503 errors.
- `tests/test_pipeline.py`, `tests/test_api.py`, `tests/conftest.py` (fixtures)

**Run**

```powershell
streamlit run app/streamlit_app.py
uvicorn api.main:app --reload --port 8000        # then open http://localhost:8000/docs
python -m scripts.build_index --data-dir data/banking_sample
python -m scripts.ask "When will the bank give me a shadow credit?"
python -m scripts.build_index                    # switch back to seller policies
pytest tests/test_pipeline.py tests/test_api.py -q
```

**Self-check**

1. Why are the embedder and reranker created lazily? What would happen to a BM25-only question otherwise?
2. Why does `chunks.json` exist if BM25 and Chroma already store the text?
3. How do the API tests swap in a temporary index (`dependency_overrides`)? Why is that better than using your real `storage/`?

## Phase 7: Evaluation, real models and shipping

**Read**

- `evals/questions.json`: how each question is labelled (`expected_doc_id`, `section_keyword`, `style`).
- `scripts/evaluate.py`: `first_relevant_rank`, `summarise` (Hit@1, Hit@k, MRR).
- `Dockerfile`, `.env.example`, `requirements.txt`

**Run**

```powershell
# 1) Offline baseline
$env:EMBEDDING_PROVIDER="hashing"; python -m scripts.build_index; python -m scripts.evaluate

# 2) Real semantic embeddings (downloads ~90 MB once)
$env:EMBEDDING_PROVIDER="sentence-transformers"; python -m scripts.build_index; python -m scripts.evaluate

# 3) Add the reranker (downloads ~1 GB once)
python -m scripts.evaluate --rerank

# 4) Full test suite and container
pytest -q
docker build -t sellerpolicy .
```

**Self-check**

1. What is the difference between Hit@5 and MRR? Give an example where Hit@5 is the same but MRR differs.
2. Did hybrid beat BM25 on the *paraphrase* questions with real embeddings? Why would that happen?
3. Write two new questions for `evals/questions.json` that you think BM25 will miss, then check whether you were right.

## Stretch goals (after Phase 7)

- Add metadata filtering (search only one document family via Chroma `where={"doc_id": ...}`).
- Add an LLM-as-judge faithfulness check (does every sentence have a citation that supports it?).
- Stream the LLM answer token by token in Streamlit (`stream=True`).
- Add query rewriting: turn a long seller email into 1-3 short search queries before retrieval.
