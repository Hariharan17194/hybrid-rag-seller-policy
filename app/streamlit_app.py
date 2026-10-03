"""Streamlit chat UI: streamlit run app/streamlit_app.py  ->  http://localhost:8501"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from sellerpolicy.pipeline import SellerPolicyRAG  # noqa: E402

st.set_page_config(page_title="SellerPolicy Assistant", page_icon="📘", layout="wide")

EXAMPLES = [
    "How long does a seller have to appeal a deactivation?",
    "My buyer says the parcel never showed up and now I got some A2Z thing. How long do I have?",
    "Can I charge a restocking fee if the customer opened the box?",
    "What is the SAFE-T claim deadline?",
]


@st.cache_resource(show_spinner="Loading indexes...")
def get_rag() -> SellerPolicyRAG:
    return SellerPolicyRAG()


rag = get_rag()

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Retrieval mode", ["hybrid", "bm25", "vector"], horizontal=True)
    top_k = st.slider("Sources (top k)", 1, 10, rag.settings.top_k)
    rerank = st.toggle("Cross-encoder reranker", value=rag.settings.use_reranker,
                       help="Downloads BAAI/bge-reranker-base (~1 GB) on first use.")
    llm_available = rag.generator.llm_enabled
    use_llm = st.toggle("LLM answer (OpenAI)", value=llm_available, disabled=not llm_available,
                        help=None if llm_available else "Add OPENAI_API_KEY to .env to enable.")
    st.caption(f"Embedder: `{rag.settings.embedding_provider}`")
    if st.button("Clear chat"):
        st.session_state.messages = []
    st.divider()
    st.caption("Synthetic policies for a portfolio project. Not official Amazon or bank policy.")

st.title("📘 SellerPolicy Assistant")
st.caption("Hybrid search (BM25 + vectors + RRF) over Seller Support policies. Every answer cites its source.")

if not rag.is_ready():
    st.error("Index not built yet. Run `python -m scripts.build_index` in the project folder, then reload.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            star = "⭐ " if s["cited"] else ""
            st.markdown(f"**{star}[{s['number']}] {s['citation']}**")
            st.caption(f"BM25 rank: {s['bm25_rank'] or '-'} · vector rank: {s['vector_rank'] or '-'} · "
                       f"score: {s['score']:.4f}" + (f" · rerank: {s['rerank_score']:.3f}" if s["rerank_score"] is not None else ""))
            st.markdown(f"> {s['text'][:600]}{'…' if len(s['text']) > 600 else ''}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

if not st.session_state.messages:
    st.write("Try an example:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = example
            st.rerun()

question = st.chat_input("Ask a policy question...") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            answer = rag.answer(question, mode=mode, top_k=top_k, rerank=rerank, use_llm=use_llm)
        body = answer.text + (f"\n\n_{answer.note}_" if answer.note else "")
        body += f"\n\n<sub>retrieval: {answer.retrieval_mode} · answer: {answer.mode}</sub>"
        st.markdown(body, unsafe_allow_html=True)
        sources = [dict(s.to_dict(), number=i, cited=i in answer.cited) for i, s in enumerate(answer.sources, start=1)]
        if sources:
            render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": body, "sources": sources})
