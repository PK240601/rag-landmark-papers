"""Streamlit UI: upload PDFs, ask questions, see cited sources."""
import streamlit as st

import corpus
import rag

st.set_page_config(page_title="PDF Chat (RAG)", page_icon="📄")
st.title("📄 Ask the Papers That Built Modern AI")
st.caption("RAG over landmark AI research (Transformer, BERT, GPT-3, RAG, LoRA, ...) with page-level citations.")

with st.sidebar:
    st.header("Documents")
    if st.button("Load landmark AI papers", type="primary"):
        status = st.empty()
        corpus.load_corpus(progress=status.info)
        status.success("Corpus ready")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files and st.button("Index my uploads"):
        for f in files:
            with st.spinner(f"Indexing {f.name}..."):
                n = rag.ingest_pdf(f, f.name)
            st.success(f"{f.name}: {n} chunks")
    sources = rag.indexed_sources()
    st.caption("Indexed: " + (", ".join(sources) if sources else "nothing yet"))
    k = st.slider("Chunks to retrieve", 2, 10, 5)
    if st.button("Clear index"):
        rag.reset()
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("e.g. How does LoRA reduce trainable parameters vs full fine-tuning?"):
    if not rag.indexed_sources():
        st.warning("Upload and index a PDF first.")
        st.stop()
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        stream, chunks = rag.answer(question, st.session_state.messages, k)
        reply = st.write_stream(stream)
        with st.expander("Sources"):
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**[{i}] {c['source']}, page {c['page']}** (similarity {c['score']:.2f})")
                st.caption(c["text"][:400] + "...")
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply},
    ]
