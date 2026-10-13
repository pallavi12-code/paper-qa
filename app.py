import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

from sciqa.index import VectorStore, sentence_transformer_embedder
from sciqa.llm import generate
from sciqa.pdf_loader import load_pdf
from sciqa.qa import answer

st.set_page_config(page_title="Paper QA", page_icon="📄")
st.title("📄 Scientific Paper QA")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return sentence_transformer_embedder()


with st.sidebar:
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    k = st.slider("Passages to retrieve", 3, 10, 5)

# Rebuild the index only when the uploaded files change
key = tuple((f.name, f.size) for f in files or [])
if st.session_state.get("key") != key:
    store = VectorStore(get_embedder())
    with st.spinner("Indexing PDFs..."):
        for f in files or []:
            store.add(load_pdf(f.getvalue(), f.name))
    st.session_state.update(key=key, store=store, messages=[])

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for src in m.get("sources", []):
            with st.expander(src["label"]):
                st.write(src["text"])

if q := st.chat_input("Ask about your papers"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                text, hits = answer(q, st.session_state.store, generate, k=k)
            except Exception as e:
                text, hits = f"LLM error: {e}", []
        st.markdown(text)
        sources = [
            {"label": f"{c.source} — p.{c.page} (score {s:.2f})", "text": c.text}
            for c, s in hits
        ]
        for src in sources:
            with st.expander(src["label"]):
                st.write(src["text"])
    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources}
    )
