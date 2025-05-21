import os
import streamlit as st
import tempfile

from backend_utils import (
    extract_chunks_from_pdf,
    embed_chunks,
    save_chunks_to_chroma,
    chroma_vector_search_and_rerank,
    summarize_context
)

with open("openai_api_key.txt") as f:
    os.environ['OPENAI_API_KEY'] = f.read().strip()

# Set up app page
st.set_page_config(page_title="MSME Policy Chatbot", layout="wide")
st.title("📄 MSME Policy Chatbot with Contextual Memory & ChromaDB")


# Session state: chunks & history
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "history" not in st.session_state:
    st.session_state.history = []


uploaded_file = st.file_uploader("Upload a MSME Policy PDF", type=["pdf"])

if uploaded_file:
    # Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    pdf_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
    with st.spinner("Extracting and embedding..."):
        chunks = extract_chunks_from_pdf(tmp_file_path)
        chunks = embed_chunks(chunks)
        st.session_state.chunks = chunks
        # Save to ChromaDB
        save_chunks_to_chroma(chunks, pdf_name)
    st.success(f"Extracted, embedded, and saved {len(st.session_state.chunks)} sections to ChromaDB.")


# Chunk Inspector
if st.session_state.chunks:
    st.markdown("### 🧩 Chunk Inspector")
    for idx, chunk in enumerate(st.session_state.chunks):
        with st.expander(f"Chunk {idx+1} (Page {chunk.get('page')}, Type: {chunk.get('type', 'N/A')})"):
            st.write("**Text:**")
            st.code(chunk["text"][:500])
            st.write("**Metadata:**")
            st.json({k: v for k, v in chunk.items() if k != "embedding"})

query = st.text_input("Ask a question about the policy")
max_history = st.slider("How many previous queries to keep in context?", min_value=1, max_value=20, value=10)


if query:
    st.session_state.history.append(query)
    if len(st.session_state.history) > max_history:
        st.session_state.history = st.session_state.history[-max_history:]
    context_summary = summarize_context(st.session_state.history[:-1])
    full_query = f"{context_summary}\nUser question: {query}" if context_summary else query
    with st.spinner("Retrieving the most relevant policy sections..."):
        results = chroma_vector_search_and_rerank(full_query)
    st.markdown("### 🔍 Top Matching Responses")
    for chunk, score in results:
        meta = chunk['metadata']
        st.markdown(f"**📄 Page {meta.get('page')} | 🔢 Score: {score:.2f}**")
        st.code(chunk["text"])

# Query history display
if st.session_state.history:
    st.markdown("**Query history (for context):**")
    for i, h in enumerate(st.session_state.history, 1):
        st.write(f"{i}. {h}")
