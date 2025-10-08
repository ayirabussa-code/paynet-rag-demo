import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
import tempfile
import os
import openai
import re
from openai import OpenAIError

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="RAG Demo with Version Awareness", layout="wide")
st.title("📘 Version-Aware RAG Demo")

# 1️⃣ Ask user for OpenAI API key
api_key = st.text_input(
    "Enter your OpenAI API Key",
    type="password",
    placeholder="sk-...",
)
if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# 2️⃣ Upload .docx files
uploaded_files = st.file_uploader(
    "Upload one or more DOCX files (e.g., Guide_v1.docx, Guide_v2.docx)",
    type=["docx"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload at least one DOCX file.")
    st.stop()

# -----------------------------
# 3️⃣ Load and process documents
# -----------------------------
docs = []
for uploaded_file in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        loader = Docx2txtLoader(tmp_file.name)
        loaded_docs = loader.load()
        # Store filename in metadata
        for d in loaded_docs:
            d.metadata["source"] = uploaded_file.name
        docs.extend(loaded_docs)
    os.unlink(tmp_file.name)

# -----------------------------
# 4️⃣ Split documents into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
split_docs = text_splitter.split_documents(docs)

# -----------------------------
# 5️⃣ Detect latest version
# -----------------------------
def extract_version(filename):
    """Extract numeric version (e.g., v2 → 2)."""
    match = re.search(r'v(\d+)', filename.lower())
    return int(match.group(1)) if match else 0  # default to v0 if no tag

uploaded_files_sorted = sorted(uploaded_files, key=lambda f: extract_version(f.name), reverse=True)
latest_file = uploaded_files_sorted[0]
latest_filename = latest_file.name
st.info(f"📗 Prioritizing latest version: **{latest_filename}**")

# -----------------------------
# 6️⃣ Create embeddings & vectorstore
# -----------------------------
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(split_docs, embeddings)
except OpenAIError as e:
    st.error(f"OpenAI API error: {e}")
    st.stop()

st.success("✅ Documents processed and vectorstore created successfully!")

# -----------------------------
# 7️⃣ Ask a question
# -----------------------------
query = st.text_input("🔍 Ask a question about your documents:")
if query:
    try:
        # Retrieve results
        results = vectordb.similarity_search(query, k=6)

        # Prioritize chunks from latest version
        prioritized = [r for r in results if latest_filename in r.metadata.get("source", "")]
        others = [r for r in results if latest_filename not in r.metadata.get("source", "")]
        ordered_results = prioritized + others

        st.write(f"Top matching chunks (prioritizing {latest_filename}):")
        for i, doc in enumerate(ordered_results[:4]):  # show top 4
            source = doc.metadata.get("source", "Unknown")
            st.markdown(f"**Chunk {i+1} (from {source}):**")
            st.write(doc.page_content[:500] + "...")
    except Exception as e:
        st.error(f"Error retrieving results: {e}")
