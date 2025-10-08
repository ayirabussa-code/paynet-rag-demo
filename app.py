import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
import tempfile
import os
import openai
from openai import OpenAIError

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("📄 RAG Demo with Docx Upload (Latest Info)")

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
    "Upload one or more DOCX files (v1, v2, etc.)",
    type=["docx"],
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload at least one DOCX file.")
    st.stop()

# 3️⃣ Load documents
docs = []
for uploaded_file in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        loader = Docx2txtLoader(tmp_file.name)
        loaded_docs = loader.load()
        # Add metadata: source filename
        for doc in loaded_docs:
            doc.metadata["source"] = uploaded_file.name
            # Optional: mark version from filename (v1, v2)
            if "v2" in uploaded_file.name.lower():
                doc.metadata["version"] = "v2"
            else:
                doc.metadata["version"] = "v1"
        docs.extend(loaded_docs)
    os.unlink(tmp_file.name)  # delete temp file

# 4️⃣ Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
split_docs = text_splitter.split_documents(docs)

# 5️⃣ Create embeddings
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
except OpenAIError as e:
    st.error(f"OpenAI API error: {e}")
    st.stop()

st.success("✅ Documents processed successfully!")

# 6️⃣ Ask a query and retrieve top result from latest version
query = st.text_input("Ask a question about your documents:")
if query:
    try:
        # Filter latest version first
        latest_docs = [d for d in split_docs if d.metadata.get("version") == "v2"]
        if not latest_docs:
            latest_docs = split_docs  # fallback to all docs

        vectordb_latest = Chroma.from_documents(latest_docs, embeddings)
        results = vectordb_latest.similarity_search(query, k=1)  # only top chunk

        if results:
            doc = results[0]
            st.write(f"**Answer (latest info from {doc.metadata.get('source')}):**")
            st.write(doc.page_content)
        else:
            st.info("No matching information found in the latest documents.")

    except Exception as e:
        st.error(f"Error retrieving results: {e}")
