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
st.title("📄 RAG Demo with Docx Upload (Latest Info Prioritized)")

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
    "Upload one or more DOCX files", type=["docx"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload at least one DOCX file.")
    st.stop()

# 3️⃣ Load documents with metadata for version
docs = []
for uploaded_file in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        loader = Docx2txtLoader(tmp_file.name)
        file_docs = loader.load()
        # Detect version from filename
        version = "v1"
        if "v2" in uploaded_file.name.lower():
            version = "v2"
        for doc in file_docs:
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["version"] = version
        docs.extend(file_docs)
    os.unlink(tmp_file.name)

# 4️⃣ Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# 5️⃣ Sort chunks by version (v2 first)
split_docs.sort(key=lambda x: x.metadata.get("version", "v1"), reverse=True)

# 6️⃣ Create embeddings and vectorstore
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(split_docs, embeddings)
except OpenAIError as e:
    st.error(f"OpenAI API error: {e}")
    st.stop()

st.success("✅ Documents processed and vectorstore created successfully!")

# 7️⃣ Ask a query and retrieve top result from latest version
query = st.text_input("Ask a question about your documents:")
if query:
    try:
        results = vectordb.similarity_search(query, k=1)  # only top chunk
        if results:
            doc = results[0]
            st.write(f"**Answer (latest info from {doc.metadata.get('source')}):**")
            st.write(doc.page_content)
        else:
            st.info("No matching information found in your documents.")
    except Exception as e:
        st.error(f"Error retrieving results: {e}")
