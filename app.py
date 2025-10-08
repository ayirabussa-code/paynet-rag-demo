import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from io import BytesIO
from docx import Document

st.set_page_config(page_title="DOCX RAG Demo", layout="wide")

st.title("RAG Demo with DOCX Upload")

# -------------------
# Step 1: Get OpenAI API key
# -------------------
if "OPENAI_API_KEY" not in st.session_state:
    api_key_input = st.text_input(
        "Enter your OpenAI API Key", type="password"
    )
    if api_key_input:
        st.session_state["OPENAI_API_KEY"] = api_key_input

# Stop execution if no API key provided
if "OPENAI_API_KEY" not in st.session_state:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

api_key = st.session_state["OPENAI_API_KEY"]

# -------------------
# Step 2: Upload DOCX files
# -------------------
uploaded_files = st.file_uploader(
    "Upload DOCX file(s)", type=["docx"], accept_multiple_files=True
)

if uploaded_files:
    all_docs = []

    for uploaded_file in uploaded_files:
        # Read DOCX content
        doc = Document(uploaded_file)
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        all_docs.append(full_text)

    # Split text into smaller chunks for embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    split_docs = text_splitter.create_documents(all_docs)

    st.write(f"Processed {len(split_docs)} text chunks from {len(uploaded_files)} file(s).")

    # -------------------
    # Step 3: Generate embeddings and vector store
    # -------------------
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(split_docs, embeddings)

    st.success("Vector store created successfully!")
