# app.py

import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import os
import openai

# -------------------
# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("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")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    st.stop()

# -------------------
# Upload a Word document
uploaded_file = st.file_uploader("Upload a .docx file", type=["docx"])

if uploaded_file is not None:
    try:
        # Read the Word document
        doc = Document(uploaded_file)
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])
        
        if not full_text:
            st.warning("The uploaded document is empty.")
            st.stop()
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_text(full_text)
        
        st.success(f"Document split into {len(split_docs)} chunks.")

        # Initialize OpenAI embeddings
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        except openai.OpenAIError as e:
            st.error(f"OpenAI error during embeddings initialization: {e}")
            st.stop()
        
        # Create vector database
        try:
            vectordb = Chroma.from_documents(split_docs, embeddings)
            st.success("Vector database created successfully!")
        except openai.OpenAIError as e:
            st.error(f"OpenAI error while creating vector DB: {e}")
            st.stop()
        
        # Simple query input
        user_query = st.text_input("Ask a question about your document:")

        if user_query:
            try:
                docs = vectordb.similarity_search(user_query, k=3)
                st.subheader("Top matching document chunks:")
                for i, d in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}:** {d.page_content}")
            except openai.OpenAIError as e:
                st.error(f"OpenAI error during similarity search: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    except Exception as e:
        st.error(f"Error processing document: {e}")
