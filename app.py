import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="LangChain + OpenAIEmbeddings", layout="wide")
st.title("LangChain + OpenAIEmbeddings Demo")

# Get OpenAI API key
api_key = st.text_input("Enter your OpenAI API key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# ---------------------------
# File upload
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload text files (.txt) or Word files (.docx) to create embeddings", 
    accept_multiple_files=True,
    type=["txt", "docx"]
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")

    docs = []
    for file in uploaded_files:
        if file.type == "text/plain":
            content = file.read().decode("utf-8")
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            from docx import Document as DocxDocument
            docx_file = DocxDocument(file)
            content = "\n".join([p.text for p in docx_file.paragraphs])
        else:
            st.error(f"Unsupported file type: {file.type}")
            continue
        
        docs.append(Document(page_content=content, metadata={"source": file.name}))
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    st.info(f"Text split into {len(split_docs)} chunks for embedding.")

    # Create vector store
    vectordb = Chroma.from_documents(split_docs, embeddings)

    st.success("Embeddings created and stored in memory.")

    # ---------------------------
    # Query interface
    # ---------------------------
    query = st.text_input("Enter a query to search the uploaded documents:")

    if query:
        results = vectordb.similarity_search(query, k=3)
        st.subheader("Top 3 matching chunks:")
        for i, doc in enumerate(results, 1):
            st.write(f"**Result {i}** from `{doc.metadata['source']}`:")
            st.write(doc.page_content)
