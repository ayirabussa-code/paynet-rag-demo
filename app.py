import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from openai import OpenAI
from openai.error import OpenAIError
import os

# -------------------
# Set your OpenAI API key
# -------------------
OPENAI_API_KEY = os.getenv("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")  # make sure your env variable is set
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# -------------------
# Streamlit UI
# -------------------
st.title("PayNet RAG Demo")
st.write("Upload your documents and query using OpenAI embeddings.")

uploaded_files = st.file_uploader("Upload documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    # -------------------
    # Load documents
    # -------------------
    st.info("Loading documents...")
    loader = DirectoryLoader("")  # placeholder, you can implement file loaders per type
    documents = []
    for file in uploaded_files:
        content = file.read().decode("utf-8", errors="ignore")
        documents.append({"page_content": content, "metadata": {"filename": file.name}})
    
    # -------------------
    # Split documents into chunks
    # -------------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    # -------------------
    # Initialize embeddings
    # -------------------
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    except OpenAIError as e:
        if getattr(e, "code", "") == "insufficient_quota":
            st.error("OpenAI quota exceeded. Please check your plan.")
        else:
            st.error(f"OpenAI error: {e}")
        st.stop()
    
    # -------------------
    # Create or load vectorstore
    # -------------------
    st.info("Creating Chroma vectorstore...")
    try:
        vectordb = Chroma.from_documents(split_docs, embeddings)
        st.success("Vectorstore created successfully!")
    except OpenAIError as e:
        st.error(f"Error creating vectorstore: {e}")
        st.stop()
    
    # -------------------
    # Query section
    # -------------------
    query = st.text_input("Ask a question about your documents:")
    if query:
        st.info("Searching vectorstore...")
        results = vectordb.similarity_search(query, k=3)
        for i, doc in enumerate(results):
            st.write(f"**Result {i+1} from {doc.metadata.get('filename','Unknown')}**")
            st.write(doc.page_content)
