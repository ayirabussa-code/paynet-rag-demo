import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from docx import Document

# -------------------
# Make sure you have set your OpenAI API key:
# export OPENAI_API_KEY="your_openai_api_key"
# -------------------

st.title("Docx Upload + LangChain Vector Search Demo")

# Upload multiple DOCX files
uploaded_files = st.file_uploader("Upload DOCX files", type=["docx"], accept_multiple_files=True)

if uploaded_files:
    all_texts = []

    # Read text from each uploaded DOCX
    for uploaded_file in uploaded_files:
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])
        all_texts.append(text)

    st.write("Extracted text from uploaded documents:")
    st.write(all_texts)

    # Split text into chunks for embeddings
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = []
    for text in all_texts:
        split_docs.extend(splitter.split_text(text))

    st.write(f"Total chunks for embedding: {len(split_docs)}")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Create vector store from chunks
    vectordb = Chroma.from_texts(split_docs, embedding=embeddings)
    st.success("Vector store created successfully!")

    # Query the vector store
    query = st.text_input("Enter a query to search in uploaded documents:")
    if query:
        results = vectordb.similarity_search(query)
        st.write("Results:")
        for i, res in enumerate(results):
            st.write(f"{i+1}. {res.page_content}")
