import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile
import os
import openai
from openai import OpenAIError

st.set_page_config(page_title="RAG Demo", layout="wide")

st.title("📄 RAG Demo with Docx Upload + Q&A")

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

# 3️⃣ Load documents
docs = []
for uploaded_file in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        loader = Docx2txtLoader(tmp_file.name)
        docs.extend(loader.load())
    os.unlink(tmp_file.name)  # delete temp file

# 4️⃣ Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
split_docs = text_splitter.split_documents(docs)

# 5️⃣ Create embeddings and vectorstore
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(split_docs, embeddings)
except OpenAIError as e:
    st.error(f"OpenAI API error: {e}")
    st.stop()

st.success("✅ Documents processed and vectorstore created successfully!")

# 6️⃣ Ask a query
query = st.text_input("Ask a question about your documents:")

if query:
    try:
        # Retrieval step
        results = vectordb.similarity_search(query, k=4)
        st.subheader("Top matching chunks:")
        for i, doc in enumerate(results):
            st.markdown(f"**Chunk {i+1}:** {doc.page_content[:400]}...")

        # Generation step
        llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=False
        )

        st.markdown("### 💬 Final Answer:")
        answer = qa_chain.run(query)
        st.write(answer)

    except Exception as e:
        st.error(f"Error retrieving results: {e}")
