import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from docx import Document as DocxDocument
from openai.error import RateLimitError, AuthenticationError

# -------------------------
# Streamlit UI
# -------------------------
st.title("DOCX RAG Demo with OpenAI Embeddings")

# Ask for OpenAI API key
api_key = st.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    help="You can get your API key from https://platform.openai.com/account/api-keys"
)

if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# Upload DOCX files
uploaded_files = st.file_uploader(
    "Upload one or more DOCX files",
    type=["docx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload at least one DOCX file to continue.")
    st.stop()

# -------------------------
# Read DOCX content
# -------------------------
def read_docx(file):
    doc = DocxDocument(file)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    return "\n".join(full_text)

all_texts = []
for file in uploaded_files:
    text = read_docx(file)
    all_texts.append(Document(page_content=text, metadata={"filename": file.name}))

# -------------------------
# Split text into chunks
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_docs = text_splitter.split_documents(all_texts)

st.success(f"Total chunks created: {len(split_docs)}")

# -------------------------
# Create embeddings and vector store
# -------------------------
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(split_docs, embeddings)
    st.success("Vector store created successfully!")
except AuthenticationError:
    st.error("Invalid API key. Please check your key and try again.")
    st.stop()
except RateLimitError:
    st.error(
        "You have exceeded your OpenAI API quota. "
        "Please check your plan and billing details or try later."
    )
    st.stop()
except Exception as e:
    st.error(f"Unexpected error: {e}")
    st.stop()

# -------------------------
# Simple search interface
# -------------------------
query = st.text_input("Enter your search query:")

if query:
    results = vectordb.similarity_search(query)
    if results:
        st.write("Top results:")
        for i, doc in enumerate(results[:5]):
            st.write(f"**Result {i+1}** (from `{doc.metadata.get('filename')}`):")
            st.write(doc.page_content)
    else:
        st.info("No matching results found.")
