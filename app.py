
import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

CHUNK_FOLDER = "rag_docs"
EMBEDDINGS_MODEL = OpenAIEmbeddings()
LLM_MODEL = OpenAI(temperature=0)

def load_chunks(folder):
    all_text = []
    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder, file_name), "r", encoding="utf-8") as f:
                all_text.append(f.read())
    return all_text

def build_vector_store(chunks):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_texts = []
    for chunk in chunks:
        split_texts.extend(text_splitter.split_text(chunk))
    vector_store = Chroma.from_texts(split_texts, EMBEDDINGS_MODEL)
    return vector_store

st.title("PayNet API RAG Demo for Technical Writers")
st.markdown("Ask questions about PayNet API. The system retrieves info from v1 and v2 docs and provides context-aware answers.")

chunks = load_chunks(CHUNK_FOLDER)
vector_store = build_vector_store(chunks)
qa_chain = RetrievalQA.from_chain_type(llm=LLM_MODEL, chain_type="stuff", retriever=vector_store.as_retriever())

query = st.text_input("Enter your question here:")

if query:
    with st.spinner("Fetching answer..."):
        answer = qa_chain.run(query)
    st.subheader("Answer:")
    st.write(answer)
