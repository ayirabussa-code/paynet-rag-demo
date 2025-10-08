# app.py
import os
import streamlit as st

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from openai.error import RateLimitError

# -------------------
# Set your OpenAI API key
# -------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY environment variable.")
    st.stop()

# -------------------
# Upload documents
# -------------------
uploaded_files = st.file_uploader(
    "Upload your documents (PDF, TXT, DOCX)", accept_multiple_files=True
)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        documents.append({"page_content": content, "metadata": {"filename": uploaded_file.name}})

    # -------------------
    # Split documents into chunks
    # -------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(documents)

    # -------------------
    # Initialize embeddings
    # -------------------
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    except RateLimitError:
        st.error("OpenAI API quota exceeded. Cannot generate embeddings at this time.")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        st.stop()

    # -------------------
    # Vectorstore (with caching)
    # -------------------
    vectordb_path = "chroma_db"
    if os.path.exists(vectordb_path):
        vectordb = Chroma(persist_directory=vectordb_path, embedding_function=embeddings)
    else:
        try:
            vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory=vectordb_path)
            vectordb.persist()
        except RateLimitError:
            st.error("OpenAI API quota exceeded while creating embeddings.")
            st.stop()
        except Exception as e:
            st.error(f"Error creating vectorstore: {e}")
            st.stop()

    # -------------------
    # Retrieval QA
    # -------------------
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY),
        retriever=retriever,
        chain_type="stuff"
    )

    # -------------------
    # Ask user questions
    # -------------------
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        try:
            answer = qa.run(user_question)
            st.write("Answer:", answer)
        except RateLimitError:
            st.error("OpenAI API quota exceeded while generating answer.")
        except Exception as e:
            st.error(f"Error generating answer: {e}")
