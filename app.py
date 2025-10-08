import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
import tempfile
import os
from openai import OpenAI, OpenAIError

st.set_page_config(page_title="RAG Demo - Latest Answer", layout="wide")
st.title("📄 RAG Demo - Latest Answer from DOCX Files")

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
    "Upload one or more DOCX files (e.g., v1, v2)", 
    type=["docx"], 
    accept_multiple_files=True
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# 5️⃣ Create embeddings and vectorstore
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_documents(split_docs, embeddings)
except OpenAIError as e:
    st.error(f"OpenAI API error: {e}")
    st.stop()

st.success("✅ Documents processed and vectorstore created successfully!")

# 6️⃣ Ask a query and retrieve the **single most relevant answer**
query = st.text_input("Ask a question about your documents:")
if query:
    try:
        # Retrieve top 3 chunks
        results = vectordb.similarity_search(query, k=3)

        if not results:
            st.info("No relevant content found in uploaded documents.")
        else:
            # Combine retrieved chunks for latest info
            combined_text = "\n\n".join(
                [f"(from {getattr(doc, 'metadata', {}).get('source', 'unknown file')}): {doc.page_content}" 
                 for doc in results]
            )

            # Generate final answer using new OpenAI v1 API
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant summarizing the latest information."},
                    {"role": "user", "content": f"Answer the question using the latest info only:\n\nDocuments:\n{combined_text}\n\nQuestion: {query}"}
                ],
                temperature=0
            )

            answer = response.choices[0].message.content
            st.markdown("**Answer (latest info):**")
            st.write(answer)

    except OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
    except Exception as e:
        st.error(f"Error retrieving results: {e}")
