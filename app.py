import os
import streamlit as st

# -------------------
# OpenAI API Key Handling
# -------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning(
        "OpenAI API key not found. Please enter your key to continue."
    )
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
    
if OPENAI_API_KEY:
    # Save key for session use
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # -------------------
    # Importing LangChain Modules
    # -------------------
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.text_splitter import CharacterTextSplitter

    # -------------------
    # Sample Document Setup
    # -------------------
    st.title("LangChain + Chroma Embeddings Demo")

    sample_docs = [
        "Hello world! This is a test document.",
        "LangChain makes working with embeddings easier.",
        "Streamlit allows you to build web apps in Python quickly."
    ]

    st.subheader("Sample Documents")
    st.write(sample_docs)

    # -------------------
    # Text Splitting
    # -------------------
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=50,
        chunk_overlap=5
    )
    split_docs = text_splitter.split_documents(sample_docs)
    st.subheader("Split Documents")
    st.write(split_docs)

    # -------------------
    # Create Embeddings & Vector Store
    # -------------------
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    try:
        vectordb = Chroma.from_documents(split_docs, embeddings)
        st.success("Vector store created successfully!")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

    # -------------------
    # Query Example
    # -------------------
    query = st.text_input("Enter a query to search embeddings:")

    if query and vectordb:
        results = vectordb.similarity_search(query)
        st.subheader("Query Results")
        st.write(results)

else:
    st.stop()  # Stop execution until API key is provided
