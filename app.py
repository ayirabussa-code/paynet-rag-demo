import streamlit as st
import openai
import os
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# Streamlit UI
# -------------------------
st.title("📘 Smart RAG App with Version Control (v1/v2 Priority)")

# Step 1: API Key Input
api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

if api_key:
    openai.api_key = api_key

    # Step 2: File Upload
    uploaded_files = st.file_uploader("📂 Upload one or more DOCX files", type=["docx"], accept_multiple_files=True)

    if uploaded_files:
        # Step 3: Process uploaded documents
        docs_text = {}
        for file in uploaded_files:
            doc = Document(file)
            full_text = " ".join([p.text for p in doc.paragraphs])
            docs_text[file.name] = full_text

        st.success(f"✅ Loaded {len(uploaded_files)} documents successfully!")

        # Step 4: Question input
        query = st.text_input("💬 Ask a question about your documents:")

        if query:
            # Step 5: Chunk and index all documents
            all_chunks = []
            doc_mapping = []
            for doc_name, text in docs_text.items():
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                all_chunks.extend(chunks)
                doc_mapping.extend([doc_name] * len(chunks))

            # Step 6: Compute embeddings via TF-IDF
            vectorizer = TfidfVectorizer(stop_words="english")
            doc_vectors = vectorizer.fit_transform(all_chunks)
            query_vector = vectorizer.transform([query])

            # Step 7: Find the top matching chunk
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            top_index = similarities.argmax()
            top_chunk = all_chunks[top_index]
            source_doc = doc_mapping[top_index]

            # Step 8: If multiple versions exist (e.g., v1 and v2), prefer the latest one
            doc_names = list(docs_text.keys())
            if len(doc_names) > 1:
                # Sort by version number (v2 > v1)
                latest_doc = sorted(doc_names, key=lambda x: int(''.join(filter(str.isdigit, x.split('_v')[-1].split('.')[0])) or 0))[-1]
                if latest_doc != source_doc:
                    # Replace with latest document chunk if similar content exists
                    for idx, name in enumerate(doc_mapping):
                        if name == latest_doc:
                            if cosine_similarity(doc_vectors[top_index], doc_vectors[idx])[0][0] > 0.85:
                                source_doc = latest_doc
                                top_chunk = all_chunks[idx]
                                break

            # Step 9: Generate a concise final answer from LLM
            try:
                prompt = f"""
                You are an expert summarizer for RAG systems.
                Based on the document content below, answer the user's question in a single, clean paragraph.
                
                User's question: {query}
                Document source: {source_doc}
                Relevant text: {top_chunk}
                """

                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=250,
                    temperature=0.2
                )

                answer = response.choices[0].message["content"].strip()
                st.markdown(f"### 🧠 Answer (from {source_doc}):")
                st.write(answer)

            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
else:
    st.info("Please enter your OpenAI API key to start.")
