import streamlit as st
import pandas as pd
import os
from google import genai
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

st.set_page_config(page_title="RAG Chatbot - Restaurant Reviews", page_icon="🍽️")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ Gemini Client ------------------
client = genai.Client(api_key="AIzaSyA-bjwh-ZfB9kFv9W5pPNPGWvZA8iNO43I")

# ------------------ Load Data & Embedding ------------------
@st.cache_resource
def load_vector_store():
    df = pd.read_csv("realistic_restaurant_reviews.csv")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db_location = "./chroma_langchain_db"
    add_documents = not os.path.exists(db_location)

    documents, ids = [], []
    if add_documents:
        for i, row in df.iterrows():
            content = (
                f"Title: {row['Title']}\n"
                f"Review: {row['Review']}\n"
                f"Rating: {row['Rating']}\n"
                f"Date: {row['Date']}"
            )
            doc = Document(
                page_content=content,
                metadata={"rating": row["Rating"], "date": row["Date"]},
                id=str(i),
            )
            documents.append(doc)
            ids.append(str(i))

    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=db_location,
        embedding_function=embeddings,
    )

    if add_documents:
        vector_store.add_documents(documents=documents, ids=ids)

    return vector_store.as_retriever(search_kwargs={"k": 5})

retriever = load_vector_store()

# ------------------ Streamlit UI ------------------
st.title("🍽️ Restaurant Review Assistant")
st.markdown("Ask any question based on real restaurant reviews!")

st.markdown("## 💬 Chat History")
for i, (q, a) in enumerate(st.session_state.chat_history):
    with st.expander(f"❓ {q}"):
        st.markdown(f"**🤖** {a}")

question = st.chat_input("Enter your question:")

if question:
    st.chat_message("user").markdown(f"**❓ Question:** {question}")
    with st.spinner("Searching and thinking..."):
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = (
            f"You are a helpful assistant. Answer the question based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        response = client.models.generate_content(model="gemini-2.0-flash",contents=prompt,)

        answer = response.text

        st.session_state.chat_history.append((question, answer))
        
        st.markdown("### ✨ Answer")
        st.write(response.text)

        with st.expander("🔍 Show retrieved reviews"):
            for i, doc in enumerate(retrieved_docs):
                st.markdown(f"**Review #{i+1}**")
                st.markdown(doc.page_content.replace("\n", "<br>"), unsafe_allow_html=True)
