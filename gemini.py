from google import genai
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

#1-----------------Start Gemni Client-------------------
# Make sure to install the google-genai package

client = genai.Client(api_key="AIzaSyA-bjwh-ZfB9kFv9W5pPNPGWvZA8iNO43I")

#2-----------------Load Embeddings and Vector Store-------------------

df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
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
            id=str(i)
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

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

#3-----------------Start Chat Loop-------------------

while True:
    print("\n\n--------------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question.lower() == 'q':
        break
    
    #4 -----------------Retrieve Relevant Reviews-------------------
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    #5 -----------------Generate Response with Gemini-------------------
    promt = f"""You are a helpful assistant. Answer the question based on the context provided. Here are some relevant reviews: {context}. Here is the question to answer: {question}"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=promt,
    )
    
    print(response.text)