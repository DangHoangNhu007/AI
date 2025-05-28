from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")
embaddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documnet = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content= row["Title"] + " " + row["Review"],
            metadata = {"rating": row["Rating"], "date": row["Date"]},
            id= str(i)
            )
        ids.append(str(i))
        documnet.append(document)

vector_store = Chroma(collection_name="restaurant_reviews",
                      persist_directory=db_location,
                      embedding_function=embaddings,)
if add_documents:
    vector_store.add_documents(documents=documnet, ids=ids)

retriever = vector_store.as_retriever(searcjh_kwargs={"k": 5})
