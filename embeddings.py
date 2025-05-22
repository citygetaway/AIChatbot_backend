# This script stores and retrieves embeddings using ChromaDB
# 54 chunks are created from the knowledge.docx file
# and stored in the chromadb_store directory in chroma.sqlite3.
# --- backend/embeddings.py ---

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from docx import Document
from pymongo import MongoClient
import numpy as np
import chromadb
import os

# Load document and chunk
DOC_PATH = 'knowledge.docx'
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
print(f"MongoDB URI: {MONGO_URI}")
DB_NAME = "embeddings_db"
CHROMA_DIR = 'chromadb_store'
COLLECTION_NAME = 'montreal_tourism'

api_key = os.getenv("OPENAI_API_KEY")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
openai_embedding_fn = OpenAIEmbeddingFunction(api_key)
collection = db[COLLECTION_NAME]

# print(f"📄 Collection contains {collection.count()} documents.")

# Run once to populate ChromaDB
if collection.count_documents({}) == 0:
    doc = Document(DOC_PATH)
    paragraphs = [p.text for p in doc.paragraphs if len(p.text.strip()) > 50]
    for i, chunk in enumerate(paragraphs):
        embedding = openai_embedding_fn([chunk])[0].tolist()
        collection.insert_one({
            "id": f"chunk_{i}",
            "embedding": embedding,
            "text": chunk
        })

# Retrieval function
def get_relevant_passages(query, n_results=54):
    query_embedding = openai_embedding_fn([query])[0]
    documents = list(collection.find({}))

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


    results = [
        (doc["text"], cosine_similarity(query_embedding, np.array(doc["embedding"])))
        for doc in documents
    ]
    results = sorted(results, key=lambda x: x[1], reverse=True)[:n_results]
    return "\n".join([r[0] for r in results])