# This script stores and retrieves embeddings using ChromaDB
# 54 chunks are created from the knowledge.docx file
# and stored in the chromadb_store directory in chroma.sqlite3.
# --- backend/embeddings.py ---

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from docx import Document
import chromadb
import os

# Load document and chunk
DOC_PATH = 'knowledge.docx'
CHROMA_DIR = 'chromadb_store'
COLLECTION_NAME = 'montreal_tourism'

api_key = os.getenv("OPENAI_API_KEY")
client = chromadb.PersistentClient(path=CHROMA_DIR)
openai_embedding_fn = OpenAIEmbeddingFunction(api_key)
collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=openai_embedding_fn)

# print(f"📄 Collection contains {collection.count()} documents.")

# Run once to populate ChromaDB
if not collection.count():
    doc = Document(DOC_PATH)
    paragraphs = [p.text for p in doc.paragraphs if len(p.text.strip()) > 50]
    for i, chunk in enumerate(paragraphs):
        collection.add(documents=[chunk], ids=[f"chunk_{i}"])

# Retrieval function
def get_relevant_passages(query, n_results=54):
    results = collection.query(query_texts=[query], n_results=n_results)
    return "\n".join(results['documents'][0])