# save as test_setup.py and run:  python test_setup.py

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("1. ChromaDB importing... OK")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("2. HuggingFace embeddings loaded... OK")

client = chromadb.Client()
print("3. ChromaDB client created... OK")

print("\n All set! Ready to build your Superman RAG.")