import warnings
warnings.filterwarnings('ignore')
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = Chroma(persist_directory='./superman_vectordb', embedding_function=embeddings)

# Get ALL docs and filter by slide number
all_docs = db.get()
docs_with_meta = list(zip(all_docs['documents'], all_docs['metadatas']))

target_slides = [12, 35, 140, 141, 142, 143, 144, 145]

for slide_num in target_slides:
    matching = [(d, m) for d, m in docs_with_meta if m.get('slide') == slide_num]
    if matching:
        for doc, meta in matching:
            print(f"\n--- Slide {slide_num} ({meta.get('source')}) ---")
            print(doc)
    else:
        print(f"\n--- Slide {slide_num}: NOT FOUND IN CHROMADB ---")