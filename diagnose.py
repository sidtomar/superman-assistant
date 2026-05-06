import warnings
warnings.filterwarnings('ignore')
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = Chroma(persist_directory='./superman_vectordb', embedding_function=embeddings)

queries = [
    'modules home screen navigation',
    'MTP submission window 15th 25th',
    'DCR submit daily call reporting steps',
    'App Setting home screen icon'
]

for q in queries:
    print(f'\nQUERY: {q}')
    results = db.similarity_search(q, k=3)
    for r in results:
        print(f'  Slide {r.metadata["slide"]}: {r.page_content[:150]}')