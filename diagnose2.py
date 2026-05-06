import warnings
warnings.filterwarnings('ignore')
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = Chroma(persist_directory='./superman_vectordb', embedding_function=embeddings)

queries = [
    'MTP submission window 15th 25th monthly',
    'DCR submit steps daily call reporting',
    'App Setting home screen',
    'modules available Superman application'
]

for q in queries:
    print(f'\nQUERY: {q}')
    results = db.similarity_search(q, k=10)
    slides_found = [str(r.metadata["slide"]) for r in results]
    print(f'  Slides found: {", ".join(slides_found)}')
    for r in results:
        if r.metadata["slide"] in [12, 35, 140, 141, 142, 143, 144, 145]:
            print(f'  >>> RELEVANT Slide {r.metadata["slide"]}: {r.page_content[:200]}')