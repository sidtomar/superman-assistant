DOCS_FOLDER = "./superman_docs"
VECTOR_DB_FOLDER = "./superman_vectordb"
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)