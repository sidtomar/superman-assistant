import os
import warnings
warnings.filterwarnings("ignore")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTOR_DB_FOLDER = "./superman_vectordb"

def load_vectordb():
    print("Loading ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory=VECTOR_DB_FOLDER,
        embedding_function=embeddings
    )
    print(f"ChromaDB loaded. Vectors available: {vectordb._collection.count()}")
    return vectordb

def build_chain(vectordb):
    print("Building Q&A chain...")

    prompt_template = PromptTemplate(
        template="""
You are a Superman CRM product assistant for Mankind Pharma.
Use the context below from Superman documentation to answer the question.
Piece together information from multiple chunks if needed.
Give a complete answer using all relevant information found in the context.
If the answer is genuinely not present in context, say:
"I don't have that information in the Superman documentation."
Do NOT make up answers not supported by the context.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print("Q&A chain ready!\n")
    return chain

def ask(chain, question):
    print(f"\n❓ Question: {question}")
    print("-" * 60)
    result = chain.invoke(question)
    print(f"✅ Answer: {result}")
    print("-" * 60)

def debug_retrieval(vectordb, question):
    print(f"\n🔍 DEBUG: '{question}'")
    print("=" * 60)
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(question)
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1} | Slide {doc.metadata.get('slide', '?')}:")
        print(doc.page_content)
    print("=" * 60)

def main():
    vectordb = load_vectordb()
    chain = build_chain(vectordb)

    # Debug failing questions first
    debug_retrieval(vectordb, "What is the MTP submission window?")
    debug_retrieval(vectordb, "How does a MR submit their DCR?")

    # Full question set
    questions = [
        "Where are the modules located on the Superman home screen?",
        "What modules are visible in the navigation menu on homepage?",
        "What are the core modules in Superman?",
        "How does MTP approval work?",
        "What is the MTP submission window?",
        "How does a MR submit their DCR?",
        "What is NMNE in chemist journey?",
        "What is the new App Setting feature on home screen?"
    ]

    for q in questions:
        ask(chain, q)

if __name__ == "__main__":
    main()