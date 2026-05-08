import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
VECTOR_DB_FOLDER = "./superman_vectordb"
DOCS_FOLDER = "./superman_docs"
st.set_page_config(page_title="Superman Assistant", page_icon="S", layout="centered")
st.markdown("<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("## S")
with col2:
    st.markdown("## Superman Assistant")
    st.caption("Ask anything about the Superman CRM platform")
st.divider()
@st.cache_resource(show_spinner="Loading Superman knowledge base...")
def load_chain():
    needs_build = (not os.path.exists(VECTOR_DB_FOLDER)) or (len(os.listdir(VECTOR_DB_FOLDER)) == 0)
    if needs_build:
        import importlib
        svdb = importlib.import_module("setup_vectordb")
        svdb.build_vectordb()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=VECTOR_DB_FOLDER, embedding_function=embeddings)
    prompt_template = PromptTemplate(
        template="You are a Superman CRM product assistant for Mankind Pharma.\nUse the context below to answer the question.\nIf not found in context say: I don't have that information in the Superman documentation.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
        input_variables=["context", "question"]
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt_template | llm | StrOutputParser())
    return chain, vectordb
def get_sources(vectordb, question):
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(question)
    slides = sorted(set(doc.metadata.get("slide", "?") for doc in docs if doc.metadata.get("slide")))
    sources = sorted(set(doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")))
    return slides, sources
SUGGESTED = ["What are the core modules in Superman?", "How does MTP approval work?", "How does a MR submit their DCR?", "What is NMNE in chemist journey?", "How does E-detailing work?", "What is the MTP submission window?", "How does expense submission work?", "What is AMS in Superman?"]
if "messages" not in st.session_state:
    st.session_state.messages = []
try:
    chain, vectordb = load_chain()
except Exception as e:
    st.error(f"Failed to load knowledge base: {e}")
    st.stop()
if not st.session_state.messages:
    st.markdown("#### Try asking:")
    cols = st.columns(2)
    for i, question in enumerate(SUGGESTED):
        with cols[i % 2]:
            if st.button(question, key=f"suggest_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"][0]:
                with st.expander("Source slides used"):
                    st.caption(f"Slides: {', '.join(str(s) for s in message['sources'][0])}")
                    st.caption(f"Document: {', '.join(message['sources'][1])}")
if prompt := st.chat_input("Ask anything about Superman..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
		with st.spinner("Searching Superman docs..."):
            try:
                # DEBUG: Show what chunks are being passed to GPT-4o
                retriever_dbg = vectordb.as_retriever(search_kwargs={"k": 10})
                debug_docs = retriever_dbg.invoke(prompt)
                
                with st.expander("🔍 DEBUG: Chunks sent to GPT-4o"):
                    for i, d in enumerate(debug_docs):
                        st.markdown(f"**Chunk {i+1} | Slide {d.metadata.get('slide','?')} | {d.metadata.get('source','?')}**")
                        st.code(d.page_content[:300])
                
                response = chain.invoke(prompt)
                slides, sources = get_sources(vectordb, prompt)
            except Exception as e:
                response = f"ERROR: {type(e).__name__}: {e}"
                slides, sources = [], []
                st.error(f"Full error: {e}")		
				
        st.markdown(response)
        if slides:
            with st.expander("Source slides used"):
                st.caption(f"Slides: {', '.join(str(s) for s in slides)}")
                st.caption(f"Document: {', '.join(sources)}")
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": (slides, sources)})
with st.sidebar:
    st.markdown("### Superman Assistant")
    st.caption("Powered by Mankind Pharma")
    st.divider()
    st.markdown("**Knowledge Base**")
    st.success(f"Vectors indexed: {vectordb._collection.count()}")
    st.divider()
    st.markdown("**Documents Loaded**")
    if os.path.exists(DOCS_FOLDER):
        files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith((".pptx", ".pdf", ".docx"))]
        for f in files:
            st.caption(f"- {f}")
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("For issues contact: IT / Digital Transformation team")
