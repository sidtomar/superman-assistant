import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTOR_DB_FOLDER = "./superman_vectordb"

st.set_page_config(
    page_title="Superman Assistant",
    page_icon="🦸",
    layout="centered"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("## 🦸")
with col2:
    st.markdown("## Superman Assistant")
    st.caption("Ask anything about the Superman CRM platform")

st.divider()

@st.cache_resource(show_spinner="Loading Superman knowledge base...")
def load_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory=VECTOR_DB_FOLDER,
        embedding_function=embeddings
    )
    prompt_template = PromptTemplate(
        template="""
You are a Superman CRM product assistant for Mankind Pharma.
Use the context below from Superman documentation to answer the question.
Piece together information from multiple chunks if needed.
Give a complete and helpful answer using all relevant information found in context.
If the answer is genuinely not present in the context, say:
"I don't have that information in the Superman documentation. Please refer to your Superman admin or training team."
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
    return chain, vectordb

def get_sources(vectordb, question):
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(question)
    slides = sorted(set(
        doc.metadata.get("slide", "?")
        for doc in docs
        if doc.metadata.get("slide")
    ))
    sources = sorted(set(
        doc.metadata.get("source", "")
        for doc in docs
        if doc.metadata.get("source")
    ))
    return slides, sources

SUGGESTED = [
    "What are the core modules in Superman?",
    "How does MTP approval work?",
    "How does a MR submit their DCR?",
    "What is NMNE in chemist journey?",
    "How does E-detailing work?",
    "What is the MTP submission window?",
    "How does expense submission work?",
    "What is AMS in Superman?"
]

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
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"][0]:
                with st.expander("📄 Source slides used"):
                    st.caption(f"**Slides:** {', '.join(str(s) for s in message['sources'][0])}")
                    st.caption(f"**Document:** {', '.join(message['sources'][1])}")

if prompt := st.chat_input("Ask anything about Superman..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching Superman docs..."):
            try:
                response = chain.invoke(prompt)
                slides, sources = get_sources(vectordb, prompt)
            except Exception as e:
                response = f"Something went wrong: {e}"
                slides, sources = [], []
        st.markdown(response)
        if slides:
            with st.expander("📄 Source slides used"):
                st.caption(f"**Slides:** {', '.join(str(s) for s in slides)}")
                st.caption(f"**Document:** {', '.join(sources)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": (slides, sources)
    })

with st.sidebar:
    st.markdown("### 🦸 Superman Assistant")
    st.caption("Powered by Mankind Pharma")
    st.divider()
    st.markdown("**Knowledge Base**")
    st.success(f"✅ {vectordb._collection.count()} vectors indexed")
    st.divider()
    st.markdown("**Documents Loaded**")
    if os.path.exists("./superman_docs"):
        files = [f for f in os.listdir("./superman_docs")
                 if f.endswith(('.pptx', '.pdf', '.docx'))]
        for f in files:
            st.caption(f"📄 {f}")
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("For issues contact: IT / Digital Transformation team")