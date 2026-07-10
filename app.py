
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("Missing GROQ_API_KEY. Add it in .env or Streamlit Secrets.")
        st.stop()


# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide"
)


st.title("📄 DocuMind AI")
st.caption(
    "AI-powered PDF assistant using LangChain + Groq + RAG"
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    st.success("LLM: Groq Llama 3.1 8B")

    st.success(
        "Embeddings: HuggingFace MiniLM"
    )

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


if "uploaded" not in st.session_state:
    st.session_state.uploaded = False



# --------------------------------------------------
# Load Models
# --------------------------------------------------

@st.cache_resource
def load_llm():

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=500
    )


@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


llm = load_llm()

embeddings = load_embeddings()



# --------------------------------------------------
# PDF Processing
# --------------------------------------------------

def create_vector_store(pdf_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp:

        temp.write(pdf_file.read())

        file_path = temp.name


    loader = PyPDFLoader(file_path)

    documents = loader.load()


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


    chunks = splitter.split_documents(
        documents
    )


    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )


    os.remove(file_path)


    return vector_store



# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

if not st.session_state.uploaded:


    uploaded_file = st.file_uploader(
        "📂 Upload PDF",
        type=["pdf"]
    )


    if uploaded_file:


        with st.spinner(
            "Processing document..."
        ):

            st.session_state.vector_store = (
                create_vector_store(uploaded_file)
            )


            st.session_state.uploaded = True


        st.success(
            "✅ PDF ready!"
        )

        st.rerun()



# --------------------------------------------------
# Chat Interface
# --------------------------------------------------

if st.session_state.uploaded:


    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    query = st.chat_input(
        "Ask something about your PDF..."
    )


    if query:


        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )


        with st.chat_message("user"):

            st.markdown(query)



        with st.spinner(
            "Searching document..."
        ):

            results = (
                st.session_state.vector_store
                .similarity_search(
                    query,
                    k=5
                )
            )


        context = "\n\n".join(
            [
                f"""
Page {doc.metadata.get('page',0)+1}:

{doc.page_content}
"""
                for doc in results
            ]
        )


        prompt = f"""

You are DocuMind AI.

Answer the user only using the PDF content.

Rules:
- Do not invent information.
- If the answer is unavailable say:
"I couldn't find that information in the uploaded PDF."
- Keep answers clear.
- Mention page numbers when possible.


PDF CONTENT:

{context}


QUESTION:

{query}


ANSWER:

"""


        with st.chat_message("assistant"):


            with st.spinner(
                "Thinking..."
            ):

                response = llm.invoke(
                    prompt
                )

                answer = response.content


            st.markdown(answer)



        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

