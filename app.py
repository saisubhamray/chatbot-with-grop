import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment variables
load_dotenv()


# --------------------------------------------------
# Streamlit Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
)


st.title("📄 DocuMind AI")
st.subheader(
    "Your intelligent PDF assistant — ask questions, extract insights, and chat with your documents."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("⚙️ Settings")

    st.markdown("### Language Model")
    st.success("Groq - Llama 3.1 8B")

    st.markdown("### Embedding Model")
    st.success("Ollama - nomic-embed-text")

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []
        st.rerun()


    st.divider()

    st.caption(
        "Built with LangChain + Groq + Ollama Embeddings + Streamlit"
    )



# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False



# --------------------------------------------------
# Load Groq Model
# --------------------------------------------------

@st.cache_resource
def load_llm():

    return ChatGroq(

        model="llama-3.1-8b-instant",

        temperature=0,

        max_tokens=250,
    )



# --------------------------------------------------
# Load Embeddings
# --------------------------------------------------

@st.cache_resource
def load_embeddings():

    return OllamaEmbeddings(
        model="nomic-embed-text"
    )



llm = load_llm()

embeddings = load_embeddings()



# --------------------------------------------------
# Process PDF
# --------------------------------------------------

def process_document(file_path):

    loader = PyPDFLoader(file_path)

    documents = loader.load()


    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )


    chunks = splitter.split_documents(documents)



    vector_db = InMemoryVectorStore.from_documents(

        documents=chunks,

        embedding=embeddings

    )


    st.session_state.vector_db = vector_db

    st.session_state.document_uploaded = True



# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

if not st.session_state.document_uploaded:


    uploaded_file = st.file_uploader(

        "📂 Upload your PDF document",

        type=["pdf"]

    )


    if uploaded_file:


        with tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".pdf"

        ) as temp:


            temp.write(uploaded_file.read())

            pdf_path = temp.name



        with st.spinner(
            "Reading and understanding your document..."
        ):

            process_document(pdf_path)



        os.remove(pdf_path)



        st.success(
            "✅ Document processed successfully!"
        )


        st.rerun()



# --------------------------------------------------
# Chat
# --------------------------------------------------

if st.session_state.document_uploaded:


    for message in st.session_state.messages:


        with st.chat_message(message["role"]):

            st.markdown(message["content"])



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
            "Searching your document..."
        ):


            docs = st.session_state.vector_db.similarity_search(

                query,

                k=4

            )



        context = "\n\n".join(

            doc.page_content

            for doc in docs

        )



        prompt = f"""

You are DocuMind AI, a helpful PDF assistant.

Conversation style:
- Friendly and natural.
- Talk like a human assistant.
- Keep responses clear and simple.
- Do not sound robotic.

Rules:
1. Answer only from the provided document.
2. If information is missing, say:
"I couldn't find that information in the uploaded PDF."
3. Never invent facts.
4. Use bullet points when useful.
5. Give short explanations first, then details if needed.


Document:

{context}


User question:

{query}


Answer:

"""



        with st.chat_message("assistant"):


            with st.spinner(
                "Thinking..."
            ):


                result = llm.invoke(prompt)


                answer = result.content



            st.markdown(answer)



        st.session_state.messages.append(

            {
                "role": "assistant",

                "content": answer

            }

        )