
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
import streamlit as st
from time import sleep
llm = ChatOllama(model="llama3.1",temperature=0)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


if "messages" not in st.session_state:
    st.session_state.messages=[]
def document_process(path):

   loader = PyPDFLoader(path)
   docs = loader.load()

   splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
   docs = splitter.split_documents(docs)

   embeddings = OllamaEmbeddings(model="nomic-embed-text")
   vector_db = InMemoryVectorStore.from_documents(documents=docs,embedding=embeddings)
   st.session_state.vector_db=vector_db
   st.session_state.document_uploaded=True
   
   
## documnet upload 
st.title("📄 Local PDF Chatbot")
st.subheader("Your private document assistant, powered by local AI")

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded=False

if not st.session_state.document_uploaded:
    file= st.file_uploader(label="select your PDF file",type="pdf")
    if file:
        with open("uploaded_document.pdf","wb") as f:
            f.write(file.getvalue())
            
        with st.spinner("processing......."):
            document_process("./uploaded_document.pdf")
        st.markdown("documnet processed sucessfully")
        sleep(2)
        st.rerun()
        



if  st.session_state.document_uploaded and  st.session_state.vector_db :
    for oneMessage in st.session_state.messages:
        role = oneMessage["role"]
        content=oneMessage["content"]
        
        st.chat_message(role).markdown(content)
    query = st.chat_input("Ask Anything....")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        documents = st.session_state.vector_db.similarity_search(query,k=2)
        context=""
        
        for doc in documents:
          context +=doc.page_content+ "\n\n"
          
        prompt=f""" You are a helpful assistant and you provide answers for user questions based on the provided conext.
                context:{context} and question is:{query}"""
        result=llm.invoke(prompt)
        st.session_state.messages.append({"role":"ai","content":result.content})
        st.chat_message("ai").markdown(result.content)