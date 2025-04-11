import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time
import uuid

# Sidebar Settings
st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")
hf_token = st.sidebar.text_input("Enter your HF Token", type="password")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
else:
    st.sidebar.warning("Please enter your GROQ API Key.")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Upload PDF in Main Area
st.title("📄 Upload Your PDF")
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

def create_vector_embedding():
    """Creates vector embeddings from uploaded PDF."""
    if pdf_file and "vectors" not in st.session_state:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())
        
        st.session_state.loader = PyPDFLoader("temp.pdf")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, embeddings)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.chat_history[st.session_state.current_chat_id] = []

def new_chat():
    """Starts a new chat session and stores previous chats."""
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.chat_history[st.session_state.current_chat_id] = []

def clear_chat():
    """Clears the chat history of the current session."""
    st.session_state.chat_history[st.session_state.current_chat_id] = []

if st.button("🔍 Create Document Embeddings"):
    create_vector_embedding()
    st.success("Vector database is ready!")

st.sidebar.button("🆕 New Chat", on_click=new_chat)
st.sidebar.button("🗑️ Clear Chat", on_click=clear_chat)

st.title("💬 Chat with your PDF")
st.markdown("Ask questions based on your uploaded PDF!")

for message in st.session_state.chat_history[st.session_state.current_chat_id]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question...")
if user_input:
    st.session_state.chat_history[st.session_state.current_chat_id].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    if "vectors" not in st.session_state:
        create_vector_embedding()
    
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': user_input})
        response_text = response.get('answer', "I'm sorry, I couldn't find relevant information.")
        print(f"Response time: {time.process_time() - start_time:.2f} seconds")
        
        st.session_state.chat_history[st.session_state.current_chat_id].append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
        
        with st.expander("📄 Relevant Document Snippets"):
            for i, doc in enumerate(response.get('context', [])):
                st.write(doc.page_content)
                st.write('------------------------')