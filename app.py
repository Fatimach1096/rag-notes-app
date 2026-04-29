import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

load_dotenv()

st.title("📚 University Notes Q&A App")
st.write("Ask any question from your study material!")

@st.cache_resource
def load_rag_pipeline():
    loader = PyPDFLoader("notes.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

with st.spinner("Loading your notes..."):
    qa_chain = load_rag_pipeline()

st.success("✅ Notes loaded! Ask your question below.")

question = st.text_input("🔍 Enter your question:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
            st.write("### 💡 Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question first!")