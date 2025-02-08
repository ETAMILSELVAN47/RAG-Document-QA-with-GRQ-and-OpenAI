from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import time

from dotenv import load_dotenv
load_dotenv()

import openai

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model='llama3-8b-8192')

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only:
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>

    Question:{input}

    """
)



def create_vector_embedding():
    if 'vectors' not in st.session_state:
        # 1.Data
        st.session_state.loader=PyPDFDirectoryLoader(path='research_papers')
        st.session_state.docs=st.session_state.loader.load()
        # 2.Data --> text chunks
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_doc=st.session_state.text_splitter.split_documents(documents=st.session_state.docs[:50])
        # 3.Text --> Vectors and Store vectors into vectorstoreDB
        st.session_state.embedding=OpenAIEmbeddings(model="text-embedding-3-large")
        st.session_state.vectors=FAISS.from_documents(documents=st.session_state.final_doc,embedding=st.session_state.embedding)

st.title('RAG Document Q&A with GROQ and OpenAI')

user_input=st.text_input('Enter your query from the research paper')

if st.button(label='Document Embedding'):
    create_vector_embedding()
    st.write('Vector Database is ready')



if user_input:
    retriever=st.session_state.vectors.as_retriever()
    document_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
    rag_chain=create_retrieval_chain(retriever=retriever,combine_docs_chain=document_chain)

    start=time.process_time()
    response=rag_chain.invoke({'input':user_input})
    end=time.process_time()
    st.write(response.get('answer'))

    with st.expander('Document Similarity Search'):
        for i,doc in enumerate(response.get('context')):
            st.write(doc.page_content)
            st.write('-------------')
    








