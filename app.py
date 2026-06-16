import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

CORPUS_PATH = 'docs/'

@st.cache_resource
def build_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 6, 'fetch_k': 15})
    llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.1, max_tokens=512)
    return retriever, llm

def ask(question, retriever, llm):
    OOS_PROMPT = ChatPromptTemplate.from_template(
        'Does this question relate to HR topics like leave, payroll, benefits, WFH, '
        'code of conduct, performance, compensation, IT security, POSH, onboarding, '
        'travel expenses, or company profile? Reply YES or NO only.\n\nQuestion: {question}'
    )
    verdict = (OOS_PROMPT | llm | StrOutputParser()).invoke({'question': question}).strip().upper()
    if 'YES' not in verdict:
        return 'I can only answer HR-related questions based on Zyro Dynamics policy documents.'
    docs = retriever.invoke(question)
    context = '\n\n'.join(d.page_content for d in docs)
    RAG_PROMPT = ChatPromptTemplate.from_template(
        'You are an HR assistant for Zyro Dynamics. Answer using ONLY the context below.\n'
        'The context may contain tables - read them carefully and give exact numbers where available.\n'
        'If not found, say: I dont have that information in the HR policy documents.\n\n'
        'Context:\n{context}\n\nQuestion: {question}\n\nAnswer:'
    )
    return (RAG_PROMPT | llm | StrOutputParser()).invoke({'context': context, 'question': question})

st.set_page_config(page_title='Zyro Dynamics HR Help Desk', page_icon='ðŸ¢')
st.title('ðŸ¢ Zyro Dynamics HR Help Desk')
st.caption('Ask me anything about HR policies.')

retriever, llm = build_pipeline()

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input('Ask an HR question...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    st.chat_message('user').write(prompt)
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            answer = ask(prompt, retriever, llm)
        st.write(answer)
    st.session_state.messages.append({'role': 'assistant', 'content': answer})
