import streamlit as st
from config import Config
from dotenv import load_dotenv
from helpers.llm_helper import chat, stream_parser, get_relevant_context
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import os
import pickle

load_dotenv()

openai_api_key = Config.OPENAI_API_KEY

st.set_page_config(
    page_title="Audie's OpenAI Chatbot",
    initial_sidebar_state='expanded'
)

st.title("Welcome to Audie's OpenAI Chatbot")

# Initialize session state for messages and vector store
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.markdown("# Chat Options")

    # Widgets
    model = st.selectbox("What model would you like to use",
                         ('gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo','gpt-4o'))
    temperature = st.number_input("Temperature", value=0.7,
                                  min_value=0.0, max_value=1.0, step=0.1)
    max_token_length = st.number_input("Max Token Length", value=1000,
                                  min_value=100, max_value=128000)
    
    st.markdown('# Upload PDF File')
    pdf = st.file_uploader("**Upload your PDF**", type='pdf')
    if pdf:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        if not chunks:
            st.error("No extractable text found in the PDF.")
        file_name = pdf.name[:-4]
        if os.path.exists(f"{file_name}.pkl"):
            with open(f"{file_name}.pkl", "rb") as f:
                st.session_state.vector_store = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            st.session_state.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{file_name}.pkl", "wb") as f:
                pickle.dump(st.session_state.vector_store, f)

# Display the chat messages stored in session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("What questions do you have?"):
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.spinner("Generating response ..."):
        # Fetch relevant context from past messages
        past_messages = '\n'.join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        combined_prompt = f"{past_messages}\nuser: {user_prompt}"
        
        if st.session_state.vector_store:
            llm_response = chat(combined_prompt, model=model, vector_store=st.session_state.vector_store, max_tokens=max_token_length, temp=temperature)
        else:
            llm_response = chat(combined_prompt, model=model, max_tokens=max_token_length, temp=temperature)
        
        response_content = ''.join([content for content in stream_parser(llm_response)])
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        with st.chat_message("assistant"):
            st.markdown(response_content)
