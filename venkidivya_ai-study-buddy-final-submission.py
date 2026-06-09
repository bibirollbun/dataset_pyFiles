# Force upgrade google-generativeai to fix the conflict
!pip install -U -q google-generativeai google-ai-generativelanguage

# Install your project dependencies
!pip install -q streamlit pypdf langchain==0.3.27 langchain-community==0.3.31 \
langchain-core==0.3.80 langchain-google-genai==2.1.12 \
langchain-text-splitters==0.3.11 langchain-huggingface==0.3.0 \
sentence-transformers faiss-cpu


import langchain
import google.generativeai as genai
print("Libraries loaded successfully!")


import os
from kaggle_secrets import UserSecretsClient

# âœ… Securely access the Google API Key from Kaggle Secrets
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Google API key setup complete.")
except Exception as e:
    print(f"ðŸ”‘ Auth Error: Please add 'GOOGLE_API_KEY' to Kaggle Secrets. Details: {e}")


%%writefile main.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# 1. SETUP THE PAGE
st.set_page_config(page_title="My Personal Study Assistant", layout="wide")
st.header("ðŸ“š AI Study Buddy (Powered by Gemini)")

# 2. SIDEBAR: CONFIGURATION
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter Google API Key:", type="password")
    
    # Manual Model Selector (Safe Mode)
    chat_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
    selected_model = st.selectbox("Choose AI Model:", chat_models, index=0)

    uploaded_file = st.file_uploader("Upload your Study Material (PDF)", type="pdf")

# 3. INITIALIZE SESSION STATE (MEMORY)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. MAIN LOGIC
if api_key and uploaded_file:
    try:
        # Process PDF only once (using session state to avoid reloading)
        if "vector_store" not in st.session_state:
            with st.spinner("Analyzing PDF... (Downloading Local Embeddings)"):
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # Semantic Chunking
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=text)

                # Local Embeddings (Privacy-First & Free)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                st.success("PDF Loaded & Indexed Successfully!")

        # 5. DISPLAY CHAT HISTORY
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 6. CHAT INPUT
        if prompt := st.chat_input("Ask a question about your notes..."):
            
            # User Message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # AI Generation
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieval
                    docs = st.session_state.vector_store.similarity_search(prompt)
                    
                    # Generation
                    llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=api_key)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    response = chain.run(input_documents=docs, question=prompt)
                    st.markdown(response)
            
            # Save AI Response
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {e}")

elif not api_key:
    st.warning("Please enter your Google API Key in the sidebar to start.")

