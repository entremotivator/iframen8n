import streamlit as st
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain.chains import RetrievalQA
import nltk
from nltk.tokenize import sent_tokenize
import os
import tempfile
import logging
from typing import List, Dict, Any

# Cloud API Configuration
DEFAULT_API_URL = "https://theaisource-u29564.vm.elestio.app:57987"
DEFAULT_USERNAME = "root"
DEFAULT_PASSWORD = "eZfLK3X4-SX0i-UmgUBe6E"

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=Warning)

# Download NLTK data at startup
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.warning(f"Failed to download NLTK punkt: {e}")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs):
        try:
            self.text += token
            clean_text = self.text
            
            if "AIMessage" in clean_text:
                if "content=\"" in clean_text:
                    try:
                        clean_text = clean_text.split("content=\"")[1].rsplit("\"", 1)[0]
                    except IndexError:
                        pass
                
                clean_text = (clean_text.replace("AIMessage(", "")
                                      .replace(", additional_kwargs={}", "")
                                      .replace(", response_metadata={})", "")
                                      .replace('{ "data":' , "")
                                      .replace('}' , ""))
            
            self.container.markdown(clean_text)
            
        except Exception as e:
            logging.warning(f"StreamHandler error: {str(e)}")
            self.container.markdown(self.text)

class RAGChat:
    EMBEDDING_MODELS = {
        "bge-small": {
            "name": "BAAI/bge-small-en-v1.5",
            "type": "huggingface",
            "description": "Optimized for retrieval tasks, good balance of speed/quality"
        },
        "bge-large": {
            "name": "BAAI/bge-large-en-v1.5",
            "type": "huggingface",
            "description": "Highest quality, but slower and more resource intensive"
        },
        "minilm": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "type": "huggingface",
            "description": "Lightweight, fast, good general purpose model"
        },
        "mpnet": {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "type": "huggingface",
            "description": "Higher quality, slower than MiniLM"
        },
        "e5-small": {
            "name": "intfloat/e5-small-v2",
            "type": "huggingface",
            "description": "Efficient model optimized for semantic search"
        },
        "snowflake-arctic-embed2:568m": {
            "name": "snowflake-arctic-embed2:568m",
            "type": "ollama",
            "description": "Multilingual frontier model with strong performance [Ollama Embedding Model]"
        }
    }

    def __init__(self):
        self.vectorstore = None
        self.headers = {
            "Authorization": f"Basic {DEFAULT_USERNAME}:{DEFAULT_PASSWORD}"
        }
    
    def _create_sentence_windows(self, text: str, window_size: int = 4) -> List[str]:
        sentences = sent_tokenize(text)
        windows = []
        
        for i in range(len(sentences)):
            start = max(0, i - window_size)
            end = min(len(sentences), i + window_size + 1)
            window = sentences[start:end]
            windows.append(" ".join(window))
        
        return windows

    def process_pdfs(self, pdf_files, embedding_model: str = "bge-small") -> int:
        try:
            all_windows = []
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for pdf_file in pdf_files:
                    temp_path = os.path.join(temp_dir, pdf_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    
                    loader = PyPDFLoader(temp_path)
                    documents = loader.load()
                    
                    for doc in documents:
                        windows = self._create_sentence_windows(doc.page_content)
                        all_windows.extend([Document(page_content=window) for window in windows])
            
            model_config = self.EMBEDDING_MODELS[embedding_model]
            if model_config["type"] == "ollama":
                embeddings = OllamaEmbeddings(
                    model=model_config["name"],
                    base_url=DEFAULT_API_URL,
                    headers=self.headers
                )
            else:
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_config["name"],
                    encode_kwargs={'normalize_embeddings': True}
                )
            
            self.vectorstore = FAISS.from_documents(documents=all_windows, embedding=embeddings)
            return len(all_windows)
            
        except Exception as e:
            logging.error(f"PDF processing error: {str(e)}")
            raise

    def get_retrieval_chain(self, ollama_model: str, stream_handler=None):
        llm = Ollama(
            model=ollama_model,
            temperature=0.2,
            base_url=DEFAULT_API_URL,
            headers=self.headers,
            callbacks=[stream_handler] if stream_handler else None
        )
        
        template = """
        Context: {context}
        Question: {question}
        
        Provide a detailed, well-structured answer based only on the above context.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 7, "fetch_k": 20}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

def get_ollama_models() -> List[str]:
    try:
        headers = {
            "Authorization": f"Basic {DEFAULT_USERNAME}:{DEFAULT_PASSWORD}"
        }
        response = requests.get(f"{DEFAULT_API_URL}/api/tags", headers=headers)
        
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models['models']
                    if all(keyword not in model['name'].lower()
                        for keyword in ('failed', 'embed', 'bge'))]
        return []
    except Exception as e:
        logging.error(f"Error fetching models: {str(e)}")
        return []

def init_session_state():
    session_vars = {
        'messages': [],
        'rag_chain': None,
        'rag_system': RAGChat(),
        'show_chat': False,
        'last_experiment_name': None,
        'processed_files': False,
        'previous_model': None,
        'previous_embedding': None,
        'previous_files': None,
        'process_ready': False,
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

def render_header():
    st.markdown('''
    <div class="header-container">
        <p class="header-subtitle">🔍 Cloud RAG - Powered PDF Chat Assistant</p>
    </div>
    ''', unsafe_allow_html=True)

def setup_model_selection():
    models = get_ollama_models()
    if not models:
        st.warning("Cannot connect to Ollama Cloud API. Please check your connection and credentials.")
        return None, None, None
    
    col1, col2 = st.columns(2)
    with col1:
        embedding_model = st.selectbox(
            "Select Embedding Model:",
            list(RAGChat.EMBEDDING_MODELS.keys()),
            format_func=lambda x: f"{x} - {RAGChat.EMBEDDING_MODELS[x]['description']}"
        )
            
        llm_model = st.selectbox(
            "Select Language Model:",
            models,
            format_func=lambda x: f'🔮 {x}'
        )
    
    with col2:
        st.markdown("##### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files:",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select PDF files to analyze"
        )

        if uploaded_files:
            st.markdown(f"*{len(uploaded_files)} files selected*")

    # Check for changes in configuration
    if (st.session_state.previous_model != llm_model or 
        st.session_state.previous_embedding != embedding_model or 
        st.session_state.previous_files != uploaded_files):
        st.session_state.process_ready = False
        st.session_state.show_chat = False
        st.session_state.processing_completed = False
    
    # Update previous states
    st.session_state.previous_model = llm_model
    st.session_state.previous_embedding = embedding_model
    st.session_state.previous_files = uploaded_files

    return uploaded_files, embedding_model, llm_model

def process_documents(experiment_name: str, uploaded_files: List[Any], embedding_model: str) -> bool:
    if not experiment_name:
        st.error("Please enter an experiment name")
        return False
    
    if not uploaded_files:
        st.error("Please upload PDF files first")
        return False
        
    with st.spinner("📚 Processing documents..."):
        try:
            st.session_state.rag_system.process_pdfs(
                uploaded_files,
                embedding_model=embedding_model
            )
            st.session_state.show_chat = True
            st.session_state.messages = []
            st.session_state.last_experiment_name = experiment_name
            st.session_state.processed_files = True
            return True
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            return False

def handle_chat_interaction(llm_model: str):
    chat_container = st.container()
    prompt = st.chat_input("Ask about your documents")

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
        if prompt and st.session_state.rag_system.vectorstore:
            if st.session_state.rag_system.vectorstore:
                process_chat_message(prompt, llm_model)
            else:
                st.warning("Please process documents first!")

def process_chat_message(prompt: str, llm_model: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)
        
        try:
            retrieval_chain = st.session_state.rag_system.get_retrieval_chain(
                llm_model,
                stream_handler=stream_handler
            )
            
            response = retrieval_chain.invoke({
                "query": prompt
            })
            
            final_response = response["result"].strip()
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response
            })
            response_placeholder.markdown(final_response)
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

def run():
    init_session_state()   
    render_header()
    
    st.markdown("#### 📋 Configuration")
    
    uploaded_files, embedding_model, llm_model = setup_model_selection()
    if not llm_model:
        return
    
    experiment_name = st.text_input(
        "📝 Name your Experiment",
        key="experiment_name",
        placeholder="Enter experiment name",
    )
    
    process_ready = st.checkbox("Start RAG Analysis", 
                              key="process_ready", 
                              value=st.session_state.process_ready)
    
    if process_ready:
        if 'processing_completed' not in st.session_state:
            st.session_state.processing_completed = False
            
        if not st.session_state.processing_completed:
            st.session_state.processing_completed = process_documents(
                experiment_name, uploaded_files, embedding_model
            )
    else:
        st.session_state.processing_completed = False
        st.session_state.show_chat = False
    
    if st.session_state.show_chat:
        handle_chat_interaction(llm_model)

if __name__ == "__main__":
    run()
