import streamlit as st
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
import json
from datetime import datetime
import time
import re
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import io
import PyPDF2
from PIL import Image
import base64
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
from pathlib import Path
import cv2
import pytesseract
import hashlib
from collections import defaultdict
import difflib
from bs4 import BeautifulSoup
import urllib.parse
import aiohttp
import asyncio
import feedparser
from newspaper import Article
from fake_useragent import UserAgent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import yake
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebSearchEngine:
    """Class to handle web searches and content extraction"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.session = aiohttp.ClientSession(trust_env=True)
        self.search_cache = {}
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
    async def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Perform web search using multiple search engines"""
        cache_key = f"{query}_{num_results}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
            
        results = []
        search_engines = [
            self._search_google,
            self._search_bing,
            self._search_duckduckgo
        ]
        
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(engine(query, num_results))
                for engine in search_engines
            ]
            
        all_results = []
        for task in tasks:
            try:
                all_results.extend(await task)
            except Exception as e:
                logger.error(f"Search engine error: {str(e)}")
                
        # Deduplicate and rank results
        results = self._process_search_results(all_results)
        self.search_cache[cache_key] = results[:num_results]
        return self.search_cache[cache_key]
        
    async def _search_google(self, query: str, num_results: int) -> List[Dict]:
        """Search using Google Custom Search API"""
        # Implementation for Google search
        pass
        
    async def _search_bing(self, query: str, num_results: int) -> List[Dict]:
        """Search using Bing Web Search API"""
        # Implementation for Bing search
        pass
        
    async def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict]:
        """Search using DuckDuckGo"""
        # Implementation for DuckDuckGo search
        pass
        
    async def extract_content(self, url: str) -> Dict:
        """Extract and process content from a URL"""
        try:
            article = Article(url)
            await self._async_download(article)
            article.parse()
            article.nlp()
            
            content = {
                'title': article.title,
                'text': article.text,
                'summary': article.summary,
                'keywords': article.keywords,
                'authors': article.authors,
                'publish_date': article.publish_date,
                'top_image': article.top_image,
                'metadata': self._extract_metadata(article)
            }
            
            # Enhance content with additional analysis
            content.update(self._analyze_content(article.text))
            return content
            
        except Exception as e:
            logger.error(f"Content extraction error for {url}: {str(e)}")
            return None
            
    def _analyze_content(self, text: str) -> Dict:
        """Perform detailed content analysis"""
        try:
            # Sentiment analysis
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            # Keyword extraction
            kw_extractor = yake.KeywordExtractor()
            keywords = kw_extractor.extract_keywords(text)
            
            # Named entity recognition
            entities = self._extract_entities(text)
            
            # Readability metrics
            readability = self._calculate_readability(text)
            
            return {
                'sentiment': {
                    'polarity': sentiment.polarity,
                    'subjectivity': sentiment.subjectivity
                },
                'keywords': [kw[0] for kw in keywords[:10]],
                'entities': entities,
                'readability': readability
            }
            
        except Exception as e:
            logger.error(f"Content analysis error: {str(e)}")
            return {}
            
    @staticmethod
    def _extract_entities(text: str) -> Dict:
        """Extract named entities from text"""
        # Implementation for named entity recognition
        pass
        
    @staticmethod
    def _calculate_readability(text: str) -> Dict:
        """Calculate various readability metrics"""
        # Implementation for readability metrics
        pass

class SearchAwareContext:
    """Enhanced context manager with search integration"""
    
    def __init__(self, search_engine: WebSearchEngine):
        self.search_engine = search_engine
        self.context_history = []
        self.search_results = {}
        self.embeddings_cache = {}
        
    async def enhance_prompt(self, prompt: str) -> Tuple[str, Dict]:
        """Enhance prompt with relevant search results"""
        search_results = await self.search_engine.search(prompt)
        relevant_results = await self._process_results(search_results, prompt)
        
        enhanced_prompt = self._construct_enhanced_prompt(prompt, relevant_results)
        context = {
            'search_results': relevant_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return enhanced_prompt, context
        
    async def _process_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Process and filter search results"""
        processed_results = []
        
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self.search_engine.extract_content(result['url']))
                for result in results
            ]
            
        for task, result in zip(tasks, results):
            try:
                content = await task
                if content:
                    relevance = self._calculate_relevance(content, query)
                    if relevance > 0.5:  # Relevance threshold
                        processed_results.append({
                            **result,
                            **content,
                            'relevance_score': relevance
                        })
            except Exception as e:
                logger.error(f"Result processing error: {str(e)}")
                
        return sorted(processed_results, key=lambda x: x['relevance_score'], reverse=True)
        
    def _calculate_relevance(self, content: Dict, query: str) -> float:
        """Calculate relevance score between content and query"""
        # Implementation for relevance calculation
        pass

class EnhancedOllamaAPI:
    """Enhanced Ollama API with search capabilities"""
    
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.auth = (username, password)
        self.search_context = SearchAwareContext(WebSearchEngine())
        
    async def generate_response(self, 
                              prompt: str,
                              search_enabled: bool = True,
                              **kwargs) -> Dict:
        """Generate response with search enhancement"""
        try:
            if search_enabled:
                enhanced_prompt, context = await self.search_context.enhance_prompt(prompt)
            else:
                enhanced_prompt, context = prompt, {}
                
            response = await self._generate(enhanced_prompt, context, **kwargs)
            return self._process_response(response, context)
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise
class EnhancedStreamlitUI:
    """Advanced Streamlit UI with search integration and multi-modal support"""
    
    def __init__(self):
        self.initialize_session_state()
        self.file_processor = FileProcessor()
        self.data_analyzer = DataAnalyzer()
        
    def initialize_session_state(self):
        """Initialize enhanced session state"""
        defaults = {
            'chat_history': [],
            'search_history': [],
            'file_cache': {},
            'current_mode': 'chat',  # chat, pdf, csv, image, search
            'api_settings': {
                'url': "https://theaisource-u29564.vm.elestio.app:57987",
                'username': "root",
                'password': "eZfLK3X4-SX0i-UmgUBe6E",
                'model': "llama2",
                'temperature': 0.7,
                'max_tokens': 2048,
            },
            'search_settings': {
                'enabled': True,
                'max_results': 5,
                'relevance_threshold': 0.5,
                'cache_duration': 3600,  # 1 hour
            },
            'visualization_settings': {
                'theme': 'light',
                'chart_style': 'plotly',
                'color_scheme': 'viridis',
            },
            'active_files': {},
            'analysis_cache': {},
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render_sidebar(self):
        """Render enhanced sidebar with all controls"""
        with st.sidebar:
            self._render_mode_selector()
            self._render_file_upload()
            self._render_search_settings()
            self._render_model_settings()
            self._render_visualization_settings()
            
    def _render_mode_selector(self):
        """Render mode selection interface"""
        st.sidebar.markdown("### 🔄 Mode Selection")
        modes = {
            'chat': '💬 General Chat',
            'pdf': '📄 PDF Analysis',
            'csv': '📊 Data Analysis',
            'image': '🖼️ Image Analysis',
            'search': '🔍 Web Search'
        }
        
        selected_mode = st.sidebar.selectbox(
            "Choose Mode",
            list(modes.keys()),
            format_func=lambda x: modes[x]
        )
        
        if selected_mode != st.session_state.current_mode:
            st.session_state.current_mode = selected_mode
            st.experimental_rerun()

    def _render_file_upload(self):
        """Render enhanced file upload interface"""
        st.sidebar.markdown("### 📁 File Management")
        
        uploaded_files = st.sidebar.file_uploader(
            "Upload Files",
            type=['pdf', 'csv', 'jpg', 'jpeg', 'png', 'txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                self._process_uploaded_file(file)

    def _process_uploaded_file(self, file):
        """Process uploaded file with progress tracking"""
        try:
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            # Update status
            status_text.text(f"Processing {file.name}...")
            progress_bar.progress(25)
            
            # Read and hash file
            file_bytes = file.read()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            
            # Check cache
            if file_hash in st.session_state.file_cache:
                status_text.text(f"Loading {file.name} from cache...")
                file_data = st.session_state.file_cache[file_hash]
            else:
                # Process file based on type
                progress_bar.progress(50)
                file_data = self.file_processor.process_file(file_bytes, file.type, file.name)
                
                # Analyze file
                progress_bar.progress(75)
                analysis = self.data_analyzer.analyze_file(file_data)
                file_data.analysis = analysis
                
                # Cache results
                st.session_state.file_cache[file_hash] = file_data
            
            # Update active files
            st.session_state.active_files[file.name] = file_data
            
            # Complete progress
            progress_bar.progress(100)
            status_text.text(f"✅ {file.name} processed successfully!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.sidebar.error(f"Error processing {file.name}: {str(e)}")

    def render_main_interface(self):
        """Render main chat and analysis interface"""
        st.title("🤖 Enhanced Multi-Modal Assistant")
        
        # Render mode-specific interface
        if st.session_state.current_mode == 'chat':
            self._render_chat_interface()
        elif st.session_state.current_mode == 'pdf':
            self._render_pdf_interface()
        elif st.session_state.current_mode == 'csv':
            self._render_csv_interface()
        elif st.session_state.current_mode == 'image':
            self._render_image_interface()
        elif st.session_state.current_mode == 'search':
            self._render_search_interface()

    def _render_chat_interface(self):
        """Render enhanced chat interface"""
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "metadata" in message:
                    self._render_message_metadata(message["metadata"])
        
        # Chat input
        if prompt := st.chat_input("Enter your message..."):
            self._handle_chat_input(prompt)

    def _handle_chat_input(self, prompt: str):
        """Process chat input with search enhancement"""
        # Add user message
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.chat_history.append(user_message)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get search-enhanced response
                    response = asyncio.run(self._get_enhanced_response(prompt))
                    
                    # Display response with any search results
                    self._display_enhanced_response(response)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["content"],
                        "metadata": response["metadata"],
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

    async def _get_enhanced_response(self, prompt: str) -> Dict:
        """Get response with search enhancement"""
        api = EnhancedOllamaAPI(
            st.session_state.api_settings["url"],
            st.session_state.api_settings["username"],
            st.session_state.api_settings["password"]
        )
        
        response = await api.generate_response(
            prompt,
            search_enabled=st.session_state.search_settings["enabled"],
            model=st.session_state.api_settings["model"],
            temperature=st.session_state.api_settings["temperature"],
            max_tokens=st.session_state.api_settings["max_tokens"]
        )
        
        return response

    def _display_enhanced_response(self, response: Dict):
        """Display response with search results and visualizations"""
        # Display main response
        st.markdown(response["content"])
        
        # Display search results if available
        if "search_results" in response["metadata"]:
            with st.expander("🔍 Related Search Results"):
                for result in response["metadata"]["search_results"]:
                    st.markdown(f"**[{result['title']}]({result['url']})**")
                    st.markdown(f"_{result['summary']}_")
                    st.markdown(f"Relevance Score: {result['relevance_score']:.2f}")
                    st.markdown("---")
        
        # Display visualizations if available
        if "visualizations" in response["metadata"]:
            with st.expander("📊 Visualizations"):
                for viz in response["metadata"]["visualizations"]:
                    st.plotly_chart(viz, use_container_width=True)

    def _render_pdf_interface(self):
        """Render PDF analysis interface"""
        if not st.session_state.active_files:
            st.info("Please upload a PDF file to begin analysis")
            return
            
        # File selector
        pdf_files = {k: v for k, v in st.session_state.active_files.items() 
                    if v.content_type == "application/pdf"}
        if not pdf_files:
            st.warning("No PDF files uploaded")
            return
            
        selected_file = st.selectbox("Select PDF", list(pdf_files.keys()))
        file_data = pdf_files[selected_file]
        
        # Display PDF analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📄 Document Content")
            st.markdown(file_data.content)
            
        with col2:
            st.markdown("### 📊 Document Analysis")
            st.markdown(f"**Pages:** {file_data.metadata['num_pages']}")
            st.markdown(f"**Words:** {file_data.analysis['word_count']}")
            st.markdown(f"**Reading Time:** {file_data.analysis['reading_time']} min")
            
            # Display key topics
            st.markdown("### 🔑 Key Topics")
            for topic, score in file_data.analysis['key_topics']:
                st.progress(score, text=topic)

    def _render_csv_interface(self):
        """Render CSV analysis interface"""
        if not st.session_state.active_files:
            st.info("Please upload a CSV file to begin analysis")
            return
            
        # File selector
        csv_files = {k: v for k, v in st.session_state.active_files.items() 
                    if v.content_type == "text/csv"}
        if not csv_files:
            st.warning("No CSV files uploaded")
            return
            
        selected_file = st.selectbox("Select CSV", list(csv_files.keys()))
        file_data = csv_files[selected_file]
        
        # Analysis options
        st.markdown("### 📊 Data Analysis Options")
        analysis_type = st.selectbox(
            "Choose Analysis Type",
            ["Overview", "Detailed Analysis", "Visualization", "Correlation"]
        )
        
        if analysis_type == "Overview":
            self._render_csv_overview(file_data)
        elif analysis_type == "Detailed Analysis":
            self._render_csv_detailed_analysis(file_data)
        elif analysis_type == "Visualization":
            self._render_csv_visualization(file_data)
        else:
            self._render_csv_correlation(file_data)

    def _render_image_interface(self):
        """Render image analysis interface"""
        if not st.session_state.active_files:
            st.info("Please upload an image file to begin analysis")
            return
            
        # File selector
        image_files = {k: v for k, v in st.session_state.active_files.items() 
                      if v.content_type.startswith("image/")}
        if not image_files:
            st.warning("No image files uploaded")
            return
            
        selected_file = st.selectbox("Select Image", list(image_files.keys()))
        file_data = image_files[selected_file]
        
        # Display image and analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🖼️ Image")
            st.image(file_data.content)
            
        with col2:
            st.markdown("### 📊 Image Analysis")
            st.markdown(f"**Dimensions:** {file_data.metadata['dimensions']}")
            st.markdown(f"**Format:** {file_data.metadata['format']}")
            st.markdown(f"**Color Mode:** {file_data.metadata['mode']}")
            
            if file_data.analysis['has_text']:
                st.markdown("### 📝 Detected Text")
                st.markdown(file_data.analysis['detected_text'])

    def _render_search_interface(self):
        """Render web search interface"""
        st.markdown("### 🔍 Web Search")
        
        # Search input
        search_query = st.text_input("Enter search query")
        
        if search_query:
            with st.spinner("Searching..."):
                try:
                    # Perform search
                    search_results = asyncio.run(self._perform_search(search_query))
                    
                    # Display results
                    self._display_search_results(search_results)
                    
                except Exception as e:
                    st.error(f"Search error: {str(e)}")

def main():
    """Main application entry point"""
    # Page config
    st.set_page_config(
        page_title="Enhanced Multi-Modal Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize UI
    ui = EnhancedStreamlitUI()
    
    # Render sidebar
    ui.render_sidebar()
    
    # Render main interface
    ui.render_main_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🔒 Secure API Connection • 🔍 Web Search Enabled • 📊 Multi-Modal Analysis</p>
            <p style='font-size: 0.8em'>Enhanced Multi-Modal Assistant v2.0</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
