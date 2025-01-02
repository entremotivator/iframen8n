import streamlit as st
import requests
from typing import List, Dict, Optional, Union
import json
from datetime import datetime
import time
import re
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Data class for API response handling"""
    success: bool
    content: str
    error: Optional[str] = None
    metadata: Optional[Dict] = None
    
@dataclass
class ChatMessage:
    """Data class for chat messages"""
    role: str
    content: str
    timestamp: str
    model: Optional[str] = None
    metadata: Optional[Dict] = None

class OllamaAPI:
    """Class to handle Ollama API interactions"""
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.auth = (username, password)
        
    def _make_request(self, endpoint: str, method: str = "GET", 
                     data: Optional[Dict] = None, timeout: int = 30) -> APIResponse:
        """Make request to Ollama API with error handling"""
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            headers = {"Content-Type": "application/json"}
            
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                timeout=timeout
            )
            
            response.raise_for_status()
            return APIResponse(success=True, content=response.json())
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API Request failed: {str(e)}")
            return APIResponse(success=False, content="", error=str(e))
            
    def get_models(self) -> List[str]:
        """Get available models from Ollama"""
        response = self._make_request("api/tags")
        if response.success:
            try:
                models = response.content.get('models', [])
                return [model['name'] for model in models]
            except (KeyError, AttributeError) as e:
                logger.error(f"Error parsing models: {str(e)}")
                return self.get_default_models()
        return self.get_default_models()
    
    @staticmethod
    def get_default_models() -> List[str]:
        """Return default models if API call fails"""
        return ["llama2", "mistral", "codellama", "neural-chat", "starling-lm"]
    
    def generate_response(self, 
                         prompt: str, 
                         model: str,
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.7,
                         max_tokens: int = 2048,
                         top_p: float = 0.95,
                         context_window: Optional[List[Dict]] = None) -> APIResponse:
        """Generate response from Ollama with advanced parameters"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        if context_window:
            payload["context"] = context_window
            
        response = self._make_request("api/generate", method="POST", data=payload)
        return response

class ChatHistory:
    """Class to manage chat history and context"""
    def __init__(self, max_context: int = 10):
        self.messages: List[ChatMessage] = []
        self.max_context = max_context
        
    def add_message(self, message: ChatMessage):
        """Add message to history"""
        self.messages.append(message)
        
    def get_context_window(self) -> List[Dict]:
        """Get recent context for API calls"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages[-self.max_context:]
        ]
        
    def clear_history(self):
        """Clear chat history"""
        self.messages = []
        
    def export_history(self) -> str:
        """Export chat history as JSON"""
        return json.dumps([vars(msg) for msg in self.messages], indent=2)

class StreamlitUI:
    """Class to handle Streamlit UI components"""
    def __init__(self):
        self.initialize_session_state()
        
    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables"""
        defaults = {
            "messages": [],
            "api_url": "https://theaisource-u29564.vm.elestio.app:57987",
            "username": "root",
            "password": "eZfLK3X4-SX0i-UmgUBe6E",
            "selected_model": "llama2",
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.95,
            "system_prompt": "You are a helpful AI assistant.",
            "context_window_size": 10,
            "theme": "light",
            "chat_history": ChatHistory()
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render_sidebar(self, api: OllamaAPI):
        """Render sidebar with all configuration options"""
        with st.sidebar:
            st.markdown("### 🛠️ Configuration")
            
            # API Settings
            with st.expander("API Settings"):
                new_api_url = st.text_input("API URL", value=st.session_state.api_url, type="password")
                new_username = st.text_input("Username", value=st.session_state.username, type="password")
                new_password = st.text_input("Password", value=st.session_state.password, type="password")
                
                if st.button("Update API Settings"):
                    st.session_state.api_url = new_api_url
                    st.session_state.username = new_username
                    st.session_state.password = new_password
                    st.success("✅ API settings updated!")
            
            # Model Selection
            st.markdown("### 🤖 Model Selection")
            available_models = api.get_models()
            selected_model = st.selectbox(
                "Choose a model",
                available_models,
                index=available_models.index(st.session_state.selected_model) 
                if st.session_state.selected_model in available_models else 0
            )
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                st.success(f"🔄 Model changed to {selected_model}")
            
            # Advanced Settings
            with st.expander("Advanced Settings"):
                st.markdown("#### Generation Parameters")
                st.session_state.temperature = st.slider(
                    "Temperature", min_value=0.0, max_value=1.0, 
                    value=st.session_state.temperature, step=0.1
                )
                st.session_state.max_tokens = st.number_input(
                    "Max Tokens", min_value=64, max_value=4096,
                    value=st.session_state.max_tokens
                )
                st.session_state.top_p = st.slider(
                    "Top P", min_value=0.0, max_value=1.0,
                    value=st.session_state.top_p, step=0.05
                )
                
                st.markdown("#### System Prompt")
                st.session_state.system_prompt = st.text_area(
                    "System Prompt",
                    value=st.session_state.system_prompt
                )
                
                st.markdown("#### Context Window")
                st.session_state.context_window_size = st.number_input(
                    "Context Window Size",
                    min_value=1, max_value=20,
                    value=st.session_state.context_window_size
                )
            
            # Theme Selection
            st.markdown("### 🎨 Appearance")
            theme = st.selectbox(
                "Theme",
                ["light", "dark"],
                index=0 if st.session_state.theme == "light" else 1
            )
            if theme != st.session_state.theme:
                st.session_state.theme = theme
            
            # Chat Controls
            st.markdown("### 💬 Chat Controls")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear History"):
                    st.session_state.chat_history.clear_history()
                    st.rerun()
            with col2:
                if st.button("Export Chat"):
                    chat_export = st.session_state.chat_history.export_history()
                    st.download_button(
                        label="Download",
                        data=chat_export,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

    def render_chat_interface(self, api: OllamaAPI):
        """Render main chat interface"""
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history.messages:
                with st.chat_message(message.role):
                    st.markdown(message.content)
                    st.caption(
                        f"Model: {message.model or 'user'} • "
                        f"{message.timestamp} • "
                        f"Temperature: {message.metadata.get('temperature', 'N/A') if message.metadata else 'N/A'}"
                    )

        # Chat input
        if prompt := st.chat_input("What would you like to discuss?"):
            # Add user message
            user_message = ChatMessage(
                role="user",
                content=prompt,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                metadata={"type": "user_message"}
            )
            st.session_state.chat_history.add_message(user_message)
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner(f"Thinking using {st.session_state.selected_model}..."):
                    response = api.generate_response(
                        prompt=prompt,
                        model=st.session_state.selected_model,
                        system_prompt=st.session_state.system_prompt,
                        temperature=st.session_state.temperature,
                        max_tokens=st.session_state.max_tokens,
                        top_p=st.session_state.top_p,
                        context_window=st.session_state.chat_history.get_context_window()
                    )
                    
                    if response.success:
                        assistant_response = response.content.get('response', 'No response received')
                        st.markdown(assistant_response)
                        
                        # Add assistant message to history
                        assistant_message = ChatMessage(
                            role="assistant",
                            content=assistant_response,
                            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            model=st.session_state.selected_model,
                            metadata={
                                "temperature": st.session_state.temperature,
                                "max_tokens": st.session_state.max_tokens,
                                "top_p": st.session_state.top_p
                            }
                        )
                        st.session_state.chat_history.add_message(assistant_message)
                    else:
                        st.error(f"Error: {response.error}")

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Advanced Ollama Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize components
    ui = StreamlitUI()
    api = OllamaAPI(
        base_url=st.session_state.api_url,
        username=st.session_state.username,
        password=st.session_state.password
    )

    # Apply theme
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
            .stApp {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
        </style>
        """, unsafe_allow_html=True)

    # Render title
    st.title("🤖 Advanced Ollama Chat Interface")
    
    # Render sidebar
    ui.render_sidebar(api)
    
    # Render main chat interface
    ui.render_chat_interface(api)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🔒 Secure API Connection • Built with Streamlit • Advanced Features Enabled</p>
        <p style='font-size: 0.8em'>Version 2.0.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
