import streamlit as st
import requests
from typing import List, Dict
import json
from datetime import datetime

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_configured" not in st.session_state:
    st.session_state.api_configured = False

# Default API settings
DEFAULT_API_URL = "https://theaisource-u29564.vm.elestio.app:57987"
DEFAULT_USERNAME = "root"
DEFAULT_PASSWORD = "eZfLK3X4-SX0i-UmgUBe6E"

def initialize_api_config():
    """Initialize API configuration in session state"""
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    if "username" not in st.session_state:
        st.session_state.username = DEFAULT_USERNAME
    if "password" not in st.session_state:
        st.session_state.password = DEFAULT_PASSWORD
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama2"

def send_message_to_ollama(message: str) -> str:
    """Send a message to Ollama API and return the response."""
    try:
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "prompt": message,
            "model": st.session_state.selected_model,
            "stream": False
        }
        
        response = requests.post(
            f"{st.session_state.api_url}/api/generate",
            auth=(st.session_state.username, st.session_state.password),
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json().get('response', 'No response received')
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return f"Error: {str(e)}"

def get_available_models() -> List[str]:
    """Get list of available models from Ollama API"""
    try:
        response = requests.get(
            f"{st.session_state.api_url}/api/tags",
            auth=(st.session_state.username, st.session_state.password),
            timeout=5
        )
        response.raise_for_status()
        models = response.json().get('models', [])
        return [model['name'] for model in models] or ["llama2", "mistral", "codellama"]
    except:
        return ["llama2", "mistral", "codellama"]  # Fallback models

def main():
    initialize_api_config()
    
    st.set_page_config(
        page_title="Ollama Chat",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Ollama Chat Interface")
    
    # Sidebar with API configuration
    with st.sidebar:
        st.markdown("### API Configuration")
        with st.expander("Configure API Settings"):
            new_api_url = st.text_input("API URL", value=st.session_state.api_url, type="password")
            new_username = st.text_input("Username", value=st.session_state.username, type="password")
            new_password = st.text_input("Password", value=st.session_state.password, type="password")
            
            if st.button("Update API Settings"):
                st.session_state.api_url = new_api_url
                st.session_state.username = new_username
                st.session_state.password = new_password
                st.success("API settings updated!")

        # Model selection
        st.markdown("### Model Selection")
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Choose a model",
            available_models,
            index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
        )
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.success(f"Model changed to {selected_model}")

        st.markdown("### Chat Controls")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("### About")
        st.markdown("""
        This is a chat interface for Ollama AI.
        - Configure API settings above
        - Select your preferred model
        - Type your message in the input box
        - Press Enter or click Send to chat
        """)

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                st.caption(f"Model: {message.get('model', 'unknown')} • {message.get('timestamp', '')}")

    # Chat input
    if prompt := st.chat_input("What would you like to discuss?"):
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking using {st.session_state.selected_model}..."):
                response = send_message_to_ollama(prompt)
                st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "model": st.session_state.selected_model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🔒 Secure API Connection • Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
