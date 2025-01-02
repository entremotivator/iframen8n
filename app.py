import streamlit as st
import requests
from typing import List, Dict
import json

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_message_to_ollama(message: str) -> str:
    """Send a message to Ollama API and return the response."""
    try:
        response = requests.post(
            "https://theaisource-u29564.vm.elestio.app:57987/query",
            auth=("root", "eZfLK3X4-SX0i-UmgUBe6E"),
            json={"prompt": message},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get('response', 'No response received')
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Ollama Chat",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Ollama Chat Interface")
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This is a chat interface for Ollama AI.
        - Type your message in the input box
        - Press Enter or click Send to chat
        - Chat history is maintained during the session
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to discuss?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_message_to_ollama(prompt)
                st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🔒 Secure API Connection • Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
