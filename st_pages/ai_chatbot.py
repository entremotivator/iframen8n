import os
import requests
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# API Configuration
DEFAULT_API_URL = "https://theaisource-u29564.vm.elestio.app:57987"
DEFAULT_USERNAME = "root"
DEFAULT_PASSWORD = "eZfLK3X4-SX0i-UmgUBe6E"


class StreamHandler(BaseCallbackHandler):
    """
    Custom callback handler for streaming LLM responses token by token.
    """
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        try:
            self.text += token
            clean_text = self.text
            # Clean up AIMessage formatting
            if "AIMessage" in clean_text:
                clean_text = clean_text.replace("AIMessage(", "").replace(", additional_kwargs={}", "")
                clean_text = clean_text.replace(", response_metadata={})", "").replace('{ "data":', "").replace('}', "")
            self.container.markdown(clean_text)
        except Exception as e:
            print(f"Warning in StreamHandler: {str(e)}")
            self.container.markdown(self.text)


def get_ollama_models() -> list:
    """
    Retrieves available models from the cloud Ollama API.
    """
    try:
        response = requests.get(
            f"{DEFAULT_API_URL}/api/tags",
            auth=(DEFAULT_USERNAME, DEFAULT_PASSWORD)
        )
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models['models']
                    if all(keyword not in model['name'].lower()
                           for keyword in ('failed', 'embed', 'bge'))]
        return []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


def get_conversation_chain(model_name: str) -> ConversationChain:
    """
    Initializes LangChain conversation chain with the cloud-based Ollama model.
    """
    llm = Ollama(
        model=model_name,
        temperature=0.2,
        base_url=DEFAULT_API_URL,
        auth=(DEFAULT_USERNAME, DEFAULT_PASSWORD),
    )
    prompt = PromptTemplate(
        input_variables=["history", "input"], 
        template="""Current conversation:
                    {history}
                    Human: {input}
                    Assistant:""")
    memory = ConversationBufferMemory(return_messages=True)
    return ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=True)


def on_model_change():
    """
    Callback function triggered when the selected model changes.
    """
    st.session_state.messages = []
    st.session_state.conversation = None


def run():
    """
    Main function to run the Streamlit chat interface.
    """
    st.markdown('''
    <div class="header-container">
        <p class="header-subtitle">🤖 Chat with State-of-the-Art Language Models</p>
    </div>
    ''', unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    # Get available models
    models = get_ollama_models()
    if not models:
        st.warning("Ollama is not running. Make sure to have the Ollama API accessible.")
        return

    # Model selection
    st.subheader("Select a Language Model:")
    col1, _ = st.columns([2, 6])
    with col1:
        model_name = st.selectbox(
            "Model",
            models,
            format_func=lambda x: f'🔮 {x}',
            key="model_select",
            on_change=on_model_change,
            label_visibility="collapsed"
        )

    # Initialize conversation if needed
    if st.session_state.conversation is None:
        st.session_state.conversation = get_conversation_chain(model_name)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input(f"Chat with {model_name}"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                # Create a new stream handler for this response
                stream_handler = StreamHandler(response_placeholder)
                # Temporarily add stream handler to the conversation
                st.session_state.conversation.llm.callbacks = [stream_handler]
                # Generate response
                response = st.session_state.conversation.run(prompt)
                # Clear the stream handler after generation
                st.session_state.conversation.llm.callbacks = []
                # Add response to message history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                response_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    run()

