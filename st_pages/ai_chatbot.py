import streamlit as st
import requests
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
#from langchain_ollama import ChatOllama
from langchain_community.llms import Ollama


class StreamHandler(BaseCallbackHandler):
    """
    Custom callback handler for streaming LLM responses token by token.
    
    Attributes:
        container: Streamlit container object for displaying streamed tokens
        text (str): Accumulated response text
    """
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs):
        """
        Processes each new token from the LLM response stream.
        
        Args:
            token (str): Individual token from the LLM response
            **kwargs: Additional keyword arguments from the callback
        """
        try:
            self.text += token
            clean_text = self.text
            
            # Check if we need to clean up AIMessage formatting
            if "AIMessage" in clean_text:
                # Handle complete AIMessage format
                if "content=\"" in clean_text:
                    try:
                        clean_text = clean_text.split("content=\"")[1].rsplit("\"", 1)[0]
                    except IndexError:
                        # If splitting fails, keep the original text
                        pass
                
                # Remove any remaining AIMessage wrapper
                clean_text = (clean_text.replace("AIMessage(", "")
                                      .replace(", additional_kwargs={}", "")
                                      .replace(", response_metadata={})", "")
                                      .replace('{ "data":' , "")
                                      .replace('}' , "")
                )
            
            # Update the display with cleaned text
            self.container.markdown(clean_text)
            
        except Exception as e:
            # Log the error without disrupting the stream
            print(f"Warning in StreamHandler: {str(e)}")
            # Still try to display something to the user
            self.container.markdown(self.text)

def get_ollama_models() -> list:
    """
    Retrieves available models from Ollama API.
    
    Sends GET request to Ollama API endpoint and processes response to extract
    valid model names, filtering out failed or invalid models.
    
    Returns:
        list: List of available model names, empty if API unreachable
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models['models']
                    if all(keyword not in model['name'].lower()
                        for keyword in ('failed', 'embed', 'bge'))]
        return []
    except:
        return []

def get_conversation_chain(model_name: str) -> ConversationChain:
    """
    Initializes LangChain conversation chain with specified model.
    
    Args:
        model_name (str): Name of Ollama model to use
        
    Returns:
        ConversationChain: Configured conversation chain with memory and prompt template
    """
    # Set up Ollama LLM
    #llm = ChatOllama(
    #    model=model_name,
    #    temperature=0.2,
    #    base_url="http://localhost:11434",
        #format="json"  # Updated to use simple string format
    #)

    # Set up Ollama LLM
    llm = Ollama(
        model=model_name,
        temperature=0.2,
        base_url="http://localhost:11434",
        #system_prompt="You are a helpful AI assistant. Keep your answers brief and concise."
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
    Callback function triggered when selected model changes.
    
    Resets conversation state by clearing message history and conversation chain
    to start fresh with new model.
    """
    st.session_state.messages = []
    st.session_state.conversation = None

def run():
    """
    Main function to run the Streamlit chat interface.
    
    Initializes UI components, manages conversation state, handles model selection,
    and processes chat interactions. Implements real-time streaming of model responses
    and maintains chat history.
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
        st.warning(f"Ollama is not running. Make sure to have Ollama API installed")
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
