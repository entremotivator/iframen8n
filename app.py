import streamlit as st
import requests
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# Constants
OLLAMA_API_URL = "https://theaisource-u29564.vm.elestio.app:57987"
OLLAMA_USER = "root"
OLLAMA_PASSWORD = "eZfLK3X4-SX0i-UmgUBe6E"

# Page Config
st.set_page_config(page_title="Ollama Server + LangChain", layout="wide")

# Initialize Session State
if "hide_credentials" not in st.session_state:
    st.session_state.hide_credentials = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title
st.title("Ollama Server + LangChain")

# Credential Management
with st.expander("Manage Ollama API Credentials"):
    if st.session_state.hide_credentials:
        st.write("🔒 Credentials are hidden.")
        if st.button("Show Credentials"):
            st.session_state.hide_credentials = False
    else:
        st.write(f"**URL**: {OLLAMA_API_URL}")
        st.write(f"**User**: {OLLAMA_USER}")
        st.write(f"**Password**: {OLLAMA_PASSWORD}")
        if st.button("Hide Credentials"):
            st.session_state.hide_credentials = True

# Sidebar for Navigation
st.sidebar.title("Features")
selected_feature = st.sidebar.radio(
    "Choose a feature:",
    ["Test Ollama API", "LangChain Chatbot", "Sentiment Analysis", "Text Summarization"],
)

# Feature 1: Test Ollama API
if selected_feature == "Test Ollama API":
    st.header("Test Ollama API")
    query = st.text_area("Enter your query:")
    if st.button("Submit Query"):
        if query.strip():
            try:
                response = requests.post(
                    f"{OLLAMA_API_URL}/query",
                    auth=(OLLAMA_USER, OLLAMA_PASSWORD),
                    json={"prompt": query},
                )
                if response.status_code == 200:
                    st.success("Response received successfully!")
                    st.json(response.json())
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query before submitting.")

# Feature 2: LangChain Chatbot
elif selected_feature == "LangChain Chatbot":
    st.header("LangChain Chatbot")
    st.write("Chat with a memory-enabled chatbot powered by LangChain.")

    # Initialize LangChain Memory and ConversationChain
    memory = ConversationBufferMemory()
    chatbot = ConversationChain(memory=memory, llm=OpenAI(temperature=0.7))

    user_input = st.text_input("You:", key="chat_input")
    if st.button("Send", key="chat_button"):
        if user_input.strip():
            response = chatbot.run(user_input)
            st.session_state.chat_history.append({"You": user_input, "Bot": response})
        else:
            st.warning("Please enter a message.")

    # Display Chat History
    if st.session_state.chat_history:
        st.write("### Chat History")
        for chat in st.session_state.chat_history:
            st.write(f"**You:** {chat['You']}")
            st.write(f"**Bot:** {chat['Bot']}")

# Feature 3: Sentiment Analysis
elif selected_feature == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    st.write("Analyze the sentiment of a given text using Ollama API.")

    sentiment_text = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        if sentiment_text.strip():
            try:
                response = requests.post(
                    f"{OLLAMA_API_URL}/sentiment",
                    auth=(OLLAMA_USER, OLLAMA_PASSWORD),
                    json={"text": sentiment_text},
                )
                if response.status_code == 200:
                    sentiment = response.json().get("sentiment", "Unknown")
                    st.success(f"Sentiment: {sentiment}")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter some text for analysis.")

# Feature 4: Text Summarization
elif selected_feature == "Text Summarization":
    st.header("Text Summarization")
    st.write("Summarize long texts using Ollama API.")

    summarization_text = st.text_area("Enter text for summarization:")
    if st.button("Summarize Text"):
        if summarization_text.strip():
            try:
                response = requests.post(
                    f"{OLLAMA_API_URL}/summarize",
                    auth=(OLLAMA_USER, OLLAMA_PASSWORD),
                    json={"text": summarization_text},
                )
                if response.status_code == 200:
                    summary = response.json().get("summary", "No summary available.")
                    st.success(f"Summary: {summary}")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter some text for summarization.")

# Footer
st.markdown("---")
st.caption("Powered by Ollama API and LangChain.")
