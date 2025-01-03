import streamlit as st
import st_pages  # Ensure this module is correctly installed and configured.

# Set page config
st.set_page_config(
    page_title="TalkNexus - Ollama Chatbot Multi-Model Interface",
    layout="wide",
    page_icon="🤖"
)

# Load custom CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found. Default styling will be applied.")

load_css('styles.css')

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Header
st.markdown("""
<div class="header">
    <div class="animated-bg"></div>
    <div class="header-content">
        <h1 class="header-title">Ollama Chatbot Multi-Model Interface</h1> 
        <p class="header-subtitle">Advanced Language Models & Intelligent Conversations</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Define pages and their functions
PAGES = {
    "Home": {
        "icon": "house-door",
        "func": lambda: st.write("Welcome to the Home Page! Guidelines & Overview."),
        "description": "Guidelines & Overview",
        "badge": "Informative",
        "color": "var(--primary-color)"
    },
    "Language Models Management": {
        "icon": "gear",
        "func": lambda: st.write("Manage and Download Language Models."),
        "description": "Download Models",
        "badge": "Configurations",
        "color": "var(--secondary-color)"
    },
    "AI Conversation": {
        "icon": "chat-dots",
        "func": lambda: st.write("Interactive AI Chat Interface."),
        "description": "Interactive AI Chat",
        "badge": "Application",
        "color": "var(--highlight-color)"
    },
    "RAG Conversation": {
        "icon": "chat-dots",
        "func": lambda: st.write("RAG Conversation Assistant for PDFs."),
        "description": "PDF AI Chat Assistant",
        "badge": "Application",
        "color": "var(--highlight-color)"
    },
    "Analytics Dashboard": {
        "icon": "bar-chart-line",
        "func": lambda: st.write("View Conversation Analytics."),
        "description": "View AI Conversation Analytics",
        "badge": "Insights",
        "color": "var(--tertiary-color)"
    },
    "Settings": {
        "icon": "sliders",
        "func": lambda: st.write("Configure your user preferences."),
        "description": "User Preferences & Configuration",
        "badge": "Customizable",
        "color": "var(--accent-color)"
    }
}

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
""", unsafe_allow_html=True)

# Sidebar Navigation
def navigate():
    with st.sidebar:
        st.markdown("""
        <a href="https://github.com/TsLu1s/talknexus" target="_blank" style="text-decoration: none; color: inherit;">
            <div class="header-container">
                <h1 style="font-size: 24px;">TalkNexus</h1>
                <span class="active-badge">AI Chatbot Multi-Model Application</span>
            </div>
        </a>
        """, unsafe_allow_html=True)

        st.markdown('---')

        for page, info in PAGES.items():
            selected = st.session_state.current_page == page

            if st.button(f"{page}", key=f"nav_{page}", use_container_width=True):
                st.session_state.current_page = page
                st.experimental_rerun()

            st.markdown(f"""
            <div class="menu-item {'selected' if selected else ''}">
                <i class="bi bi-{info['icon']}"></i> {page}
            </div>
            """, unsafe_allow_html=True)

# Main app logic
try:
    selected_page = navigate()
    page_function = PAGES[st.session_state.current_page]["func"]
    page_function()
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("""
<div class="footer">
    <p>© 2024 Powered by <a href="https://github.com/TsLu1s" target="_blank">TsLu1s</a>. 
    Advanced Language Models & Intelligent Conversations.</p>
</div>
""", unsafe_allow_html=True)
