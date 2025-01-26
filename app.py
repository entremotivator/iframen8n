import streamlit as st
import st_pages  # required modules


# Set page config
st.set_page_config(page_title="TalkNexus - Ollama Chatbot Multi-Model Interface", layout="wide", page_icon="🤖")

# Load custom CSS from file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('styles.css')

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Header
st.markdown(f"""
<div class="header">
    <div class="animated-bg"></div>
    <div class="header-content">
        <h1 class="header-title">Ollama Chatbot Multi-Model Interface</h1> 
        <p class="header-subtitle">Advanced Language Models & Intelligent Conversations</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced pages definition
PAGES = {
    "Home": {
        "icon": "house-door",
        "func": st_pages.home,
        "description": "Guidelines & Overview",
        "badge": "Informative",
        "color": "var(--primary-color)"
    },
    "Agent Chat": {
        "icon": "chat-dots",
        "func": "pages.💭Agent Chat",
        "description": "Chat with AI Agents",
        "badge": "Application",
        "color": "var(--highlight-color)"
    },
    "Dashboard": {
        "icon": "bar-chart",
        "func": "pages.1_📊Dashboard",
        "description": "Interactive Data Overview",
        "badge": "Analytics",
        "color": "var(--secondary-color)"
    },
    "Agent Projects": {
        "icon": "folder",
        "func": "pages.2_ 📁_Agent Projects",
        "description": "Manage Agent Projects",
        "badge": "Management",
        "color": "var(--primary-color)"
    },
    "Internet Agent": {
        "icon": "search",
        "func": "pages.2_🔍_Internet Agent",
        "description": "Web-Savvy AI Agents",
        "badge": "Exploration",
        "color": "var(--highlight-color)"
    },
    "AI Agent Roster": {
        "icon": "person-workspace",
        "func": "pages.2_🧑‍💻_AI Agent Roster",
        "description": "AI Agent Directory",
        "badge": "Information",
        "color": "var(--secondary-color)"
    },
    "Agent Headquarters": {
        "icon": "building",
        "func": "pages.3_ 🏢_Agent HeadQuaters",
        "description": "AI Base of Operations",
        "badge": "HQ",
        "color": "var(--primary-color)"
    },
    "Agent Generator": {
        "icon": "gear",
        "func": "pages.3_⚙️_Agent Generator",
        "description": "Create Custom Agents",
        "badge": "Tool",
        "color": "var(--highlight-color)"
    },
    "LLM Agents": {
        "icon": "robot",
        "func": "pages.3_🛋_LLM Agents",
        "description": "Large Language Models",
        "badge": "Application",
        "color": "var(--secondary-color)"
    },
    "LLM Library": {
        "icon": "book",
        "func": "pages.3_📚LLM Libary",
        "description": "Central Model Repository",
        "badge": "Library",
        "color": "var(--primary-color)"
    },
    "Agent Command": {
        "icon": "command",
        "func": "pages.3_🧠Agent Command",
        "description": "Control AI Agents",
        "badge": "Command",
        "color": "var(--highlight-color)"
    },
    "Agent Tool Library": {
        "icon": "tool",
        "func": "pages.3_🧠Agent Tool Libary",
        "description": "Comprehensive Toolset",
        "badge": "Tools",
        "color": "var(--secondary-color)"
    },
    "Forms": {
        "icon": "pencil",
        "func": "pages.✍️ Forms",
        "description": "Manage Data Collection",
        "badge": "Forms",
        "color": "var(--primary-color)"
    },
    "Visual Agent Flow": {
        "icon": "circle-fill",
        "func": "pages.🔀 Visual Agent Flow",
        "description": "Visualize AI Workflows",
        "badge": "Visualization",
        "color": "var(--highlight-color)"
    },
    "Content Agents": {
        "icon": "file-earmark-text",
        "func": "pages.📁 Content Agents",
        "description": "AI Content Generation",
        "badge": "Content",
        "color": "var(--secondary-color)"
    },
    "Active Agents": {
        "icon": "battery-charging",
        "func": "pages.🔋 Active Agents",
        "description": "Monitor Agent Activity",
        "badge": "Activity",
        "color": "var(--primary-color)"
    },
    "Format Agents": {
        "icon": "file-code",
        "func": "pages.🤖 Format Agents",
        "description": "Formatting Assistance",
        "badge": "Tools",
        "color": "var(--highlight-color)"
    }
}

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
""", unsafe_allow_html=True)

def navigate():
    with st.sidebar:
        st.markdown('''
        <a href="https://github.com/TsLu1s/talknexus" target="_blank" style="text-decoration: none; color: inherit; display: block;">
            <div class="header-container" style="cursor: pointer;">
                <div class="profile-section">
                    <div class="profile-info">
                        <h1 style="font-size: 32px;">TalkNexus</h1>
                        <span class="active-badge" style="font-size: 16px;">AI Chatbot Multi-Model Application</span>
                    </div>
                </div>
            </div>
        </a>
        ''', unsafe_allow_html=True)

        st.markdown('---')

        # Create menu items
        for page, info in PAGES.items():
            selected = st.session_state.current_page == page

            # Create the button (invisible but clickable)
            if st.button(
                f"{page}",
                key=f"nav_{page}",
                use_container_width=True,
                type="secondary" if selected else "primary"
            ):
                st.session_state.current_page = page
                st.rerun()

            # Visual menu item
            st.markdown(f"""
                <div class="menu-item {'selected' if selected else ''}">
                    <div class="menu-icon">
                        <i class="bi bi-{info['icon']}"></i>
                    </div>
                    <div class="menu-content">
                        <div class="menu-title">{page}</div>
                        <div class="menu-description">{info['description']}</div>
                    </div>
                    <div class="menu-badge">{info['badge']}</div>
                </div>
            """, unsafe_allow_html=True)

        # Close navigation container
        st.markdown('</div>', unsafe_allow_html=True)

        return st.session_state.current_page

# Get selected page and run its function
try:
    selected_page = navigate()
    # Update session state
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

    # Run the selected function
    page_function = PAGES[selected_page]["func"]
    page_function()
except Exception as e:
    st.error(f"Error loading page: {str(e)}")
    st_pages.home.run()

# Display the footer
st.markdown("""
<div class="footer">
    <div class="footer-content">
        <p>© 2024 Powered by <a href="https://github.com/TsLu1s" target="_blank">TsLu1s</a>. 
        Advanced Language Models & Intelligent Conversations
        | Project Source: <a href="https://github.com/TsLu1s/talknexus" target="_blank"> TalkNexus</a></p>
    </div>
</div>
""", unsafe_allow_html=True)
