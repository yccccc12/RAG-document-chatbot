import streamlit as st
import time

# Page set up
chat_page = st.Page("./pages/chat_assistant.py", title="Chat Assistant", icon="💬")
research_page = st.Page("./pages/arxiv_search_engine.py", title="ArXiv Search Engine", icon="📚")

# Navigation menu
pg = st.navigation([chat_page, research_page])

st.set_page_config(page_title="AI Research Suite", layout="wide")

# Initialize session variable
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'ingest_target' not in st.session_state:
    st.session_state.ingest_target = []

if 'search_results' not in st.session_state:
    st.session_state.search_results = []

if 'preview_idx' not in st.session_state:
    st.session_state.preview_idx = 0

if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""

if "tavily_api_key" not in st.session_state:
    st.session_state.tavily_api_key = ""

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Use the session state directly as the value
    google_input = st.text_input(
        "Google API Key", 
        value=st.session_state.google_api_key, 
        type="password"
    )
    tavily_input = st.text_input(
        "Tavily API Key", 
        value=st.session_state.tavily_api_key, 
        type="password"
    )

    if st.button("💾 Save", use_container_width=False, type="primary"):
        st.session_state.google_api_key = google_input
        st.session_state.tavily_api_key = tavily_input
        st.success("✅ API Keys Saved!")

        time.sleep(1)
        st.rerun()

    st.divider()

# Run the selected page
pg.run()