import streamlit as st

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

# Run the selected page
pg.run()