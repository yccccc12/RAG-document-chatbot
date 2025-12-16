import streamlit as st
import requests
import json

# --- FastAPI backend URL ---
BACKEND_URL = "http://127.0.0.1:8000"

# --- Steamlit Page Setup ---
st.set_page_config(
    page_title="RAG Document Chatbot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title ---
st.title("💬 RAG Document Chatbot")
st.caption("🚀 Chat with your PDF documents powered by Gemini & LangChain.")
st.markdown("---")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    gemini_api_key = st.text_input("Gemini API Key", key="gemini_api_key", type="password")
    
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="uploaded_files"
    )

    if uploaded_files:
        if st.button("Upload", type="primary"):
            for file in uploaded_files:
                file_payload = {"files": (file.name, file, "application/pdf")}

                with st.spinner(f"Uploading {file.name}..."):
                    try:
                        res = requests.post(f"{BACKEND_URL}/upload_pdf/", files=file_payload)
                        if res.status_code == 200:
                            st.success(f"✅ {file.name} uploaded!")
                        else:
                            st.error(f"❌ Error uploading {file.name}: {res.text}")
                    except Exception as e:
                         st.error(f"❌ Connection error: {e}")

    st.markdown("---")
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---
# Welcome Message if no history
if not st.session_state.messages:
    st.info("👋 Welcome! Upload a PDF document in the sidebar and start chatting.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- **Page {source['page']}**: {source['content']}...")

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):

    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []
        
        try:
            with requests.post(
                f"{BACKEND_URL}/ask/", 
                data={"question": prompt, "api_key": gemini_api_key},
                stream=True
            ) as r:
                if r.status_code == 200:
                    for line in r.iter_lines():
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            if data["type"] == "answer":
                                full_response += data["content"]
                                message_placeholder.markdown(full_response + "▌")
                            elif data["type"] == "sources":
                                sources = data["content"]
                    
                    message_placeholder.markdown(full_response)
                    
                    # Display sources in expander
                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                st.markdown(f"- **Page {source['page']}**: {source['content']}...")
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
                else:
                    st.error(f"Error: {r.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
