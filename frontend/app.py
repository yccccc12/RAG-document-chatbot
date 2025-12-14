import streamlit as st
import requests

# --- FastAPI backend URL ---
BACKEND_URL = "http://127.0.0.1:8000"

st.title("📄 RAG PDF Chatbot")

st.sidebar.title("🔑 API Keys")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# --- Upload PDFs to backend ---
if uploaded_files:
    if st.sidebar.button("Upload"):
        for file in uploaded_files:
            files = {"files": (file.name, file, "application/pdf")}
            with st.spinner(f"Uploading {file.name}..."):
                res = requests.post(f"{BACKEND_URL}/upload_pdf/", files=files)
                if res.status_code == 200:
                    st.sidebar.success(f"{file.name} uploaded successfully!")
                else:
                    st.sidebar.error(f"Error uploading {file.name}: {res.text}")

st.markdown("### 💬 Ask a question about your PDFs")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Getting answer..."):
            data = {"question": question, "api_key": api_key}
            res = requests.post(f"{BACKEND_URL}/ask/", data=data)

            if res.status_code == 200:
                result = res.json()
                st.success("✅ Answer:")
                st.write(result["answer"])
                st.write("**Sources:**")
                for source in result["sources"]:
                    st.write(f"Page {source['page']}: {source['content']}")
            else:
                st.error(f"❌ Error")
