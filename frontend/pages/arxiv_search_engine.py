import streamlit as st
import requests

# --- FastAPI backend URL ---
BACKEND_URL  = st.secrets.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="arXiv Search Engine", 
    page_icon="📚", 
    layout="wide"
)

st.title("📚 ArXiv Search Engine")
st.markdown("Search across Computer Science and Mathematics archives.")

# --- CATEGORY HIERARCHY ---
ARXIV_HIERARCHY = {
    "Computer Science": {
        "All Computer Science": "cs.*",
        "Artificial Intelligence": "cs.AI",
        "Machine Learning": "cs.LG",
        "Computer Vision": "cs.CV",
        "Computation and Language (NLP)": "cs.CL",
        "Robotics": "cs.RO",
        "Software Engineering": "cs.SE",
        "Cryptography and Security": "cs.CR",
        "Databases": "cs.DB",
        "Human-Computer Interaction": "cs.HC",
    },
    "Mathematics": {
        "All Mathematics": "math.*",
        "Algebraic Geometry": "math.AG",
        "Combinatorics": "math.CO",
        "Number Theory": "math.NT",
        "Probability": "math.PR",
        "Statistics Theory": "math.ST",
        "Logic": "math.LO",
        "Differential Geometry": "math.DG",
    }
}

# Create a reverse mapping for loks up: {'cs.AI': 'Artificial Intelligence', ...}
REVERSE_MAP = {}
for domain, subcats in ARXIV_HIERARCHY.items():
    for name, code in subcats.items():
        REVERSE_MAP[code] = name

# Function to get arxiv paper via FastAPI Endpoint
def get_papers_from_api(query: str, category: str, sort_by: str, days_back: int, limit: int):
    params = {
        "query": query,
        "category": category,
        "sort_by": sort_by,
        "days_back": days_back,
        "limit": limit
    }

    try:
        res = requests.get(f"{BACKEND_URL}/search/", params=params)

        if res.status_code == 200:
            return res.json()
        else:
            st.error(f"")
    except Exception as e:
        st.error(f"❌ Connection error: {e}")
        return []

# --- Configuration Panel ---
with st.container(border=True):
    # Row 1: Primary Inputs
    row1_col1, row1_col2, row1_col3 = st.columns([4, 1, 1], vertical_alignment="bottom")
    
    with row1_col1:
        query_input = st.text_input("Search Keywords or ArXiv ID", placeholder="e.g.: Transformer")

    with row1_col2:
        search_type = st.selectbox("Sort by", ["relevance", "recent"])

    with row1_col3:
        limit = st.number_input("Results", 1, 20, 5)

    # Row 2: Hierarchical Categories
    row2_col1, row2_col2, row2_col3 = st.columns([2, 2, 2], vertical_alignment="bottom")
    
    with row2_col1:
        main_domain = st.selectbox("Main Domain", list(ARXIV_HIERARCHY.keys()))

    with row2_col2:
        sub_options = ARXIV_HIERARCHY[main_domain]
        selected_sub = st.selectbox("Subcategory", list(sub_options.keys()))
        cat_code = sub_options[selected_sub]

    with row2_col3:
        days = st.slider("Days back (0=All)", 0, 30, 0)

    # Search button
    if st.button("🔍 Search", use_container_width=True, type="primary"):
        days_param = days if days > 0 else None
        st.session_state.search_results = get_papers_from_api(
            query=query_input, 
            category=cat_code,
            sort_by=search_type, 
            days_back=days_param, 
            limit=limit
        )
        st.session_state.preview_idx = 0


# --- Main display ---
if st.session_state.search_results:
    st.write(f"### Found {len(st.session_state.search_results)} papers")
    
    # Display list of papers found
    for i, paper in enumerate(st.session_state.search_results):
        is_selected = (i == st.session_state.preview_idx)
        with st.expander(f"**📄 {paper['title']}**", expanded=is_selected):
            st.write(f"**Authors:** {', '.join(paper['authors'])}")
            st.write(f"**Date:** {paper['date']}")

            raw_categories = paper['categories']
            format_categories = [REVERSE_MAP.get(cat, cat) for cat in raw_categories]

            st.write(f"**Categories:** {', '.join(format_categories)}")
            st.write(f"**Summary:** \n\n{paper['summary']}")
            
            col_btn1, col_btn2, col_btn3, col_spacer = st.columns([1, 1, 1, 3])

            msg_placeholder = st.empty()
            
            with col_btn1:
                if st.button("🔍 Preview PDF", key=f"btn_{i}", use_container_width=True):
                    st.session_state.preview_idx = i
                    st.rerun()
            
            with col_btn2:
                st.link_button("📖 Read Abstract", paper['url'], use_container_width=True)

            with col_btn3:
                if st.button("💬 Chat with Paper", key=f"chat_{i}", use_container_width=True):
                    # Store the data to session variable
                    st.session_state.ingest_target.append(paper)
                    msg_placeholder.success("Paper queued! Switch to Chatbot.")

    st.divider()

    # --- Sliding preview panel ---
    st.header("🔍 PDF Quick-View")
    
    # Nav Buttons for Slider
    nav_prev, nav_info, nav_next = st.columns([1, 2, 1])
    with nav_prev:
        if st.button("⬅️ Previous", use_container_width=True):
            st.session_state.preview_idx = (st.session_state.preview_idx - 1) % len(st.session_state.search_results)
            st.rerun()

    with nav_next:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.preview_idx = (st.session_state.preview_idx + 1) % len(st.session_state.search_results)
    
    current_paper = st.session_state.search_results[st.session_state.preview_idx]
    
    with nav_info:
        st.markdown(f"<p style='text-align: center;'><b>{st.session_state.preview_idx + 1} / {len(st.session_state.search_results)}</b><br>{current_paper['title']}</p>", unsafe_allow_html=True)

    # PDF Display
    st.markdown(f'<iframe src="{current_paper["pdf"]}" width="100%" height="900" style="border-radius: 10px;"></iframe>', unsafe_allow_html=True)