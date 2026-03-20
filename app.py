"""
app.py — Streamlit frontend for the Plant Disease Classification Agent.
Run with:  streamlit run app.py
"""

import os
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean, minimal, professional
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        color: #1a1a1a;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* App header */
    .app-header {
        border-bottom: 2px solid #2d6a4f;
        padding-bottom: 12px;
        margin-bottom: 24px;
    }
    .app-header h1 {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1b4332;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .app-header p {
        font-size: 0.92rem;
        color: #555;
        margin: 4px 0 0 0;
    }

    /* Section label */
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #2d6a4f;
        margin-bottom: 6px;
    }

    /* Response container */
    .response-box {
        background: #f8fffe;
        border: 1px solid #b7e4c7;
        border-left: 4px solid #2d6a4f;
        border-radius: 6px;
        padding: 20px 24px;
        margin-top: 16px;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #1a1a1a;
        white-space: pre-wrap;
    }

    /* Warning box for low confidence */
    .warning-box {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-left: 4px solid #f9a825;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 12px;
        font-size: 0.9rem;
        color: #5d4037;
    }

    /* Status tag */
    .status-running {
        display: inline-block;
        background: #e8f5e9;
        color: #2d6a4f;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    /* Sidebar labels */
    .sidebar-section {
        font-size: 0.8rem;
        font-weight: 600;
        color: #444;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 16px;
        margin-bottom: 4px;
    }

    /* Button override */
    .stButton > button {
        background-color: #2d6a4f;
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 10px 24px;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #1b4332;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>Plant Disease Diagnosis Agent</h1>
    <p>Upload a leaf image and ask a question. The agent classifies the disease, retrieves expert knowledge, and generates a structured response.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------
groq_key = os.getenv("GROQ_API_KEY", "")
if not groq_key:
    st.error(
        "GROQ_API_KEY is missing. Create a .env file in this directory with the line:\n"
        "GROQ_API_KEY=your_key_here"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Session state — cache the agent so it is not rebuilt on every interaction
# ---------------------------------------------------------------------------
if "agent" not in st.session_state:
    with st.spinner("Loading model and initialising agent..."):
        try:
            from core.agent import PlantDiseaseAgent
            st.session_state.agent = PlantDiseaseAgent()
        except Exception as e:
            st.error(f"Failed to initialise agent: {e}")
            st.stop()

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Sidebar — configuration and history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Model Configuration**")
    st.caption(f"ViT weights: `{os.getenv('MODEL_PATH', 'plant_model.pth')}`")
    st.caption("LLM: llama-3.1-8b-instant via Groq")

    st.markdown("---")
    st.markdown("**Confidence Thresholds**")
    st.caption("70%+ = High confidence")
    st.caption("40-69% = Moderate (mentioned in response)")
    st.caption("Below 40% = Low (agent requests clearer image)")

    st.markdown("---")
    if st.session_state.history:
        st.markdown("**Session History**")
        for i, entry in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.caption(f"{i}. {entry['query'][:55]}...")
    else:
        st.caption("No queries yet in this session.")

    st.markdown("---")
    if st.button("Clear session"):
        st.session_state.history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-label">Plant Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a leaf photograph",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True, caption="Uploaded image")

    st.markdown('<div class="section-label" style="margin-top:20px;">Your Question</div>', unsafe_allow_html=True)
    user_query = st.text_area(
        "Question",
        placeholder="What disease does this plant have? How do I treat it?",
        height=100,
        label_visibility="collapsed",
    )

    run_button = st.button("Run Diagnosis")


with col_right:
    st.markdown('<div class="section-label">Agent Response</div>', unsafe_allow_html=True)

    if run_button:
        if not user_query.strip():
            st.warning("Please enter a question before running the agent.")
        else:
            # Save image to a temp file so the agent tool can read it
            image_path = None
            if uploaded_file:
                suffix = os.path.splitext(uploaded_file.name)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    image_path = tmp.name

            st.markdown('<span class="status-running">Agent running...</span>', unsafe_allow_html=True)

            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.run(
                        user_query=user_query,
                        image_path=image_path,
                    )

                    # Clean up temp file
                    if image_path and os.path.exists(image_path):
                        os.unlink(image_path)

                    # Save to history
                    st.session_state.history.append({
                        "query": user_query,
                        "response": response,
                        "had_image": uploaded_file is not None,
                    })

                    st.markdown(
                        f'<div class="response-box">{response}</div>',
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    if image_path and os.path.exists(image_path):
                        os.unlink(image_path)
                    st.error(f"Agent error: {e}")

    elif st.session_state.history:
        # Show last response until a new one is generated
        last = st.session_state.history[-1]
        st.caption("Showing last response")
        st.markdown(
            f'<div class="response-box">{last["response"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="response-box" style="color:#aaa; font-style:italic;">'
            'Upload an image and enter a question, then click Run Diagnosis.'
            '</div>',
            unsafe_allow_html=True,
        )
