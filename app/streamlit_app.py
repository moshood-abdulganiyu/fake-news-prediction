"""
Streamlit app for the fake news classifier.

Loads the saved model/vectorizer via src.model.predict() - no training
happens here, ever. If models/model.joblib and models/vectorizer.joblib
don't exist yet, predict() raises FileNotFoundError, which is caught
below and shown as a clear on-screen message instead of a crash.

Run from the repo root:
    uv run streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# app/ is a sibling of src/, not a parent - add repo root to sys.path so
# `from src.model import predict` resolves regardless of where streamlit
# sets its working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import predict  # noqa: E402

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# --- Styling: green glassmorphism ---
# Targets Streamlit's own container via data-testid selectors, since
# Streamlit has no native "wrap widgets in a custom div" option. These
# selector names aren't a stable public API and can change between
# Streamlit versions - if styling breaks after an upgrade, this is why.
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b2818 0%, #0f3d24 45%, #145c34 100%);
    }
    [data-testid="stAppViewContainer"] > .main .block-container {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2.5rem 3rem 3rem 3rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.28);
        max-width: 760px;
    }
    h1, h2, h3, p, span, label, .stCaption, .stMarkdown {
        color: #eafff2 !important;
    }
    .stTextInput input, .stTextArea textarea {
        background: rgba(255, 255, 255, 0.10) !important;
        color: #f2fff8 !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 10px !important;
    }
    .stButton > button {
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        color: white;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1fb45c, #12833f);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #24cf6a, #159146);
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Hero image: inline SVG, no external asset dependency ---
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 0.5rem;">
        <svg width="72" height="72" viewBox="0 0 24 24" fill="none"
             xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L3 6V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V6L12 2Z"
                  fill="#1fb45c" fill-opacity="0.35" stroke="#eafff2" stroke-width="1.2"/>
            <path d="M8 12.5L10.8 15.3L16 9.5" stroke="#eafff2" stroke-width="1.6"
                  stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center;'>Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Paste a headline and article body. Trained on "
    "title+text with LinearSVC after fixing several data-leakage issues in the "
    "source dataset (see README for details).</p>",
    unsafe_allow_html=True,
)

# --- Session state for Clear button ---
if "headline" not in st.session_state:
    st.session_state.headline = ""
if "article_text" not in st.session_state:
    st.session_state.article_text = ""

def clear_fields():
    # Runs as an on_click callback, which fires BEFORE the script reruns
    # and recreates the widgets below. Setting session_state here (not
    # after the widgets already exist in the same run) is what Streamlit
    # requires - assigning it post-hoc raises
    # StreamlitWidgetAlreadyInstantiatedError.
    st.session_state.headline = ""
    st.session_state.article_text = ""


headline = st.text_input("Headline", key="headline")
text = st.text_area("Article text", height=250, key="article_text")

col1, col2 = st.columns([1, 1])
with col1:
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)
with col2:
    st.button("Clear", type="secondary", use_container_width=True, on_click=clear_fields)

if predict_clicked:
    if not headline.strip() and not text.strip():
        st.warning("Enter a headline and/or article text first.")
    else:
        try:
            label, confidence = predict(headline, text)
        except FileNotFoundError:
            st.error(
                "Model files not found. Run `uv run python -m src.model` "
                "from the repo root first to train and save the model."
            )
        else:
            if label == "Fake":
                st.markdown(
                    "<div style='text-align:center; font-size:2.3rem; font-weight:700; "
                    "color:#ffe1e1; background: rgba(220,38,38,0.25); "
                    "border:1px solid rgba(220,38,38,0.5); border-radius:14px; "
                    f"padding:1rem; margin: 0.75rem 0;'>Prediction: {label} ❌</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='text-align:center; font-size:2.3rem; font-weight:700; "
                    "color:#d8ffe9; background: rgba(31,180,92,0.25); "
                    "border:1px solid rgba(31,180,92,0.5); border-radius:14px; "
                    f"padding:1rem; margin: 0.75rem 0;'>Prediction: {label} ✅</div>",
                    unsafe_allow_html=True,
                )

            st.write(f"Confidence: {confidence:.1%}")
            st.progress(confidence)

            st.caption(
                "Confidence is derived from LinearSVC's decision function "
                "distance, not a calibrated probability - treat it as a "
                "rough signal of how far this fell from the decision "
                "boundary, not a precise likelihood."
            )