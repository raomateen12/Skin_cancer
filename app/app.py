import streamlit as st

st.set_page_config(page_title="Fair Medical AI", layout="wide")

# Theme selection via sidebar
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])

# Basic Custom CSS for a clean, modern medical UI
if theme == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #121212; color: #E0E0E0; }
        .st-bw { background-color: #1e1e1e; }
        .css-1d391kg { background-color: #1a1a1a; }
        h1, h2, h3 { color: #00e5ff; font-family: 'Inter', sans-serif; }
        .stTabs [data-baseweb="tab-list"] { gap: 20px; }
        .stTabs [data-baseweb="tab"] { color: #888; }
        .stTabs [aria-selected="true"] { color: #00e5ff; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background-color: #f8f9fa; color: #212529; }
        h1, h2, h3 { color: #0056b3; font-family: 'Inter', sans-serif; }
        .stTabs [data-baseweb="tab-list"] { gap: 20px; }
        .stTabs [data-baseweb="tab"] { color: #6c757d; }
        .stTabs [aria-selected="true"] { color: #0056b3; }
        </style>
    """, unsafe_allow_html=True)

st.title("Fair Medical AI Dashboard")
st.markdown("Skin Disease Diagnosis with Explainability, Fairness Analysis & RAG-Powered Q&A")

tab1, tab2, tab3, tab4 = st.tabs(["Diagnosis", "Fairness", "Ask a Question", "Model Comparison"])

with tab1:
    st.header("Diagnosis & Explainability")
    st.info("Model will be connected after training. Upload functionality is disabled temporarily.")

with tab2:
    st.header("Fairness Audit")
    st.info("Fairness metrics and demographic analysis will appear here after model evaluation.")

with tab3:
    st.header("Medical Q&A (RAG)")
    st.info("RAG pipeline will be connected after index creation. You will be able to ask medical context questions here.")

with tab4:
    st.header("Model Comparison")
    st.info("Performance comparison across architectures (ResNet, EfficientNet) will be populated post-training.")
