"""
Streamlit Sidebar Component.
Provides tenant selection, dual-mode toggles, and model parameters.
"""
import streamlit as st

def render_sidebar():
    st.sidebar.title("⚙️ Controls")
    tenant_id = st.sidebar.text_input("Tenant ID", value="default_tenant")
    offline_toggle = st.sidebar.checkbox("Force Offline Extractive Engine", value=False)
    top_k = st.sidebar.slider("Top-K Evidence Chunks", min_value=1, max_value=20, value=5)
    return tenant_id, offline_toggle, top_k