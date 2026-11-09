"""
CSS Stylesheet Injector for Streamlit UI.
Customizes buttons, metrics cards, and dark mode themes.
"""
import streamlit as st

def apply_custom_styles():
    st.markdown("""
    <style>
        .stButton>button { border-radius: 8px; font-weight: bold; }
        .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 8px; }
    </style>
    """, unsafe_allow_dict=True)