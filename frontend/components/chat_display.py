"""
Chat History Display Component.
Renders message turns and citation footnotes.
"""
import streamlit as st

def render_chat_history(messages):
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])