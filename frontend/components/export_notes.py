"""
Streamlit BibTeX and Markdown Export Tools Component.
Allows researchers to download structured Markdown notes or BibTeX entries.
"""
import streamlit as st
import requests

def render_export_tools(query: str, answer: str):
    """Renders download buttons for BibTeX and Markdown exports."""
    st.markdown("### 📥 Export Research Synthesis")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Markdown Note"):
            try:
                res = requests.post(
                    "http://localhost:8000/api/v1/export/markdown",
                    data={"query": query, "answer": answer}
                ).json()
                st.download_button(
                    label="Download Markdown (.md)",
                    data=res.get("export_text", ""),
                    file_name="verimind_synthesis.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"Markdown export failed: {str(e)}")

    with col2:
        if st.button("Generate BibTeX Citations"):
            try:
                res = requests.post(
                    "http://localhost:8000/api/v1/export/bibtex",
                    data={"query": query}
                ).json()
                st.download_button(
                    label="Download BibTeX (.bib)",
                    data=res.get("bibtex_str", ""),
                    file_name="verimind_citations.bib",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"BibTeX export failed: {str(e)}")