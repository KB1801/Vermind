"""
Streamlit Evidence Timeline Component.
Renders chronological evidence cards with document provenance and relevance scores.
"""
import streamlit as st
from typing import List, Dict, Any

def render_timeline_view(timeline_events: List[Dict[str, Any]]):
    """Renders chronological step-by-step evidence cards."""
    st.markdown("### ⏱️ Chronological Evidence Timeline")
    if not timeline_events:
        st.info("No evidence timeline events recorded for this query.")
        return

    for event in timeline_events:
        step = event.get("step", 1)
        doc_id = event.get("doc_id", "Unknown Doc")
        snippet = event.get("snippet", "")
        score = event.get("score", 0.0)

        with st.container():
            st.markdown(f"**Step {step}: Source `{doc_id}`** (Relevance Score: `{score:.4f}`)")
            st.caption(f"\"{snippet}\"")
            st.divider()