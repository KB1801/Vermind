"""
Streamlit Time-Travel Query Replay View.
Re-executes historical queries to verify output reproducibility over time.
"""
import streamlit as st
import requests

def render_replay_interface():
    """Time-travel query execution interface."""
    st.markdown("### ⏳ Time-Travel Query Replay")
    historical_query = st.text_input("Enter historical query to replay:", placeholder="What are the system requirements?")

    if st.button("Replay Query"):
        if historical_query.strip():
            try:
                res = requests.post(
                    "http://localhost:8000/api/v1/replay/run",
                    data={"query": historical_query}
                ).json()
                st.success("Replay Completed!")
                st.write("**Replayed Answer:**", res.get("replayed_answer", ""))
                st.json(res)
            except Exception as e:
                st.error(f"Replay execution failed: {str(e)}")