"""
Streamlit Machine Learning Dashboard.
Displays Scikit-Learn intent classification performance and PyTorch I-JEPA world model metrics.
"""
import streamlit as st
import requests

def render_ml_dashboard():
    """Renders Machine Learning Analytics and Intent Classifier Testing UI."""
    st.markdown("### 🔬 Native ML Models Dashboard")
    
    test_query = st.text_input("Test Scikit-Learn Intent Classifier:", value="Compare BM25 vs Vector search")
    if st.button("Classify Query Intent"):
        try:
            res = requests.post(
                "http://localhost:8000/api/v1/ml/intent",
                data={"query": test_query}
            ).json()
            st.info(f"Predicted Class: **{res.get('predicted_intent', 'unknown')}** | Confidence: **{res.get('confidence', 0.0)*100:.1f}%**")
        except Exception as e:
            st.error(f"Intent prediction error: {str(e)}")