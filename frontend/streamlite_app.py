"""
VeriMind Master Streamlit UI Dashboard Entry Point.
Multi-page full-stack interface featuring Dual-Engine Chat, 3D Vector Galaxy, and ML Analytics.
"""
import streamlit as st
import requests

st.set_page_config(
    page_title="VeriMind Enterprise AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1E88E5; font-weight: 700; }
    .sub-header { font-size: 1.1rem; color: #555555; }
    .card { background-color: #F8F9FA; padding: 20px; border-radius: 10px; border-left: 5px solid #1E88E5; }
</style>
""", unsafe_allow_dict=True)

st.title("🧠 VeriMind Enterprise Platform")
st.caption("Hybrid RAG + Agentic AI | Atomic Claim Verification | Dual-Engine Offline Resilience")

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Dual-Engine Chat", "🌌 3D Vector Galaxy", "🔬 ML & Benchmark Dashboard", "⚙️ System Telemetry"])

with tab1:
    st.subheader("High-Precision Hybrid Retrieval & Generation")
    query_text = st.text_input("Enter your question or query:", placeholder="What are the core features of VeriMind?")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        pattern = st.selectbox("RAG Pattern", ["auto", "retrieve_rerank", "crag", "self_rag", "rag_fusion", "cag"])
    
    if st.button("Execute Search & Synthesis", type="primary"):
        if query_text.strip():
            with st.spinner("Processing through VeriMind Auto-Router..."):
                try:
                    res = requests.post(
                        "http://localhost:8000/api/v1/rag/query",
                        json={"query": query_text, "pattern": pattern, "top_k": 5}
                    ).json()
                    
                    st.markdown("### Answer")
                    st.write(res.get("answer", ""))
                    
                    st.metric("Evidence Integrity Index (EII)", f"{res.get('evidence_integrity_index', 1.0)*100:.1f}%")
                    st.info(f"Pattern Executed: **{res.get('pattern_used', 'auto')}** | Latency: {res.get('execution_time_ms', 0):.2f}ms")
                    
                    with st.expander("Retrieved Context Sources"):
                        st.json(res.get("retrieved_chunks", []))
                except Exception as e:
                    st.error(f"Backend Server Unreachable: {str(e)}. Make sure 'python run.py' is running!")

with tab2:
    st.subheader("3D Vector Embedding Space Galaxy")
    st.markdown("Visualizing 384-dimensional FastEmbed BGE-small embeddings in 3D WebGL space.")
    if st.button("Fetch Vector Galaxy Coordinates"):
        try:
            galaxy_data = requests.get("http://localhost:8000/api/v1/visualizer/galaxy-coordinates").json()
            st.json(galaxy_data)
        except Exception as e:
            st.error(f"Error fetching 3D galaxy data: {str(e)}")

with tab3:
    st.subheader("Native ML Models & Benchmark Analytics")
    st.markdown("Inspect Scikit-Learn Intent Classifier and PyTorch I-JEPA Latent World Model metrics.")
    if st.button("Run Live IR Benchmark Evaluation"):
        try:
            bench_res = requests.post("http://localhost:8000/api/v1/ml/evaluate-ir").json()
            st.success("Benchmark Execution Completed!")
            st.json(bench_res)
        except Exception as e:
            st.error(f"Error running benchmark: {str(e)}")

with tab4:
    st.subheader("System Telemetry & Memory Diagnostics")
    try:
        status = requests.get("http://localhost:8000/api/v1/admin/status").json()
        st.json(status)
    except Exception as e:
        st.warning("Backend offline.")