"""
Streamlit Provenance Graph & Knowledge Mesh Visualizer.
Renders node-edge networks using Graphviz / NetworkX.
"""
import streamlit as st
from typing import Dict, Any

def render_provenance_graph(graph_data: Dict[str, Any]):
    """Renders Graphviz node-edge network graph of retrieved chunks."""
    st.markdown("### 🕸️ Knowledge Mesh & Provenance Graph")
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        st.info("No provenance graph data available.")
        return

    dot_lines = ["digraph ProvenanceGraph {", "  rankdir=LR;", "  node [shape=box, style=filled, fillcolor=lightblue];"]
    
    for node in nodes:
        nid = node["id"].replace("-", "_")
        label = node["label"].replace('"', "'")
        ntype = node.get("type", "chunk")
        color = "#1E88E5" if ntype == "query" else "#E3F2FD"
        dot_lines.append(f'  "{nid}" [label="{label}", fillcolor="{color}"];')

    for edge in edges:
        src = edge["source"].replace("-", "_")
        tgt = edge["target"].replace("-", "_")
        weight = edge.get("weight", 1.0)
        dot_lines.append(f'  "{src}" -> "{tgt}" [label="score: {weight:.2f}"];')

    dot_lines.append("}")
    dot_code = "\n".join(dot_lines)

    try:
        st.graphviz_chart(dot_code)
    except Exception as e:
        st.warning(f"Graphviz rendering fallback: {str(e)}")
        st.json(graph_data)