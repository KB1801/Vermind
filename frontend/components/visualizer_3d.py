"""
3D Plotly WebGL Vector Galaxy Visualizer Component.
Renders 384-dimensional FastEmbed BGE-small embeddings as interactive 3D spatial points.
"""
import streamlit as st
import plotly.express as pd_plotly
import pandas as pd
from typing import List, Dict, Any

def render_3d_vector_galaxy(coordinates: List[Dict[str, Any]]):
    """Renders Plotly 3D scatter plot of vector galaxy points."""
    st.markdown("### 🌌 3D Vector Embedding Space Galaxy")
    if not coordinates:
        st.info("No 3D vector coordinates loaded.")
        return

    df = pd.DataFrame(coordinates)
    fig = pd_plotly.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='score',
        hover_name='chunk_id',
        hover_data=['text'],
        title='VeriMind Latent Space Vector Coordinates',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig, use_container_width=True)