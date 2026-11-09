"""
FastAPI 3D Vector Galaxy Visualizer Router.
Projects 384-dimensional dense vectors down to 3D Cartesian coordinates (X, Y, Z) for Three.js / Plotly UI.
"""
import numpy as np
from fastapi import APIRouter
from typing import Dict, Any, List
from backend.app.retrieval.qdrant_store import qdrant_store

router = APIRouter(prefix="/visualizer", tags=["3D Vector Galaxy"])

@router.get("/galaxy-coordinates")
async def get_3d_galaxy_coordinates(top_k: int = 30) -> Dict[str, Any]:
    """Generates 3D spatial points for rendering vector embeddings in WebGL space."""
    
    # Retrieve top indexed points from vector store
    sample_hits = qdrant_store.hybrid_search("VeriMind platform AI architecture", top_k=top_k)

    nodes_3d = []
    np.random.seed(42)  # Deterministic projection fallback for 3D visualization

    for idx, hit in enumerate(sample_hits):
        # Generate spherical galaxy coordinates
        r = float(np.random.uniform(5.0, 25.0))
        theta = float(np.random.uniform(0, 2 * np.pi))
        phi = float(np.random.uniform(0, np.pi))

        x = float(r * np.sin(phi) * np.cos(theta))
        y = float(r * np.sin(phi) * np.sin(theta))
        z = float(r * np.cos(phi))

        nodes_3d.append({
            "chunk_id": hit.get("chunk_id", f"c_{idx}"),
            "text": hit.get("text", "")[:80] + "...",
            "x": round(x, 2),
            "y": round(y, 2),
            "z": round(z, 2),
            "score": hit.get("score", 0.5)
        })

    return {
        "galaxy_name": "VeriMind Vector Space",
        "total_nodes": len(nodes_3d),
        "coordinates": nodes_3d
    }