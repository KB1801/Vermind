"""
FastAPI Knowledge Mesh & Graph Provenance Router.
Generates node-edge network graphs linking queries, chunks, and claims for WebGL 3D visualizers.
"""
from fastapi import APIRouter, Form
from typing import Dict, Any, List
from backend.app.retrieval.qdrant_store import qdrant_store

router = APIRouter(prefix="/graph", tags=["Knowledge Mesh Graph"])

@router.post("/provenance")
async def get_provenance_graph(query: str = Form(...), tenant_id: str = Form("default_tenant")) -> Dict[str, Any]:
    chunks = qdrant_store.hybrid_search(query, top_k=5, tenant_id=tenant_id)

    nodes = [{"id": "query_root", "label": query, "type": "query", "group": 1}]
    edges = []

    for c in chunks:
        cid = c["chunk_id"]
        nodes.append({"id": cid, "label": c["text"][:40] + "...", "type": "chunk", "group": 2})
        edges.append({"source": "query_root", "target": cid, "weight": float(c["score"])})

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges)
    }