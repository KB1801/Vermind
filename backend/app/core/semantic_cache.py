"""
Vector Cosine Similarity Semantic Cache Engine.
Caches and retrieves query answers when similarity score >= 0.92.
"""
import numpy as np
from typing import Optional, Dict, Any
from backend.app.retrieval.unified_embedder import unified_embedder
from backend.app.core.logger import logger

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.cache_entries = []  # List of dicts: {"query": str, "vector": List[float], "answer": str}

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def get_semantic_hit(self, query: str) -> Optional[Dict[str, Any]]:
        dense_vecs, _ = unified_embedder.embed_queries([query])
        if not dense_vecs or not self.cache_entries:
            return None

        q_vec = np.array(dense_vecs[0])
        best_sim = 0.0
        best_hit = None

        for entry in self.cache_entries:
            sim = self._cosine_similarity(q_vec, np.array(entry["vector"]))
            if sim > best_sim:
                best_sim = sim
                best_hit = entry

        if best_sim >= self.threshold and best_hit:
            logger.info(f"✓ Semantic Cache HIT! (Similarity: {best_sim:.4f})")
            return {"answer": best_hit["answer"], "similarity": round(best_sim, 4)}

        return None

    def store(self, query: str, answer: str):
        dense_vecs, _ = unified_embedder.embed_queries([query])
        if dense_vecs:
            self.cache_entries.append({
                "query": query,
                "vector": dense_vecs[0],
                "answer": answer
            })

semantic_cache = SemanticCache()