"""
Cross-Encoder MiniLM Re-Ranker.
Re-scores retrieved candidate chunks using query-chunk attention interactions.
"""
from typing import List, Dict, Any
from backend.app.config import settings
from backend.app.core.logger import logger

class CrossEncoderReranker:
    def __init__(self):
        self.model = None
        self._init_reranker()

    def _init_reranker(self):
        if not settings.ENABLE_CROSS_ENCODER:
            logger.info("Cross-Encoder re-ranker disabled in config settings.")
            return

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading Cross-Encoder model: {settings.RERANKER_MODEL}")
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("✓ Cross-Encoder Re-ranker loaded successfully.")
        except Exception as e:
            logger.warning(f"! Cross-Encoder unavailable ({str(e)}). Active fallback: Heuristic token overlap re-ranking.")
            self.model = None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-ranks top candidate chunks using neural Cross-Encoder scores.
        """
        if not candidates:
            return []

        if self.model:
            try:
                pairs = [[query, c["text"]] for c in candidates]
                scores = self.model.predict(pairs)

                for idx, c in enumerate(candidates):
                    c["rerank_score"] = float(scores[idx])
                    c["score"] = float(scores[idx])  # Override composite rank score

                candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
                return candidates[:top_k]
            except Exception as e:
                logger.error(f"Cross-Encoder prediction failed: {str(e)}")

        # Fallback: Token overlap score refinement
        q_words = set(query.lower().split())
        for c in candidates:
            c_words = set(c["text"].lower().split())
            overlap = len(q_words.intersection(c_words)) / max(1, len(q_words))
            c["rerank_score"] = float(c.get("score", 0.5) + overlap * 0.5)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

reranker = CrossEncoderReranker()