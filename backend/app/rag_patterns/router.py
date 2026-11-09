"""
Master RAG Auto-Router Engine.
Dispatches queries to optimal RAG patterns using native ML intent predictions.
"""
from typing import Dict, Any
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.rag_patterns.naive_rag import naive_rag
from backend.app.rag_patterns.retrieve_rerank import retrieve_rerank_rag
from backend.app.rag_patterns.corrective_rag import corrective_rag
from backend.app.rag_patterns.self_rag import self_rag
from backend.app.rag_patterns.adaptive_rag import adaptive_rag
from backend.app.rag_patterns.hyde_rag import hyde_rag
from backend.app.rag_patterns.rag_fusion import rag_fusion
from backend.app.rag_patterns.cag import cag_pattern
from backend.app.ml.intent_classifier import intent_classifier
from backend.app.schemas import QueryResponse
from backend.app.core.logger import logger

class RAGRouter:
    def __init__(self):
        self.registry: Dict[str, BaseRAGPattern] = {
            "naive": naive_rag,
            "retrieve_rerank": retrieve_rerank_rag,
            "crag": corrective_rag,
            "self_rag": self_rag,
            "adaptive_rag": adaptive_rag,
            "hyde": hyde_rag,
            "rag_fusion": rag_fusion,
            "cag": cag_pattern
        }

    async def route_and_execute(
        self,
        query: str,
        requested_pattern: str = "auto",
        tenant_id: str = "default_tenant",
        top_k: int = 5
    ) -> QueryResponse:

        pattern_to_use = requested_pattern

        if requested_pattern == "auto":
            intent, confidence = intent_classifier.predict(query)
            logger.info(f"Auto-Router classified intent: '{intent}' (conf: {confidence:.2f})")

            if intent == "summary":
                pattern_to_use = "rag_fusion"
            elif intent == "comparison":
                pattern_to_use = "adaptive_rag"
            elif intent == "unsupported":
                pattern_to_use = "crag"
            else:
                pattern_to_use = "retrieve_rerank"

        selected_pattern = self.registry.get(pattern_to_use, retrieve_rerank_rag)
        logger.info(f"Auto-Router executing strategy: [{selected_pattern.name}]")

        return await selected_pattern.execute(query=query, tenant_id=tenant_id, top_k=top_k)

rag_router = RAGRouter()