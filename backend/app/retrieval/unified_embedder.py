"""
Unified Embedder Engine powered by Local ONNX FastEmbed.
Computes Dense (BAAI/bge-small-en-v1.5) and Sparse (BM25) vector representations.
Includes honest degradation without zero-vector hacks.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from backend.app.config import settings
from backend.app.core.logger import logger
from backend.app.core.exceptions import RetrievalEngineException

class UnifiedEmbedder:
    def __init__(self):
        self.dense_model = None
        self.sparse_model = None
        self._init_models()

    def _init_models(self):
        try:
            from fastembed import TextEmbedding, SparseTextEmbedding
            logger.info(f"Loading Local FastEmbed Dense Model: {settings.DENSE_MODEL}")
            self.dense_model = TextEmbedding(model_name=settings.DENSE_MODEL)
            
            logger.info("Loading Local FastEmbed BM25 Sparse Model")
            self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
            logger.info("✓ FastEmbed ONNX Embedder initialized successfully.")
        except Exception as e:
            logger.warning(f"! Local FastEmbed ONNX models failed to initialize ({str(e)}). Active fallback: Extractive BM25 mode.")
            self.dense_model = None
            self.sparse_model = None

    def embed_queries(self, queries: List[str]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """
        Generates dense (384-dim) and sparse (indices/values) representations for query list.
        """
        if not queries:
            return [], []

        dense_embeddings = []
        sparse_embeddings = []

        if self.dense_model and self.sparse_model:
            try:
                # FastEmbed returns generators
                dense_gen = self.dense_model.embed(queries)
                dense_embeddings = [emb.tolist() for emb in dense_gen]

                sparse_gen = self.sparse_model.embed(queries)
                for s_emb in sparse_gen:
                    sparse_embeddings.append({
                        "indices": s_emb.indices.tolist(),
                        "values": s_emb.values.tolist()
                    })
                return dense_embeddings, sparse_embeddings
            except Exception as e:
                logger.error(f"FastEmbed embedding computation error: {str(e)}")

        # Honest Fallback: Fake zero-vector degradation is strictly forbidden.
        # Return empty list to signal honest BM25 degradation mode to vector store.
        return [], []

    def embed_documents(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """
        Generates dense and sparse embeddings for document chunks.
        """
        return self.embed_queries(texts)

unified_embedder = UnifiedEmbedder()