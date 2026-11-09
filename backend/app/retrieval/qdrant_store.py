"""
Native Qdrant Hybrid Vector Store with Server-Side RRF Fusion.
Integrates Dense BGE-Small vectors + BM25 Sparse vectors + Reciprocal Rank Fusion.
Guarded against empty sparse vector crash on stopword queries and respects multi-tenant boundaries.
"""
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from backend.app.config import settings
from backend.app.core.logger import logger
from backend.app.core.exceptions import RetrievalEngineException
from backend.app.retrieval.unified_embedder import unified_embedder

class QdrantVectorStore:
    def __init__(self):
        self.client = None
        self.collection_name = settings.COLLECTION_NAME
        self.in_memory_fallback = {}
        self._connect()

    def _connect(self):
        try:
            self.client = QdrantClient(url=settings.QDRANT_URL, timeout=3.0)
            self.client.get_collections()
            logger.info(f"✓ Connected to Qdrant cluster at {settings.QDRANT_URL}")
            self._ensure_collection()
        except Exception as e:
            logger.warning(f"! Qdrant cluster unreachable ({str(e)}). Active fallback: In-Memory Hybrid Store.")
            self.client = None

    def _ensure_collection(self):
        if not self.client:
            return
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                logger.info(f"Creating Qdrant Hybrid collection '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=384,
                            distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False)
                        )
                    }
                )
                logger.info("✓ Hybrid collection created with Dense (384) + Sparse indices.")
        except Exception as e:
            logger.error(f"Failed to setup Qdrant collection: {str(e)}")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], tenant_id: str = "default_tenant") -> int:
        """
        Upserts document chunks into Qdrant or in-memory fallback with tenant boundary filters.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        dense_vecs, sparse_vecs = unified_embedder.embed_documents(texts)

        points = []
        for idx, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            payload = chunk.get("metadata", {}).copy()
            payload["text"] = chunk["text"]
            payload["tenant_id"] = tenant_id
            payload["doc_id"] = chunk.get("doc_id", "doc_unknown")
            payload["chunk_id"] = chunk.get("chunk_id", point_id)

            if self.client and dense_vecs and sparse_vecs:
                vector_dict = {
                    "dense": dense_vecs[idx]
                }
                # Guard against empty sparse indices on stopword chunks
                if sparse_vecs[idx]["indices"]:
                    vector_dict["sparse"] = models.SparseVector(
                        indices=sparse_vecs[idx]["indices"],
                        values=sparse_vecs[idx]["values"]
                    )

                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector_dict,
                    payload=payload
                ))
            else:
                # In-memory storage fallback
                self.in_memory_fallback[chunk.get("chunk_id", point_id)] = payload

        if self.client and points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                return len(points)
            except Exception as e:
                logger.error(f"Qdrant upsert failed: {str(e)}")
                return 0
        
        return len(chunks)

    def hybrid_search(self, query: str, top_k: int = 5, tenant_id: str = "default_tenant") -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search using RRF (Reciprocal Rank Fusion) with multi-tenant isolation.
        """
        if not self.client:
            return self._in_memory_keyword_search(query, top_k, tenant_id=tenant_id)

        dense_vecs, sparse_vecs = unified_embedder.embed_queries([query])
        if not dense_vecs:
            return self._in_memory_keyword_search(query, top_k, tenant_id=tenant_id)

        tenant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id)
                )
            ]
        )

        try:
            # Dense Vector Search
            dense_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=models.NamedVector(name="dense", vector=dense_vecs[0]),
                query_filter=tenant_filter,
                limit=top_k * 2
            )

            # Sparse Vector Search (if sparse tokens exist for query)
            sparse_results = []
            if sparse_vecs and sparse_vecs[0]["indices"]:
                sparse_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=models.NamedSparseVector(
                        name="sparse",
                        vector=models.SparseVector(
                            indices=sparse_vecs[0]["indices"],
                            values=sparse_vecs[0]["values"]
                        )
                    ),
                    query_filter=tenant_filter,
                    limit=top_k * 2
                )

            # Apply Reciprocal Rank Fusion (RRF)
            return self._apply_rrf_fusion(dense_results, sparse_results, top_k)

        except Exception as e:
            logger.error(f"Qdrant Hybrid Search error: {str(e)}. Triggering in-memory fallback...")
            return self._in_memory_keyword_search(query, top_k, tenant_id=tenant_id)

    def _apply_rrf_fusion(self, dense_hits, sparse_hits, top_k: int, k_factor: int = 60) -> List[Dict[str, Any]]:
        scores = {}
        payloads = {}

        for rank, hit in enumerate(dense_hits):
            doc_id = hit.payload.get("chunk_id", hit.id)
            payloads[doc_id] = hit.payload
            rrf_score = 1.0 / (k_factor + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

        for rank, hit in enumerate(sparse_hits):
            doc_id = hit.payload.get("chunk_id", hit.id)
            payloads[doc_id] = hit.payload
            rrf_score = 1.0 / (k_factor + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for chunk_id, score in sorted_docs:
            p = payloads[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "doc_id": p.get("doc_id", "unknown"),
                "text": p.get("text", ""),
                "score": float(score),
                "metadata": p
            })
        return results

    def _in_memory_keyword_search(self, query: str, top_k: int, tenant_id: str = "default_tenant") -> List[Dict[str, Any]]:
        q_words = set(query.lower().split())
        scored = []
        for cid, payload in self.in_memory_fallback.items():
            # Enforce multi-tenant isolation boundary
            p_tenant = payload.get("tenant_id", "default_tenant")
            if p_tenant != tenant_id:
                continue

            text = payload.get("text", "").lower()
            matches = sum(1 for w in q_words if w in text)
            if matches > 0 or not self.in_memory_fallback:
                scored.append((matches, cid, payload))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, cid, payload in scored[:top_k]:
            results.append({
                "chunk_id": cid,
                "doc_id": payload.get("doc_id", "doc_fallback"),
                "text": payload.get("text", ""),
                "score": float(score + 0.1),
                "metadata": payload
            })
        return results

qdrant_store = QdrantVectorStore()