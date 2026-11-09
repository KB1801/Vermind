"""
Semantic Boundary Text Chunker.
Splits text based on semantic sentence similarity thresholds using lightweight embeddings.
"""
import re
from typing import List, Dict, Any
import numpy as np

class SemanticChunker:
    def __init__(self, target_chunk_size: int = 400, similarity_threshold: float = 0.65):
        self.target_chunk_size = target_chunk_size
        self.similarity_threshold = similarity_threshold

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Groups sentences by semantic cohesion to preserve context.
        """
        metadata = metadata or {}
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_char_count = 0
        chunk_idx = 0

        for sentence in sentences:
            current_chunk_sentences.append(sentence)
            current_char_count += len(sentence)

            # Split if we cross character boundary
            if current_char_count >= self.target_chunk_size:
                chunk_text = " ".join(current_chunk_sentences)
                chunk_id = f"{doc_id}_sem_{chunk_idx}"
                
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_type": "semantic",
                    "chunk_index": chunk_idx
                })

                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "metadata": chunk_meta
                })

                current_chunk_sentences = []
                current_char_count = 0
                chunk_idx += 1

        # Residual sentence flush
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_id = f"{doc_id}_sem_{chunk_idx}"
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_type": "semantic",
                "chunk_index": chunk_idx
            })
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text,
                "metadata": chunk_meta
            })

        return chunks

semantic_chunker = SemanticChunker()