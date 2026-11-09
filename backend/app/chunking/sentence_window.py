"""
Sentence-Window Chunker for Precise RAG Retrieval.
Indexes single focus sentences while linking surrounding context windows (k=2 sentences).
"""
import re
from typing import List, Dict, Any

class SentenceWindowChunker:
    def __init__(self, window_size: int = 2):
        self.window_size = window_size

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        for i, focus_sentence in enumerate(sentences):
            start_idx = max(0, i - self.window_size)
            end_idx = min(len(sentences), i + self.window_size + 1)
            
            context_window = " ".join(sentences[start_idx:end_idx])
            chunk_id = f"{doc_id}_win_{i}"

            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_type": "sentence_window",
                "focus_sentence": focus_sentence,
                "window_size": self.window_size
            })

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": focus_sentence,  # Embedded key
                "metadata": {**chunk_meta, "full_window_text": context_window}
            })

        return chunks

sentence_window_chunker = SentenceWindowChunker()