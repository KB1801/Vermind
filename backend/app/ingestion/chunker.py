"""
Fixed-Size & Sliding-Window Text Chunker.
Splits text into uniform overlapping chunks with precise token/character boundaries.
"""
from typing import List, Dict, Any
import uuid

class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Splits raw text into sliding-window text chunks.
        """
        metadata = metadata or {}
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        
        chunk_idx = 0
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_id = f"{doc_id}_chunk_{chunk_idx}"
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": chunk_idx,
                "word_count": len(chunk_words)
            })

            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_text,
                "metadata": chunk_meta
            })
            chunk_idx += 1

            if i + self.chunk_size >= len(words):
                break

        return chunks

document_chunker = DocumentChunker()