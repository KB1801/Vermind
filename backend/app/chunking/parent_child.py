"""
Hierarchical Parent-Child Document Chunker.
Maps small granular child chunks to broad parent sections to eliminate lost context.
"""
from typing import List, Dict, Any
from backend.app.ingestion.chunker import DocumentChunker

class ParentChildChunker:
    def __init__(self, parent_size: int = 800, child_size: int = 200):
        self.parent_chunker = DocumentChunker(chunk_size=parent_size, chunk_overlap=100)
        self.child_chunker = DocumentChunker(chunk_size=child_size, chunk_overlap=30)

    def split_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        parents = self.parent_chunker.split_text(text, doc_id, metadata)
        
        all_child_chunks = []
        for parent_idx, parent in enumerate(parents):
            parent_id = f"{doc_id}_parent_{parent_idx}"
            parent_text = parent["text"]
            
            children = self.child_chunker.split_text(parent_text, parent_id, metadata)
            for child_idx, child in enumerate(children):
                child_id = f"{doc_id}_p{parent_idx}_c{child_idx}"
                child_meta = metadata.copy()
                child_meta.update({
                    "chunk_id": child_id,
                    "doc_id": doc_id,
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "chunk_type": "parent_child"
                })

                all_child_chunks.append({
                    "chunk_id": child_id,
                    "doc_id": doc_id,
                    "text": child["text"],
                    "metadata": child_meta
                })

        return all_child_chunks

parent_child_chunker = ParentChildChunker()