"""
Prompt & Context Window Assembly Engine.
Constructs optimal prompt payloads respecting model token budgets.
"""
from typing import List, Dict, Any

class ContextEngineer:
    def construct_prompt(self, query: str, chunks: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
        context_blocks = []
        current_len = 0

        for c in chunks:
            block = f"Source [{c.get('chunk_id', 'ref')}]: {c['text']}"
            if current_len + len(block.split()) > max_tokens:
                break
            context_blocks.append(block)
            current_len += len(block.split())

        formatted_context = "\n\n".join(context_blocks)
        return f"Context:\n{formatted_context}\n\nQuestion: {query}\nAnswer:"

context_engineer = ContextEngineer()