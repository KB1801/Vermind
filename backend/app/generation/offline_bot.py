"""
Offline Extractive AI Engine.
Synthesizes answers from retrieved chunks using TF-IDF sentence overlap scoring.
Ensures VeriMind operates 100% offline without API keys or local LLM setup.
"""
import re
from typing import List, Dict, Any

class ExtractiveOfflineBot:
    def synthesize_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Extracts top informative sentences directly from retrieved evidence.
        """
        if not chunks:
            return "No relevant context found in database to answer your query."

        query_words = set(re.findall(r'\w+', query.lower()))
        scored_sentences = []

        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", "chunk_ref")
            sentences = re.split(r'(?<=[.!?])\s+', text)

            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 15:
                    continue
                s_words = set(re.findall(r'\w+', s_clean.lower()))
                overlap = len(query_words.intersection(s_words))
                scored_sentences.append((overlap, s_clean, chunk_id))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = []
        seen = set()

        for score, sentence, cid in scored_sentences:
            if sentence not in seen:
                top_sentences.append(f"{sentence} [{cid}]")
                seen.add(sentence)
            if len(top_sentences) >= 4:
                break

        if not top_sentences:
            return f"Based on retrieved sources: {chunks[0]['text'][:300]}..."

        return " ".join(top_sentences)

offline_bot = ExtractiveOfflineBot()