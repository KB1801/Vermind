"""
Pytest Verification Suite for Document Loaders & Chunkers.
"""
import pytest
from backend.app.ingestion.chunker import document_chunker
from backend.app.chunking.semantic import semantic_chunker
from backend.app.chunking.sentence_window import sentence_window_chunker
from backend.app.chunking.parent_child import parent_child_chunker

SAMPLE_TEXT = (
    "VeriMind is a enterprise Hybrid RAG platform. "
    "It uses 4 custom machine learning models to eliminate hallucination. "
    "The system runs on an 8GB RAM laptop using Groq API and Qdrant vector database. "
    "Atomic claim decomposition ensures evidence verification for high integrity."
)

def test_fixed_size_chunker():
    chunks = document_chunker.split_text(SAMPLE_TEXT, doc_id="doc_1")
    assert len(chunks) > 0
    assert "chunk_id" in chunks[0]
    assert chunks[0]["doc_id"] == "doc_1"

def test_semantic_chunker():
    chunks = semantic_chunker.split_text(SAMPLE_TEXT, doc_id="doc_sem")
    assert len(chunks) > 0
    assert chunks[0]["metadata"]["chunk_type"] == "semantic"

def test_sentence_window_chunker():
    chunks = sentence_window_chunker.split_text(SAMPLE_TEXT, doc_id="doc_win")
    assert len(chunks) > 0
    assert "full_window_text" in chunks[0]["metadata"]

def test_parent_child_chunker():
    chunks = parent_child_chunker.split_text(SAMPLE_TEXT, doc_id="doc_pc")
    assert len(chunks) > 0
    assert "parent_id" in chunks[0]["metadata"]