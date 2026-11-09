# ADR 001: RAG Auto-Router Architecture

## Context
Different queries require different retrieval and generation strategies. Naive vector search underperforms on comparative or summary questions.

## Decision
Implement a native Scikit-Learn TF-IDF + Logistic Regression Intent Classifier that routes incoming queries to optimal RAG patterns (Adaptive, Self-RAG, CRAG, HyDE, RAG-Fusion, CAG).

## Status
Accepted and Implemented in `backend/app/rag_patterns/router.py`.