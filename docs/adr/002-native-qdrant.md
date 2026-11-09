# ADR 002: Native Qdrant Hybrid Search with RRF Fusion

## Context
Combining dense semantic vectors and sparse BM25 lexical vectors requires a unified vector engine to prevent dual-database synchronization overhead.

## Decision
Adopt Qdrant native hybrid search with server-side Reciprocal Rank Fusion (RRF) and Local FastEmbed ONNX embeddings.

## Status
Accepted and Implemented in `backend/app/retrieval/qdrant_store.py`.