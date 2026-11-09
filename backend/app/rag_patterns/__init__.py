"""
VeriMind RAG Patterns Package.
Exports all native RAG architectural patterns and the RAG Auto-Router.
"""
from backend.app.rag_patterns.base import BaseRAGPattern
from backend.app.rag_patterns.naive_rag import NaiveRAG
from backend.app.rag_patterns.retrieve_rerank_rag import RetrieveRerankRAG
from backend.app.rag_patterns.corrective_rag import CorrectiveRAG
from backend.app.rag_patterns.self_rag import SelfRAG
from backend.app.rag_patterns.adaptive_rag import AdaptiveRAG
from backend.app.rag_patterns.hyde_rag import HyDERAG
from backend.app.rag_patterns.rag_fusion import RAGFusion
from backend.app.rag_patterns.long_context_rag import LongContextRAG
from backend.app.rag_patterns.compression_rag import CompressionRAG
from backend.app.rag_patterns.query_decomposition_rag import QueryDecompositionRAG
from backend.app.rag_patterns.iterative_rag import IterativeRAG
from backend.app.rag_patterns.react_rag import ReActRAG
from backend.app.rag_patterns.sql_rag import SQLRAG
from backend.app.rag_patterns.raptor_rag import RaptorRAG
from backend.app.rag_patterns.graph_rag import GraphRAG
from backend.app.rag_patterns.cag import CacheAugmentedGeneration
from backend.app.rag_patterns.router import RAGRouter

__all__ = [
    "BaseRAGPattern",
    "NaiveRAG",
    "RetrieveRerankRAG",
    "CorrectiveRAG",
    "SelfRAG",
    "AdaptiveRAG",
    "HyDERAG",
    "RAGFusion",
    "LongContextRAG",
    "CompressionRAG",
    "QueryDecompositionRAG",
    "IterativeRAG",
    "ReActRAG",
    "SQLRAG",
    "RaptorRAG",
    "GraphRAG",
    "CacheAugmentedGeneration",
    "RAGRouter"
]