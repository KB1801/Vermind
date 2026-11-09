"""
VeriMind Platform Standard Data Schemas (Pydantic v2).
Defines request/response contracts for high-precision RAG, atomic verification, and MCP.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import re

class TenantMixin(BaseModel):
    tenant_id: str = Field(default="default_tenant", description="Multi-tenant boundary identifier")

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("tenant_id must contain only alphanumeric characters, underscores, or hyphens.")
        return v

class QueryRequest(TenantMixin):
    query: str = Field(..., min_length=1, max_length=2048, description="User search or question text")
    pattern: str = Field(default="auto", description="RAG Pattern: naive, hybrid, crag, self_rag, flagship, auto")
    top_k: int = Field(default=5, ge=1, le=50, description="Top K evidence chunks to retrieve")
    stream: bool = Field(default=False, description="Enable SSE streaming mode")

class VectorSearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    jepa_latent_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ClaimVerificationResult(BaseModel):
    claim: str
    verdict: str = Field(..., description="SUPPORTED | CONTRADICTED | NEUTRAL")
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    reasoning: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    pattern_used: str
    retrieved_chunks: List[VectorSearchResult]
    claims: List[ClaimVerificationResult] = Field(default_factory=list)
    evidence_integrity_index: float = Field(default=1.0, ge=0.0, le=1.0)
    execution_time_ms: float
    offline_fallback_active: bool = False
    tenant_id: str

class DocumentIngestRequest(TenantMixin):
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunking_strategy: str = Field(default="semantic", description="semantic | sentence_window | parent_child")

class DocumentIngestResponse(BaseModel):
    doc_id: str
    chunks_created: int
    total_tokens: int
    status: str = "indexed"

class SystemStatusResponse(BaseModel):
    status: str
    version: str
    qdrant_connected: bool
    redis_connected: bool
    offline_mode: bool
    active_llm_provider: str
    models_loaded: Dict[str, bool]