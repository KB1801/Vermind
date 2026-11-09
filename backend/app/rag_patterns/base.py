"""
Base Abstract Class for all VeriMind RAG Architectural Patterns.
Enforces consistent interfaces and timing telemetry across all execution strategies.
"""
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from backend.app.schemas import QueryResponse

class BaseRAGPattern(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, query: str, tenant_id: str = "default_tenant", top_k: int = 5) -> QueryResponse:
        """Executes the specific RAG strategy and returns a QueryResponse schema."""
        pass