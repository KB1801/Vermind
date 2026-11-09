"""
VeriMind Domain Custom Exceptions Hierarchy.
Enables exact error handling across API endpoints and pipeline routines.
"""

class VeriMindException(Exception):
    """Base exception class for all VeriMind platform operational errors."""
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class SecurityViolationException(VeriMindException):
    """Raised when prompt injection, unauthorized PII, or RBAC breach is detected."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=403, details=details)

class RetrievalEngineException(VeriMindException):
    """Raised when Qdrant, FastEmbed, or RRF hybrid retrieval components fail."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=502, details=details)

class ModelInferenceException(VeriMindException):
    """Raised when Groq, Ollama, PyTorch I-JEPA, or local LLM inference fails."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=503, details=details)

class DocumentIngestionException(VeriMindException):
    """Raised when document parsing, OCR, or semantic chunking fails."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=422, details=details)

class EvaluationException(VeriMindException):
    """Raised during IR benchmark or regression test assertion failures."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=400, details=details)