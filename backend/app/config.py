"""
VeriMind Global Pydantic Settings & Environment Manager.
Optimized for 8GB RAM laptops with automatic dual-mode fallback setup.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # System Info
    PROJECT_NAME: str = "VeriMind"
    VERSION: str = "2.2.0"
    API_V1_STR: str = "/api/v1"
    
    # Dual Mode Execution
    VERIMIND_OFFLINE_MODE: bool = False
    
    # Primary API Keys (Groq llama-3.3-70b-versatile default)
    GROQ_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Jev Decision Brain
    JEV_API_KEY: Optional[str] = None
    JEV_BASE_URL: str = "https://api.typesafe.ai/v1"
    JEV_MODEL: str = "jev-latest"
    ENABLE_JEV_LOCAL_BRAIN: bool = True
    
    # Local Ollama Fallback
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:1b"
    
    # Infrastructure & Databases
    QDRANT_URL: str = "http://localhost:6333"
    REDIS_URL: str = "redis://localhost:6379/0"
    COLLECTION_NAME: str = "verimind_docs"
    
    # Native Embedding & ML Models
    DENSE_MODEL: str = "BAAI/bge-small-en-v1.5"
    SPARSE_MODEL: str = "Qdrant/bm25"
    RERANKER_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    
    # ML Weights & Latent World Model Settings
    ENABLE_JEPA_WORLD_MODEL: bool = True
    JEPA_WEIGHT: float = 0.30
    ENABLE_CROSS_ENCODER: bool = True
    
    # Security & Governance
    JWT_SECRET: str = "verimind_super_secret_jwt_key_32bytes_minimum"
    PII_MASKING_ENABLED: bool = True
    MAX_QUERY_LENGTH: int = 2048
    
    # Server Host
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()