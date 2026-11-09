"""
VeriMind Global Pydantic Settings & Environment Manager.
Optimized for 8GB RAM laptops with automatic dual-mode fallback setup.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    # System Info
    PROJECT_NAME: str = "VeriMind"
    VERSION: str = "2.2.0"
    API_V1_STR: str = "/api/v1"
    
    # Dual Mode Execution
    VERIMIND_OFFLINE_MODE: bool = Field(default=False, env="VERIMIND_OFFLINE_MODE")
    
    # Primary API Keys (Groq llama-3.3-70b-versatile default)
    GROQ_API_KEY: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Jev Decision Brain
    JEV_API_KEY: Optional[str] = Field(default=None, env="JEV_API_KEY")
    JEV_BASE_URL: str = Field(default="https://api.typesafe.ai/v1", env="JEV_BASE_URL")
    JEV_MODEL: str = Field(default="jev-latest", env="JEV_MODEL")
    ENABLE_JEV_LOCAL_BRAIN: bool = Field(default=True, env="ENABLE_JEV_LOCAL_BRAIN")
    
    # Local Ollama Fallback
    OLLAMA_URL: str = Field(default="http://localhost:11434", env="OLLAMA_URL")
    OLLAMA_MODEL: str = Field(default="llama3.2:1b", env="OLLAMA_MODEL")
    
    # Infrastructure & Databases
    QDRANT_URL: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    COLLECTION_NAME: str = Field(default="verimind_docs", env="COLLECTION_NAME")
    
    # Native Embedding & ML Models
    DENSE_MODEL: str = Field(default="BAAI/bge-small-en-v1.5", env="DENSE_MODEL")
    SPARSE_MODEL: str = Field(default="Qdrant/bm25", env="SPARSE_MODEL")
    RERANKER_MODEL: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2", env="RERANKER_MODEL")
    
    # ML Weights & Latent World Model Settings
    ENABLE_JEPA_WORLD_MODEL: bool = Field(default=True, env="ENABLE_JEPA_WORLD_MODEL")
    JEPA_WEIGHT: float = Field(default=0.30, env="JEPA_WEIGHT")
    ENABLE_CROSS_ENCODER: bool = Field(default=True, env="ENABLE_CROSS_ENCODER")
    
    # Security & Governance
    JWT_SECRET: str = Field(default="verimind_super_secret_jwt_key_32bytes_minimum", env="JWT_SECRET")
    PII_MASKING_ENABLED: bool = Field(default=True, env="PII_MASKING_ENABLED")
    MAX_QUERY_LENGTH: int = Field(default=2048, env="MAX_QUERY_LENGTH")
    
    # Server Host
    APP_HOST: str = Field(default="0.0.0.0", env="APP_HOST")
    APP_PORT: int = Field(default=8000, env="APP_PORT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()