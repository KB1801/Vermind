"""
VeriMind Ingestion Module.
Provides document loading, OCR fallback, web scraping, and chunking interfaces.
"""
from backend.app.ingestion.pdf_loader import PDFLoader
from backend.app.ingestion.docling_loader import DoclingLoader
from backend.app.ingestion.web_loader import WebLoader
from backend.app.ingestion.chunker import DocumentChunker

__all__ = ["PDFLoader", "DoclingLoader", "WebLoader", "DocumentChunker"]