"""
Advanced Structured Document Loader powered by Docling.
Extracts rich markdown, tables, and section hierarchies from modern documents.
"""
from typing import Dict, Any, Tuple
from backend.app.core.logger import logger
from backend.app.core.exceptions import DocumentIngestionException

class DoclingLoader:
    def __init__(self):
        self._docling_available = False
        try:
            import docling
            self._docling_available = True
        except ImportError:
            logger.warning("Docling package not installed. Structured document loader running in fallback mode.")

    def parse_document(self, file_path_or_bytes: Any, filename: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parses multi-modal documents into structured Markdown text.
        """
        if not self._docling_available:
            # Fallback text representation
            return f"# Document: {filename}\nDocling engine unavailable. Raw fallback active.", {
                "source": filename,
                "parser": "fallback"
            }

        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(file_path_or_bytes)
            markdown_content = result.document.export_to_markdown()

            metadata = {
                "source": filename,
                "parser": "docling_markdown",
                "total_char_length": len(markdown_content)
            }
            return markdown_content, metadata

        except Exception as e:
            logger.error(f"Docling loader failed for {filename}: {str(e)}")
            raise DocumentIngestionException(f"Docling parsing failed: {str(e)}")

docling_loader = DoclingLoader()