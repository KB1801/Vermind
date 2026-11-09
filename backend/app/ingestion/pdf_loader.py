"""
Robust PDF Document Loader with Automatic OCR Fallback.
Handles digital PDFs via PyPDF and scanned images via Tesseract OCR fallback.
"""
import io
from typing import Dict, Any, Tuple
from pypdf import PdfReader
from backend.app.core.logger import logger
from backend.app.core.exceptions import DocumentIngestionException

class PDFLoader:
    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr

    def extract_text_from_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> Tuple[str, Dict[str, Any]]:
        """
        Extracts raw text from PDF bytes.
        Falls back to Tesseract OCR if digital extraction yields empty text.
        """
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(reader.pages)
            extracted_pages = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                extracted_pages.append(text.strip())

            full_text = "\n\n".join(extracted_pages).strip()

            # If extracted text is insufficient, trigger OCR fallback if enabled
            is_scanned = len(full_text) < 50 and num_pages > 0
            if is_scanned and self.enable_ocr:
                logger.info(f"Digital text extraction empty for '{filename}'. Triggering Tesseract OCR fallback...")
                full_text = self._ocr_fallback(pdf_bytes)

            metadata = {
                "source": filename,
                "file_type": "pdf",
                "page_count": num_pages,
                "is_scanned_pdf": is_scanned,
                "total_char_length": len(full_text)
            }

            if not full_text:
                raise DocumentIngestionException(f"Failed to extract text from PDF: {filename}")

            return full_text, metadata

        except Exception as e:
            logger.error(f"Error parsing PDF file {filename}: {str(e)}")
            raise DocumentIngestionException(f"PDF parsing error: {str(e)}")

    def _ocr_fallback(self, pdf_bytes: bytes) -> str:
        """OCR Fallback processing using pdf2image and pytesseract."""
        try:
            from pdf2image import convert_from_bytes
            import pytesseract

            images = convert_from_bytes(pdf_bytes)
            ocr_text = []
            for img in images:
                text = pytesseract.image_to_string(img)
                ocr_text.append(text)
            return "\n\n".join(ocr_text).strip()
        except ImportError:
            logger.warning("pdf2image or pytesseract not installed. OCR fallback skipped.")
            return ""
        except Exception as e:
            logger.error(f"OCR Fallback execution failed: {str(e)}")
            return ""

pdf_loader = PDFLoader()