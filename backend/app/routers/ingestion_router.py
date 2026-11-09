"""
FastAPI Ingestion Router.
Exposes REST endpoints for uploading PDFs, scraping Web URLs, and ingesting raw text.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
import uuid

from backend.app.schemas import DocumentIngestResponse
from backend.app.ingestion.pdf_loader import pdf_loader
from backend.app.ingestion.web_loader import web_loader
from backend.app.ingestion.tasks import process_document_async
from backend.app.core.logger import logger

router = APIRouter(prefix="/ingest", tags=["Document Ingestion"])

@router.post("/raw", response_model=DocumentIngestResponse)
async def ingest_raw_text(
    title: str = Form(...),
    content: str = Form(...),
    chunking_strategy: str = Form("semantic"),
    tenant_id: str = Form("default_tenant")
):
    """
    Ingest raw text string into chunks ready for indexing.
    """
    try:
        res = await process_document_async(
            content=content,
            title=title,
            strategy=chunking_strategy,
            metadata={"tenant_id": tenant_id}
        )
        return DocumentIngestResponse(
            doc_id=res["doc_id"],
            chunks_created=res["chunks_count"],
            total_tokens=len(content.split()),
            status="indexed"
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf", response_model=DocumentIngestResponse)
async def ingest_pdf_file(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("semantic"),
    tenant_id: str = Form("default_tenant")
):
    """
    Ingest uploaded PDF file with OCR fallback.
    """
    try:
        pdf_bytes = await file.read()
        text, meta = pdf_loader.extract_text_from_bytes(pdf_bytes, filename=file.filename)
        meta["tenant_id"] = tenant_id

        res = await process_document_async(
            content=text,
            title=file.filename,
            strategy=chunking_strategy,
            metadata=meta
        )
        return DocumentIngestResponse(
            doc_id=res["doc_id"],
            chunks_created=res["chunks_count"],
            total_tokens=len(text.split()),
            status="indexed"
        )
    except Exception as e:
        logger.error(f"PDF Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/url", response_model=DocumentIngestResponse)
async def ingest_web_url(
    url: str = Form(...),
    chunking_strategy: str = Form("semantic"),
    tenant_id: str = Form("default_tenant")
):
    """
    Scrape and ingest web URL content.
    """
    try:
        text, meta = await web_loader.fetch_url(url)
        meta["tenant_id"] = tenant_id

        res = await process_document_async(
            content=text,
            title=meta.get("title", url),
            strategy=chunking_strategy,
            metadata=meta
        )
        return DocumentIngestResponse(
            doc_id=res["doc_id"],
            chunks_created=res["chunks_count"],
            total_tokens=len(text.split()),
            status="indexed"
        )
    except Exception as e:
        logger.error(f"URL Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))