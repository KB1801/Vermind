"""
FastAPI User Feedback Loop Router.
Captures user quality signals and automates dataset construction.
"""
from fastapi import APIRouter, Form
from typing import List, Dict, Any, Optional
from backend.app.evaluation.feedback_loop import feedback_manager

router = APIRouter(prefix="/feedback", tags=["Feedback & Evaluation Loop"])

@router.post("/submit")
async def submit_user_feedback(
    query: str = Form(...),
    answer: str = Form(...),
    chunk_ids: str = Form(""),  # Comma-separated
    rating: int = Form(...),     # 1 or -1
    comment: Optional[str] = Form("")
) -> Dict[str, Any]:

    chunk_id_list = [c.strip() for c in chunk_ids.split(",") if c.strip()]
    
    return feedback_manager.log_user_feedback(
        query=query,
        answer=answer,
        retrieved_chunk_ids=chunk_id_list,
        rating=rating,
        comment=comment or ""
    )