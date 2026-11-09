"""
FastAPI Native Machine Learning Router.
Exposes endpoints for Intent Classification and IR Benchmark Evaluation.
"""
from fastapi import APIRouter, Form, HTTPException
from typing import Dict, Any
from backend.app.ml.intent_classifier import intent_classifier
from backend.app.ml.eval_pipeline import ir_eval_engine
from backend.app.retrieval.qdrant_store import qdrant_store

router = APIRouter(prefix="/ml", tags=["Machine Learning"])

@router.post("/intent")
async def classify_query_intent(query: str = Form(...)) -> Dict[str, Any]:
    intent, confidence = intent_classifier.predict(query)
    return {
        "query": query,
        "predicted_intent": intent,
        "confidence": round(confidence, 4)
    }

@router.post("/train-intent")
async def retrain_intent_model() -> Dict[str, Any]:
    return intent_classifier.train_model()

@router.post("/evaluate-ir")
async def run_ir_eval_benchmark() -> Dict[str, Any]:
    sample_dataset = [
        {"query": "VeriMind laptop requirement", "relevant_chunk_ids": ["c1"]},
        {"query": "PyTorch I-JEPA latent world model", "relevant_chunk_ids": ["c2"]}
    ]
    
    def retrieval_wrapper(q: str, top_k: int):
        return qdrant_store.hybrid_search(q, top_k=top_k)

    return ir_eval_engine.evaluate_benchmark_run(sample_dataset, retrieval_wrapper, k=5)