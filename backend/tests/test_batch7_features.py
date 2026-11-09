"""
Pytest Verification Suite for Verification, ML Intent Classifier, and IR Evaluation.
"""
import pytest
from backend.app.verification.contradiction import contradiction_detector
from backend.app.ml.intent_classifier import intent_classifier
from backend.app.ml.eval_pipeline import ir_eval_engine
from backend.app.evaluation.feedback_loop import feedback_manager

def test_contradiction_detection():
    evidence = [{"chunk_id": "c1", "text": "VeriMind was built specifically to operate on 8GB RAM laptops."}]
    claim = "VeriMind works on an 8GB RAM laptop."
    verdict, confidence, chunks, reasoning = contradiction_detector.check_claim_against_evidence(claim, evidence)
    assert verdict == "SUPPORTED"
    assert confidence >= 0.50

def test_intent_classifier():
    intent, conf = intent_classifier.predict("Summarize the research project findings.")
    assert intent in ["summary", "factual", "comparison", "unsupported"]
    assert conf > 0.0

def test_ir_evaluation_metrics():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    ground_truth = ["c3"]
    
    recall = ir_eval_engine.compute_recall_at_k(retrieved, ground_truth, k=5)
    mrr = ir_eval_engine.compute_mrr_at_k(retrieved, ground_truth, k=5)
    ndcg = ir_eval_engine.compute_ndcg_at_k(retrieved, ground_truth, k=5)

    assert recall == 1.0
    assert mrr == 1.0 / 3.0  # Found at rank 3
    assert ndcg > 0.0

def test_feedback_logging():
    res = feedback_manager.log_user_feedback(
        query="Test feedback query",
        answer="Test answer",
        retrieved_chunk_ids=["c1"],
        rating=-1,
        comment="Accuracy issue"
    )
    assert res["status"] == "recorded"
    assert res["promoted_to_eval_set"] is True