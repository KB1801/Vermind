"""
VeriMind Native Machine Learning Package.
Exports Scikit-Learn Intent Classifier and IR Benchmark Evaluation Engine.
"""
from backend.app.ml.intent_classifier import IntentClassifier
from backend.app.ml.eval_pipeline import IREvaluationEngine

__all__ = ["IntentClassifier", "IREvaluationEngine"]