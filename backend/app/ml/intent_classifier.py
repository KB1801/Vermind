"""
Native Scikit-Learn TF-IDF + Logistic Regression Intent Classifier.
Trained from scratch on local CPU to classify user query intent without external APIs.
Classes: factual | summary | comparison | unsupported
Saves model binary to data/models/intent_classifier.joblib.
"""
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from backend.app.core.logger import logger

class IntentClassifier:
    MODEL_PATH = Path("data/models/intent_classifier.joblib")

    # Synthetic Training Corpus for Local Scratch Training
    TRAINING_DATA = [
        # Factual
        ("What is the capital of France?", "factual"),
        ("Who invented the telephone?", "factual"),
        ("What are the system requirements for VeriMind?", "factual"),
        ("How many parameters does BGE-small have?", "factual"),
        ("What port does Qdrant run on?", "factual"),
        # Summary
        ("Summarize the attached research paper.", "summary"),
        ("Give me a brief executive overview of the project.", "summary"),
        ("TLDR for this document.", "summary"),
        ("Summarize main findings in two paragraphs.", "summary"),
        # Comparison
        ("Compare BM25 lexical search vs Dense Vector search.", "comparison"),
        ("What is the difference between Self-RAG and CRAG?", "comparison"),
        ("Pros and cons of PostgreSQL vs Qdrant for RAG.", "comparison"),
        ("Which model is better, Groq or Ollama?", "comparison"),
        # Unsupported / Out-of-scope
        ("How do I bake a chocolate cake?", "unsupported"),
        ("What is the weather in Tokyo today?", "unsupported"),
        ("Tell me a funny joke about space.", "unsupported")
    ]

    def __init__(self):
        self.pipeline = None
        self._load_or_train()

    def _load_or_train(self):
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if self.MODEL_PATH.exists():
            try:
                self.pipeline = joblib.load(self.MODEL_PATH)
                logger.info(f"✓ Native Intent Classifier loaded from {self.MODEL_PATH}")
                return
            except Exception as e:
                logger.warning(f"Failed to load saved intent model ({str(e)}). Retraining...")

        self.train_model()

    def train_model(self) -> Dict[str, Any]:
        """
        Trains TF-IDF + LogisticRegression pipeline and saves model artifact.
        """
        texts, labels = zip(*self.TRAINING_DATA)
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ('clf', LogisticRegression(random_state=42, max_iter=200))
        ])
        
        self.pipeline.fit(texts, labels)
        joblib.dump(self.pipeline, self.MODEL_PATH)
        logger.info(f"✓ Intent Classifier model trained and saved to {self.MODEL_PATH}")
        return {"samples_trained": len(texts), "model_path": str(self.MODEL_PATH)}

    def predict(self, query: str) -> Tuple[str, float]:
        """
        Predicts intent class and probability confidence for incoming query string.
        """
        if not self.pipeline:
            return "factual", 0.50

        probs = self.pipeline.predict_proba([query])[0]
        classes = self.pipeline.classes_
        top_idx = probs.argmax()
        
        return str(classes[top_idx]), float(probs[top_idx])

intent_classifier = IntentClassifier()