"""
Pytest Verification Suite for Final ML & Infrastructure Modules.
"""
import pytest
from backend.app.ml.intent_classifier import intent_classifier
from backend.app.evaluation.jepa_world_model import IJEPALatentWorldModel

def test_intent_classifier_persistence():
    res = intent_classifier.train_model()
    assert res["samples_trained"] > 0

def test_ijepa_latent_model_forward():
    import torch
    model = IJEPALatentWorldModel()
    x = torch.randn(1, 384)
    latent, pred = model(x)
    assert latent.shape == (1, 128)
    assert pred.shape == (1, 384)