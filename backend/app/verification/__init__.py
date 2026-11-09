"""
VeriMind Verification Package.
Exports Atomic Claim Decomposer, Contradiction Engine, and Evidence Integrity Validator.
"""
from backend.app.verification.claim_decomposer import ClaimDecomposer
from backend.app.verification.contradiction import ContradictionDetector
from backend.app.verification.validator import EvidenceIntegrityValidator

__all__ = ["ClaimDecomposer", "ContradictionDetector", "EvidenceIntegrityValidator"]