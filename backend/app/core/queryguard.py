"""
QueryGuard Security & Sanitization Engine.
Protects against Prompt Injection, System Prompt Leakage, and redacts PII.
"""
import re
from typing import Tuple, List
from backend.app.core.exceptions import SecurityViolationException
from backend.app.core.logger import logger

class QueryGuard:
    # Common injection attack vectors
    INJECTION_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"ignore\s+all\s+prior\s+prompts",
        r"system\s+prompt\s+override",
        r"you\s+are\s+now\s+dan",
        r"do\s+anything\s+now",
        r"reveal\s+your\s+secret\s+instructions",
        r"bypass\s+safety\s+filter"
    ]
    
    # Regex rules for PII masking
    EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    PHONE_REGEX = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
    SSN_REGEX = r"\b\d{3}-\d{2}-\d{4}\b"

    def __init__(self, pii_enabled: bool = True):
        self.pii_enabled = pii_enabled
        self.compiled_injections = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def sanitize(self, input_text: str) -> Tuple[str, List[str]]:
        """
        Sanitizes raw user input.
        Returns: (sanitized_text, list_of_detected_warnings)
        """
        warnings = []
        
        # 1. Prompt Injection Defense
        for pattern in self.compiled_injections:
            if pattern.search(input_text):
                logger.warning(f"QueryGuard Security Alert: Blocked prompt injection pattern in query: {input_text[:50]}...")
                raise SecurityViolationException(
                    "Security Violation: Potential prompt injection attack detected.",
                    details={"matched_pattern": pattern.pattern}
                )
        
        sanitized_text = input_text
        
        # 2. PII Masking
        if self.pii_enabled:
            if re.search(self.EMAIL_REGEX, sanitized_text):
                sanitized_text = re.sub(self.EMAIL_REGEX, "[REDACTED_EMAIL]", sanitized_text)
                warnings.append("Email address redacted for privacy.")
                
            if re.search(self.PHONE_REGEX, sanitized_text):
                sanitized_text = re.sub(self.PHONE_REGEX, "[REDACTED_PHONE]", sanitized_text)
                warnings.append("Phone number redacted for privacy.")

            if re.search(self.SSN_REGEX, sanitized_text):
                sanitized_text = re.sub(self.SSN_REGEX, "[REDACTED_SSN]", sanitized_text)
                warnings.append("SSN redacted for privacy.")

        return sanitized_text, warnings

query_guard = QueryGuard()