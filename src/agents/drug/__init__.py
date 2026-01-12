# src/agents/drug/__init__.py
"""Drug extraction agent package.

This package contains the drug extraction agent and related components:
- DrugExtractionAgent: Main agent for extracting drugs from abstracts
- prompts: Prompt loading utilities for drug extraction/validation/verification
"""

from src.agents.drug.extraction_agent import DrugExtractionAgent
from src.agents.drug.prompts import (
    get_system_prompt,
    get_extraction_prompt,
    get_validation_prompt,
    get_verification_prompt,
    EXTRACTION_PROMPT_NAME,
    VALIDATION_PROMPT_NAME,
    VERIFICATION_PROMPT_NAME,
)

__all__ = [
    "DrugExtractionAgent",
    "get_system_prompt",
    "get_extraction_prompt",
    "get_validation_prompt",
    "get_verification_prompt",
    "EXTRACTION_PROMPT_NAME",
    "VALIDATION_PROMPT_NAME",
    "VERIFICATION_PROMPT_NAME",
]
