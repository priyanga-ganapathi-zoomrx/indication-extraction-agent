"""Drug extraction module.

Simple functions for drug extraction and validation.
No LangGraph - designed for future Temporal integration.
"""

# Functions
from src.agents.drug.extraction_agent import extract_drugs, DrugExtractionError
from src.agents.drug.validation_agent import validate_drugs, DrugValidationError

# Schemas
from src.agents.drug.schemas import (
    # Input
    DrugInput,
    ValidationInput,
    # Extraction output
    ExtractionResult,
    # Validation output
    ValidationResult,
    SearchResult,
    IssueFound,
    CheckResult,
    ChecksPerformed,
)

# Config
from src.agents.drug.config import config

# Prompts
from src.agents.drug.prompts import (
    get_extraction_prompt,
    get_validation_prompt_parts,
    EXTRACTION_PROMPT_NAME,
    VALIDATION_PROMPT_NAME,
)

__all__ = [
    # Functions
    "extract_drugs",
    "validate_drugs",
    # Errors
    "DrugExtractionError",
    "DrugValidationError",
    # Input schemas
    "DrugInput",
    "ValidationInput",
    # Output schemas
    "ExtractionResult",
    "ValidationResult",
    "SearchResult",
    "IssueFound",
    "CheckResult",
    "ChecksPerformed",
    # Config
    "config",
    # Prompts
    "get_extraction_prompt",
    "get_validation_prompt_parts",
    "EXTRACTION_PROMPT_NAME",
    "VALIDATION_PROMPT_NAME",
]
