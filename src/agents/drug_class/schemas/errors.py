"""Error classes for drug class extraction.

Custom exceptions for pipeline and extraction failures.
"""

from typing import Optional

from src.agents.drug_class.schemas.base import StepName


class DrugClassExtractionError(Exception):
    """Raised when drug class extraction fails."""
    pass


class DrugClassPipelineError(Exception):
    """Raised when pipeline fails."""
    def __init__(self, message: str, step: Optional[StepName] = None):
        self.step = step
        super().__init__(message)

