"""Base types and type aliases for drug class schemas.

Contains shared type definitions used across all schema files.
"""

from typing import Literal


# =============================================================================
# TYPE ALIASES
# =============================================================================

StepName = Literal[
    "step1_regimen",
    "step2_extraction",
    "step3_selection",
    "step4_explicit",
    "step5_consolidation"
]

StepStatus = Literal["pending", "running", "success", "failed", "skipped"]

DrugStatus = Literal["pending", "success", "failed"]

ConfidenceLevel = Literal["high", "medium", "low"]
