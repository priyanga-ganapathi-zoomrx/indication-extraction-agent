"""Temporal activities for abstract extraction.

Activities are thin wrappers around existing agent functions.
They handle:
- Accepting dataclass/Pydantic inputs
- Calling existing agent functions
- Serializing outputs to dicts for Temporal

Note: Retry logic is handled by Temporal, not tenacity.
The underlying agent functions may still have tenacity decorators
for non-Temporal use cases.
"""

from src.temporal.activities.drug import (
    extract_drugs,
    validate_drugs,
)

from src.temporal.activities.drug_class import (
    step1_regimen,
    step2_fetch_search_results,
    step2_extract_with_tavily,
    step2_extract_with_grounded,
    step3_selection,
    step4_explicit,
    step5_consolidation,
)

from src.temporal.activities.indication import (
    extract_indication,
    validate_indication,
)

# Activity lists for worker registration
DRUG_ACTIVITIES = [extract_drugs, validate_drugs]

DRUG_CLASS_ACTIVITIES = [
    step1_regimen,
    step2_fetch_search_results,
    step2_extract_with_tavily,
    step2_extract_with_grounded,
    step3_selection,
    step4_explicit,
    step5_consolidation,
]

INDICATION_ACTIVITIES = [extract_indication, validate_indication]

# All activities combined (for convenience)
ALL_ACTIVITIES = DRUG_ACTIVITIES + DRUG_CLASS_ACTIVITIES + INDICATION_ACTIVITIES

__all__ = [
    # Drug activities
    "extract_drugs",
    "validate_drugs",
    # Drug class activities
    "step1_regimen",
    "step2_fetch_search_results",
    "step2_extract_with_tavily",
    "step2_extract_with_grounded",
    "step3_selection",
    "step4_explicit",
    "step5_consolidation",
    # Indication activities
    "extract_indication",
    "validate_indication",
    # Activity lists
    "DRUG_ACTIVITIES",
    "DRUG_CLASS_ACTIVITIES",
    "INDICATION_ACTIVITIES",
    "ALL_ACTIVITIES",
]
