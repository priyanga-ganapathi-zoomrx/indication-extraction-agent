"""Temporal activities for abstract extraction.

Activities are thin wrappers around existing agent functions.
They handle:
- Accepting dataclass/Pydantic inputs
- Calling existing agent functions
- Serializing outputs to dicts for Temporal

Activity Groups:
- CHECKPOINT: Storage operations for status and step outputs (fast I/O)
- DRUG: Drug extraction and validation (fast LLM)
- DRUG_CLASS: 5-step drug class pipeline (search + LLM)
- INDICATION: Indication extraction and validation (mixed LLM speeds)

Note: Retry logic is handled by Temporal, not tenacity.
The underlying agent functions may still have tenacity decorators
for non-Temporal use cases.
"""

# =============================================================================
# CHECKPOINT ACTIVITIES (storage operations)
# =============================================================================

from src.temporal.activities.checkpoint import (
    load_workflow_status,
    save_workflow_status,
    load_step_output,
    save_step_output,
)

CHECKPOINT_ACTIVITIES = [
    load_workflow_status,
    save_workflow_status,
    load_step_output,
    save_step_output,
]

# =============================================================================
# DRUG ACTIVITIES
# =============================================================================

from src.temporal.activities.drug import (
    extract_drugs,
    validate_drugs,
)

DRUG_ACTIVITIES = [
    extract_drugs,
    validate_drugs,
]

# =============================================================================
# DRUG CLASS ACTIVITIES (5-step pipeline)
# =============================================================================

from src.temporal.activities.drug_class import (
    step1_regimen,
    step2_fetch_search_results,
    step2_extract_with_tavily,
    step2_extract_with_grounded,
    step3_selection,
    step4_explicit,
    step5_consolidation,
    validate_drug_class_activity,
)

DRUG_CLASS_ACTIVITIES = [
    step1_regimen,
    step2_fetch_search_results,
    step2_extract_with_tavily,
    step2_extract_with_grounded,
    step3_selection,
    step4_explicit,
    step5_consolidation,
    validate_drug_class_activity,
]

# =============================================================================
# INDICATION ACTIVITIES
# =============================================================================

from src.temporal.activities.indication import (
    extract_indication,
    validate_indication,
)

INDICATION_ACTIVITIES = [
    extract_indication,
    validate_indication,
]

# =============================================================================
# COMBINED LISTS
# =============================================================================

ALL_ACTIVITIES = (
    CHECKPOINT_ACTIVITIES +
    DRUG_ACTIVITIES +
    DRUG_CLASS_ACTIVITIES +
    INDICATION_ACTIVITIES
)

__all__ = [
    # Checkpoint activities
    "load_workflow_status",
    "save_workflow_status",
    "load_step_output",
    "save_step_output",
    "CHECKPOINT_ACTIVITIES",
    # Drug activities
    "extract_drugs",
    "validate_drugs",
    "DRUG_ACTIVITIES",
    # Drug class activities
    "step1_regimen",
    "step2_fetch_search_results",
    "step2_extract_with_tavily",
    "step2_extract_with_grounded",
    "step3_selection",
    "step4_explicit",
    "step5_consolidation",
    "validate_drug_class_activity",
    "DRUG_CLASS_ACTIVITIES",
    # Indication activities
    "extract_indication",
    "validate_indication",
    "INDICATION_ACTIVITIES",
    # Combined
    "ALL_ACTIVITIES",
]
