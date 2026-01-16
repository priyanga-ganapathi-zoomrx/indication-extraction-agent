"""Input schemas for drug class extraction pipeline.

Contains dataclass inputs for each pipeline step.
These are Temporal-serializable and used as function arguments.
"""

from dataclasses import dataclass, field


# =============================================================================
# PIPELINE INPUT (from drug extraction module)
# =============================================================================

@dataclass
class DrugClassInput:
    """Input for the drug class extraction pipeline.
    
    Chains from drug extraction output.
    """
    abstract_id: str
    abstract_title: str
    full_abstract: str = ""
    primary_drugs: list[str] = field(default_factory=list)
    secondary_drugs: list[str] = field(default_factory=list)
    comparator_drugs: list[str] = field(default_factory=list)
    firms: list[str] = field(default_factory=list)  # For firm search in Step 2


# =============================================================================
# STEP 1: REGIMEN IDENTIFICATION
# =============================================================================

@dataclass
class RegimenInput:
    """Input for regimen identification (single drug)."""
    abstract_id: str
    abstract_title: str
    drug: str


# =============================================================================
# STEP 2: DRUG CLASS EXTRACTION
# =============================================================================

@dataclass
class DrugClassExtractionInput:
    """Input for drug class extraction (single drug)."""
    abstract_id: str
    abstract_title: str
    drug: str
    full_abstract: str = ""
    firms: list[str] = field(default_factory=list)
    drug_class_results: list[dict] = field(default_factory=list)  # From Tavily drug class search
    firm_search_results: list[dict] = field(default_factory=list)  # From Tavily firm search


# =============================================================================
# STEP 3: DRUG CLASS SELECTION
# =============================================================================

@dataclass
class SelectionInput:
    """Input for drug class selection (single drug)."""
    abstract_id: str
    drug_name: str
    extraction_details: list[dict]


# =============================================================================
# STEP 4: EXPLICIT EXTRACTION
# =============================================================================

@dataclass
class ExplicitExtractionInput:
    """Input for explicit extraction from title."""
    abstract_id: str
    abstract_title: str


# =============================================================================
# STEP 5: CONSOLIDATION
# =============================================================================

@dataclass
class ConsolidationInput:
    """Input for consolidation."""
    abstract_id: str
    abstract_title: str
    explicit_drug_classes: list[str]
    drug_selections: list[dict]  # [{drug_name, selected_classes}, ...]


# =============================================================================
# VALIDATION
# =============================================================================

@dataclass
class ValidationInput:
    """Input for drug class validation.
    
    Contains the original extraction inputs and the result to validate.
    """
    abstract_id: str
    drug_name: str
    abstract_title: str
    full_abstract: str
    search_results: list[dict]  # [{url, content}, ...]
    extraction_result: dict  # {drug_classes, selected_sources, reasoning, extraction_details}

