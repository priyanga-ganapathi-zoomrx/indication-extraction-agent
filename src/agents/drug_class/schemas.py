"""Schemas for drug class extraction with per-step checkpointing.

This module defines:
- Dataclasses for function inputs (Temporal-serializable)
- Pydantic models for LLM structured outputs
- Checkpoint schemas for pipeline state management
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field


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

ClassType = Literal["MoA", "Chemical", "Mode", "Therapeutic"]

ConfidenceLevel = Literal["high", "medium", "low"]


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


class RegimenLLMResponse(BaseModel):
    """LLM structured output for regimen identification.
    
    Matches exactly what the LLM returns per REGIMEN_IDENTIFICATION_PROMPT.md
    """
    components: list[str] = Field(default_factory=list, description="Component drugs")


# Per-drug status for checkpointing
DrugStatus = Literal["pending", "success", "failed"]


class Step1Output(BaseModel):
    """Checkpoint output for Step 1 with per-drug tracking.
    
    Supports incremental checkpointing - on retry, only pending/failed drugs
    are reprocessed.
    """
    # Per-drug status tracking
    drug_status: dict[str, DrugStatus] = Field(
        default_factory=dict,
        description="Status of each drug: pending, success, or failed"
    )
    
    # Mapping from drug to components (only for succeeded drugs)
    drug_to_components: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Maps drug name to its components (populated on success)"
    )
    
    # Flattened list of all unique components (derived from succeeded drugs)
    all_components: list[str] = Field(
        default_factory=list,
        description="Flattened list of all unique component drugs for Step 2"
    )
    
    def is_drug_done(self, drug: str) -> bool:
        """Check if a drug has already been successfully processed."""
        return self.drug_status.get(drug) == "success"
    
    def get_pending_drugs(self, all_drugs: list[str]) -> list[str]:
        """Get list of drugs that need processing (pending or failed)."""
        return [
            drug for drug in all_drugs
            if self.drug_status.get(drug) != "success"
        ]
    
    def mark_success(self, drug: str, components: list[str]) -> None:
        """Mark a drug as successfully processed."""
        self.drug_status[drug] = "success"
        self.drug_to_components[drug] = components
        # Update all_components
        for comp in components:
            if comp not in self.all_components:
                self.all_components.append(comp)
    
    def mark_failed(self, drug: str) -> None:
        """Mark a drug as failed."""
        self.drug_status[drug] = "failed"
    
    def is_complete(self, all_drugs: list[str]) -> bool:
        """Check if all drugs have been successfully processed."""
        return all(
            self.drug_status.get(drug) == "success"
            for drug in all_drugs
        )


# =============================================================================
# STEP 2: DRUG CLASS EXTRACTION (Tavily + Grounded)
# =============================================================================

# --- Search Cache Schema ---

class DrugSearchCache(BaseModel):
    """Cached search results for a drug (global, shared across abstracts).
    
    Stored at: search_cache/{drug_normalized}.json
    """
    drug: str = Field(..., description="Original drug name")
    fetched_at: str = Field(..., description="ISO timestamp when first fetched")
    drug_class_search: dict = Field(
        default_factory=dict,
        description="Drug class search results: {'fetched_at': ..., 'results': [...]}"
    )
    firm_searches: dict[str, dict] = Field(
        default_factory=dict,
        description="Firm search results keyed by firms_key: {'[\"Firm1\"]': {'results': [...]}}"
    )


# --- Input Schemas ---

@dataclass
class DrugClassExtractionInput:
    """Input for drug class extraction (single drug)."""
    abstract_id: str
    abstract_title: str
    drug: str
    firms: list[str] = field(default_factory=list)
    drug_class_results: list[dict] = field(default_factory=list)  # From Tavily drug class search
    firm_search_results: list[dict] = field(default_factory=list)  # From Tavily firm search


# --- LLM Response Schema ---

class ExtractionDetail(BaseModel):
    """Single extracted drug class with evidence."""
    extracted_text: str = Field(default="", description="Raw extracted text from source")
    normalized_form: str = Field(default="", description="Normalized drug class name")
    class_type: ClassType = Field(default="Therapeutic", description="Type of drug class")
    evidence: str = Field(default="", description="Supporting evidence snippet")
    source: str = Field(default="", description="Source URL or reference")
    confidence: ConfidenceLevel = Field(default="medium", description="Confidence level")
    rules_applied: list[str] = Field(default_factory=list, description="Extraction rules applied")


class DrugClassLLMResponse(BaseModel):
    """LLM structured output for drug class extraction.
    
    Used with with_structured_output() for reliable parsing.
    """
    drug_name: str = Field(default="", description="Drug being classified")
    drug_classes: list[str] = Field(default_factory=list, description="Extracted drug classes")
    extraction_details: list[ExtractionDetail] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Extraction reasoning")


# --- Result Schema ---

class DrugExtractionResult(BaseModel):
    """Extraction result for a single drug."""
    drug_name: str = Field(..., description="Drug being classified")
    drug_classes: list[str] = Field(default_factory=list, description="Extracted drug classes")
    extraction_details: list[ExtractionDetail] = Field(default_factory=list)
    source_type: Literal["tavily", "grounded"] = Field(default="tavily")
    reasoning: str = Field(default="", description="Extraction reasoning")
    success: bool = Field(default=True)


# --- Grounded Search LLM Response Schema (different format) ---

class GroundedSearchClassDetail(BaseModel):
    """Single drug class from grounded search.
    
    Grounded search returns drug_classes as array of objects (not strings).
    """
    class_name: str = Field(default="", description="Drug class name")
    class_type: ClassType = Field(default="Therapeutic")
    source_url: str = Field(default="")
    source_title: str = Field(default="")
    evidence: str = Field(default="")
    confidence: ConfidenceLevel = Field(default="medium")
    rules_applied: list[str] = Field(default_factory=list)


class GroundedSearchLLMResponse(BaseModel):
    """LLM structured output for grounded search extraction.
    
    Different format from Tavily extraction:
    - drug_classes contains nested objects (not strings)
    - Has no_class_found boolean
    
    Used with with_structured_output() for reliable parsing.
    """
    drug_name: str = Field(default="")
    drug_classes: list[GroundedSearchClassDetail] = Field(default_factory=list)
    reasoning: str = Field(default="")
    no_class_found: bool = Field(default=False)
    
    def to_extraction_result(self, drug: str) -> DrugExtractionResult:
        """Convert to standard DrugExtractionResult for pipeline.
        
        Args:
            drug: Original drug name (fallback if drug_name is empty)
            
        Returns:
            DrugExtractionResult with normalized format
        """
        if self.no_class_found or not self.drug_classes:
            return DrugExtractionResult(
                drug_name=self.drug_name or drug,
                drug_classes=["NA"],
                source_type="grounded",
                reasoning=self.reasoning,
                success=False,
            )
        
        # Extract class names and build extraction details
        class_names = [c.class_name for c in self.drug_classes if c.class_name]
        extraction_details = [
            ExtractionDetail(
                extracted_text=c.class_name,
                normalized_form=c.class_name,
                class_type=c.class_type,
                evidence=c.evidence,
                source=c.source_url,
                confidence=c.confidence,
                rules_applied=c.rules_applied,
            )
            for c in self.drug_classes
        ]
        
        return DrugExtractionResult(
            drug_name=self.drug_name or drug,
            drug_classes=class_names if class_names else ["NA"],
            extraction_details=extraction_details,
            source_type="grounded",
            reasoning=self.reasoning,
            success=bool(class_names),
        )


# --- Step 2 Output with Per-Drug Checkpointing ---

class Step2Output(BaseModel):
    """Checkpoint output for Step 2 with per-drug tracking.
    
    Supports incremental checkpointing - on retry, only pending/failed drugs
    are reprocessed.
    """
    # Per-drug status tracking
    drug_status: dict[str, DrugStatus] = Field(
        default_factory=dict,
        description="Status of each drug: pending, success, or failed"
    )
    
    # Extraction results keyed by drug name
    extractions: dict[str, DrugExtractionResult] = Field(
        default_factory=dict,
        description="Maps drug name to extraction result"
    )
    
    def is_drug_done(self, drug: str) -> bool:
        """Check if a drug has already been successfully processed."""
        return self.drug_status.get(drug) == "success"
    
    def get_pending_drugs(self, all_drugs: list[str]) -> list[str]:
        """Get list of drugs that need processing (pending or failed)."""
        return [
            drug for drug in all_drugs
            if self.drug_status.get(drug) != "success"
        ]
    
    def mark_success(self, drug: str, result: DrugExtractionResult) -> None:
        """Mark a drug as successfully processed."""
        self.drug_status[drug] = "success"
        self.extractions[drug] = result
    
    def mark_failed(self, drug: str, error: str) -> None:
        """Mark a drug as failed."""
        self.drug_status[drug] = "failed"
        self.extractions[drug] = DrugExtractionResult(
            drug_name=drug,
            drug_classes=["NA"],
            reasoning=f"Extraction failed: {error}",
            success=False,
        )
    
    def is_complete(self, all_drugs: list[str]) -> bool:
        """Check if all drugs have been successfully processed."""
        return all(
            self.drug_status.get(drug) == "success"
            for drug in all_drugs
        )
    
    def get_results_list(self) -> list[DrugExtractionResult]:
        """Convert extractions dict to list for downstream steps."""
        return list(self.extractions.values())


# =============================================================================
# STEP 3: DRUG CLASS SELECTION
# =============================================================================

@dataclass
class SelectionInput:
    """Input for drug class selection (single drug)."""
    abstract_id: str
    drug_name: str
    extraction_details: list[dict]


class DrugSelectionResult(BaseModel):
    """Selection result for a single drug."""
    drug_name: str = Field(..., description="Drug name")
    selected_drug_classes: list[str] = Field(
        default_factory=list,
        description="Selected class(es) - usually one unless multiple targets"
    )
    reasoning: str = Field(default="", description="Selection reasoning")


class Step3Output(BaseModel):
    """Checkpoint output for Step 3 with per-drug tracking.
    
    Supports incremental checkpointing - on retry, only pending/failed drugs
    are reprocessed.
    """
    # Per-drug status tracking
    drug_status: dict[str, DrugStatus] = Field(
        default_factory=dict,
        description="Status of each drug: pending, success, or failed"
    )
    
    # Selection results keyed by drug name
    selections: dict[str, DrugSelectionResult] = Field(
        default_factory=dict,
        description="Maps drug name to selection result"
    )
    
    def is_drug_done(self, drug: str) -> bool:
        """Check if a drug has already been successfully processed."""
        return self.drug_status.get(drug) == "success"
    
    def get_pending_drugs(self, all_drugs: list[str]) -> list[str]:
        """Get list of drugs that need processing (pending or failed)."""
        return [
            drug for drug in all_drugs
            if self.drug_status.get(drug) != "success"
        ]
    
    def mark_success(self, drug: str, result: DrugSelectionResult) -> None:
        """Mark a drug as successfully processed."""
        self.drug_status[drug] = "success"
        self.selections[drug] = result
    
    def mark_failed(self, drug: str, error: str) -> None:
        """Mark a drug as failed."""
        self.drug_status[drug] = "failed"
        self.selections[drug] = DrugSelectionResult(
            drug_name=drug,
            selected_drug_classes=["NA"],
            reasoning=f"Selection failed: {error}",
        )
    
    def is_complete(self, all_drugs: list[str]) -> bool:
        """Check if all drugs have been successfully processed."""
        return all(
            self.drug_status.get(drug) == "success"
            for drug in all_drugs
        )
    
    def get_results_list(self) -> list[DrugSelectionResult]:
        """Convert selections dict to list for downstream steps."""
        return list(self.selections.values())


# =============================================================================
# STEP 4: EXPLICIT DRUG CLASS EXTRACTION
# =============================================================================

@dataclass
class ExplicitExtractionInput:
    """Input for explicit extraction from title."""
    abstract_id: str
    abstract_title: str


class ExplicitExtractionDetail(BaseModel):
    """Extraction detail from explicit drug class extraction.
    
    Matches LLM response format from DRUG_CLASS_EXTRACTION_FROM_TITLE prompt.
    Has is_active_intervention field unlike regular ExtractionDetail.
    """
    extracted_text: str = Field(default="", description="Original text from title")
    class_type: str = Field(default="Therapeutic", description="MoA | Chemical | Mode | Therapeutic")
    normalized_form: str = Field(default="", description="Normalized drug class name")
    evidence: str = Field(default="", description="Exact quote from abstract title")
    is_active_intervention: bool = Field(default=True, description="Whether it's an active intervention")
    rules_applied: list[str] = Field(default_factory=list, description="Rules applied for normalization")
    
    def to_extraction_detail(self) -> ExtractionDetail:
        """Convert to standard ExtractionDetail for pipeline."""
        return ExtractionDetail(
            extracted_text=self.extracted_text,
            normalized_form=self.normalized_form,
            class_type=self.class_type,
            evidence=self.evidence,
            source="abstract_title",
            confidence="high" if self.is_active_intervention else "medium",
            rules_applied=self.rules_applied,
        )


class ExplicitLLMResponse(BaseModel):
    """LLM structured output for explicit drug class extraction from title.
    
    Matches the response format specified in DRUG_CLASS_EXTRACTION_FROM_TITLE prompt.
    """
    drug_classes: list[str] = Field(default_factory=list, description="Extracted drug classes")
    source: str = Field(default="abstract_title", description="Source of extraction")
    confidence_score: float = Field(default=0.0, description="Confidence score 0-1")
    reasoning: str = Field(default="", description="Step-by-step explanation")
    extraction_details: list[ExplicitExtractionDetail] = Field(
        default_factory=list,
        description="Detailed extraction information for each class"
    )
    
    def to_step4_output(self) -> "Step4Output":
        """Convert LLM response to Step4Output for pipeline."""
        # Handle empty/no classes
        drug_classes = self.drug_classes if self.drug_classes else ["NA"]
        
        # Convert extraction details to standard format
        details = [d.to_extraction_detail() for d in self.extraction_details]
        
        return Step4Output(
            explicit_drug_classes=drug_classes,
            extraction_details=details,
            reasoning=self.reasoning,
        )


class Step4Output(BaseModel):
    """Checkpoint output for Step 4."""
    explicit_drug_classes: list[str] = Field(default_factory=list)
    extraction_details: list[ExtractionDetail] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Extraction reasoning")


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


class RemovedClassInfo(BaseModel):
    """Information about a removed class during consolidation.
    
    Matches the nested structure in LLM response.
    Note: field is named 'class' in LLM output but we use 'class_name' to avoid Python keyword.
    """
    class_name: str = Field(default="", alias="class", description="The removed drug class")
    reason: str = Field(default="", description="Reason for removal")
    
    model_config = {"populate_by_name": True}


class RefinedExplicitClasses(BaseModel):
    """Nested structure for refined explicit classes in LLM response.
    
    Matches the 'refined_explicit_drug_classes' object in the LLM output.
    """
    drug_classes: list[str] = Field(default_factory=list, description="Remaining explicit classes")
    removed_classes: list[RemovedClassInfo] = Field(
        default_factory=list, 
        description="Classes that were removed with reasons"
    )


class ConsolidationLLMResponse(BaseModel):
    """LLM structured output for consolidation (Step 5).
    
    Matches the response format specified in DRUG_CLASS_CONSOLIDATION_PROMPT.
    Has nested structure: refined_explicit_drug_classes contains drug_classes and removed_classes.
    """
    refined_explicit_drug_classes: RefinedExplicitClasses = Field(
        default_factory=RefinedExplicitClasses,
        description="Nested structure with refined classes and removals"
    )
    reasoning: str = Field(default="", description="Explanation of consolidation decisions")
    
    def to_step5_output(self) -> "Step5Output":
        """Convert LLM response to Step5Output for pipeline."""
        # Extract drug classes
        drug_classes = self.refined_explicit_drug_classes.drug_classes
        if not drug_classes:
            drug_classes = ["NA"]
        
        # Extract removed class names (just the names, not full info)
        removed = [r.class_name for r in self.refined_explicit_drug_classes.removed_classes if r.class_name]
        
        return Step5Output(
            refined_explicit_classes=drug_classes,
            removed_classes=removed,
            reasoning=self.reasoning,
        )


class Step5Output(BaseModel):
    """Checkpoint output for Step 5."""
    refined_explicit_classes: list[str] = Field(
        default_factory=list,
        description="Explicit classes after removing duplicates/parents"
    )
    removed_classes: list[str] = Field(
        default_factory=list,
        description="Classes that were removed"
    )
    reasoning: str = Field(default="", description="Consolidation reasoning")


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


class ValidationIssue(BaseModel):
    """A single validation issue found during checks.
    
    Matches the nested structure in LLM response.
    """
    check_type: str = Field(
        ..., 
        description="Type of issue: hallucination | omission | rule_compliance"
    )
    severity: str = Field(
        ..., 
        description="Issue severity: high | medium | low"
    )
    description: str = Field(
        default="", 
        description="Clear description of the issue found"
    )
    evidence: str = Field(
        default="", 
        description="Specific evidence from sources supporting this finding"
    )
    drug_class: str = Field(
        default="", 
        description="The specific drug class involved (if applicable)"
    )
    transformed_drug_class: Optional[str] = Field(
        default=None, 
        description="Correctly transformed drug class (REQUIRED for rule_compliance only)"
    )
    rule_reference: str = Field(
        default="", 
        description="Rule that was violated or should have been applied"
    )


class CheckResult(BaseModel):
    """Result of a single validation check.
    
    Used in checks_performed nested structure.
    """
    passed: bool = Field(default=True, description="Whether the check passed")
    note: str = Field(default="", description="Brief note about the check result")


class ChecksPerformed(BaseModel):
    """Results of all three validation checks.
    
    Nested structure in LLM response.
    """
    hallucination_detection: CheckResult = Field(
        default_factory=CheckResult,
        description="Result of hallucination detection check"
    )
    omission_detection: CheckResult = Field(
        default_factory=CheckResult,
        description="Result of omission detection check"
    )
    rule_compliance: CheckResult = Field(
        default_factory=CheckResult,
        description="Result of rule compliance check"
    )


class ValidationLLMResponse(BaseModel):
    """LLM structured output for drug class validation.
    
    Matches the response format specified in DRUG_CLASS_VALIDATION_SYSTEM_PROMPT.
    """
    validation_status: str = Field(
        default="REVIEW", 
        description="Overall status: PASS | REVIEW | FAIL"
    )
    validation_confidence: float = Field(
        default=0.0, 
        description="Confidence score 0.0 to 1.0"
    )
    missed_drug_classes: list[str] = Field(
        default_factory=list, 
        description="Drug classes that should have been extracted but were missed"
    )
    issues_found: list[ValidationIssue] = Field(
        default_factory=list, 
        description="List of validation issues found"
    )
    checks_performed: ChecksPerformed = Field(
        default_factory=ChecksPerformed,
        description="Results of the three validation checks"
    )
    validation_reasoning: str = Field(
        default="", 
        description="Step-by-step validation reasoning"
    )


class ValidationOutput(BaseModel):
    """Pipeline-friendly validation output.
    
    Extends LLM response with metadata for tracking.
    """
    # LLM response fields
    validation_status: str = Field(default="REVIEW")
    validation_confidence: float = Field(default=0.0)
    missed_drug_classes: list[str] = Field(default_factory=list)
    issues_found: list[ValidationIssue] = Field(default_factory=list)
    checks_performed: ChecksPerformed = Field(default_factory=ChecksPerformed)
    validation_reasoning: str = Field(default="")
    
    # Metadata
    llm_calls: int = Field(default=0, description="Number of LLM calls made")
    validation_success: bool = Field(default=True, description="Whether validation completed without errors")
    raw_llm_response: Optional[str] = Field(default=None, description="Raw LLM response for debugging")
    
    @classmethod
    def from_llm_response(
        cls, 
        response: ValidationLLMResponse, 
        llm_calls: int = 1,
        raw_response: Optional[str] = None
    ) -> "ValidationOutput":
        """Create ValidationOutput from LLM response."""
        return cls(
            validation_status=response.validation_status,
            validation_confidence=response.validation_confidence,
            missed_drug_classes=response.missed_drug_classes,
            issues_found=response.issues_found,
            checks_performed=response.checks_performed,
            validation_reasoning=response.validation_reasoning,
            llm_calls=llm_calls,
            validation_success=True,
            raw_llm_response=raw_response,
        )
    
    @classmethod
    def error_response(cls, error: str, llm_calls: int = 0) -> "ValidationOutput":
        """Create error ValidationOutput."""
        return cls(
            validation_status="REVIEW",
            validation_confidence=0.0,
            issues_found=[
                ValidationIssue(
                    check_type="system_error",
                    severity="high",
                    description=f"Validation failed: {error}",
                )
            ],
            validation_reasoning=f"Validation could not be completed: {error}",
            llm_calls=llm_calls,
            validation_success=False,
        )


# =============================================================================
# CHECKPOINT / STATUS TRACKING
# =============================================================================

@dataclass
class StepResult:
    """Result of a single pipeline step (for checkpointing)."""
    step_name: StepName
    status: StepStatus
    output: Optional[dict] = None  # The actual step output as dict
    llm_calls: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0


@dataclass
class PipelineStatus:
    """Status tracking for the pipeline."""
    abstract_id: str
    abstract_title: str
    
    # Overall status
    pipeline_status: StepStatus = "pending"
    last_completed_step: Optional[StepName] = None
    failed_step: Optional[StepName] = None
    error: Optional[str] = None
    
    # Per-step status
    steps: dict[StepName, dict] = field(default_factory=dict)
    
    # Metrics
    total_llm_calls: int = 0
    last_updated: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "abstract_id": self.abstract_id,
            "abstract_title": self.abstract_title,
            "pipeline_status": self.pipeline_status,
            "last_completed_step": self.last_completed_step,
            "failed_step": self.failed_step,
            "error": self.error,
            "steps": self.steps,
            "total_llm_calls": self.total_llm_calls,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PipelineStatus":
        """Create from dictionary."""
        return cls(
            abstract_id=data["abstract_id"],
            abstract_title=data["abstract_title"],
            pipeline_status=data.get("pipeline_status", "pending"),
            last_completed_step=data.get("last_completed_step"),
            failed_step=data.get("failed_step"),
            error=data.get("error"),
            steps=data.get("steps", {}),
            total_llm_calls=data.get("total_llm_calls", 0),
            last_updated=data.get("last_updated"),
        )


# =============================================================================
# FINAL PIPELINE OUTPUT
# =============================================================================

@dataclass
class DrugClassMapping:
    """Mapping of a drug to its components and selected classes."""
    drug: str
    components: list[dict]  # [{component, selected_classes}, ...]


@dataclass
class DrugClassPipelineResult:
    """Final output of the drug class extraction pipeline."""
    abstract_id: str
    abstract_title: str
    
    # Per-drug results (from Steps 1-3)
    drug_class_mappings: list[dict] = field(default_factory=list)
    
    # Explicit classes (from Steps 4-5)
    explicit_drug_classes: list[str] = field(default_factory=list)
    
    # Metadata
    success: bool = True
    error: Optional[str] = None
    total_llm_calls: int = 0
    
    # References to checkpoint files
    checkpoint_files: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "abstract_id": self.abstract_id,
            "abstract_title": self.abstract_title,
            "drug_class_mappings": self.drug_class_mappings,
            "explicit_drug_classes": self.explicit_drug_classes,
            "success": self.success,
            "error": self.error,
            "total_llm_calls": self.total_llm_calls,
            "checkpoint_files": self.checkpoint_files,
        }


# =============================================================================
# ERROR CLASSES
# =============================================================================

class DrugClassExtractionError(Exception):
    """Raised when drug class extraction fails."""
    pass


class DrugClassPipelineError(Exception):
    """Raised when pipeline fails."""
    def __init__(self, message: str, step: Optional[StepName] = None):
        self.step = step
        super().__init__(message)

