"""LLM response schemas for drug class extraction.

Contains Pydantic models that match LLM structured outputs.
Used with llm.with_structured_output() for reliable parsing.

Required fields (no defaults) ensure LLM response matches expected structure.
If LLM returns malformed JSON, Pydantic will raise ValidationError,
which triggers Temporal retry at the activity level.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.agents.drug_class.schemas.base import ConfidenceLevel

if TYPE_CHECKING:
    from src.agents.drug_class.schemas.outputs import (
        DrugExtractionResult,
        ExtractionDetail,
        Step4Output,
        Step5Output,
    )


# =============================================================================
# STEP 1: REGIMEN IDENTIFICATION
# =============================================================================

class RegimenLLMResponse(BaseModel):
    """LLM response for regimen identification.
    
    Components is required - LLM must return at least the original drug.
    """
    components: list[str] = Field(
        ...,  # Required - must be present
        description="Component drugs. If not a regimen, returns [drug]."
    )


# =============================================================================
# STEP 2: DRUG CLASS EXTRACTION (Tavily)
# =============================================================================

class ExtractionDetail(BaseModel):
    """Single extracted drug class with evidence.
    
    Matches the extraction_details format in DRUG_CLASS_EXTRACTION_FROM_SEARCH_REACT_PATTERN.md
    Note: confidence is NOT part of this schema (it's a top-level field).
    """
    extracted_text: str = Field(default="", description="Raw extracted text from source")
    normalized_form: str = Field(default="", description="Normalized drug class name")
    class_type: str = Field(default="Therapeutic", description="MoA | Chemical | Mode | Therapeutic")
    evidence: str = Field(default="", description="Exact quote from source")
    source: str = Field(default="", description="Where found: 'abstract_title' | 'abstract_text' | '<url>'")
    rules_applied: list[str] = Field(default_factory=list, description="Rules applied for extraction")


class DrugClassLLMResponse(BaseModel):
    """LLM structured output for drug class extraction.
    
    Matches exactly what the LLM returns per DRUG_CLASS_EXTRACTION_FROM_SEARCH_REACT_PATTERN.md
    Used with with_structured_output() for reliable parsing.
    
    Required fields (no defaults) ensure LLM response matches expected structure.
    If LLM returns malformed JSON, Pydantic will raise ValidationError.
    """
    # Required fields - must be present in LLM response
    reasoning: str = Field(
        ...,  # Required - no default
        description="Step-by-step extraction reasoning"
    )
    
    # Optional fields - can be empty/missing
    drug_name: str = Field(default="", description="Drug being classified")
    drug_classes: list[str] = Field(default_factory=list, description="Extracted drug classes")
    selected_sources: list[str] = Field(
        default_factory=list,
        description="Sources where classes were found: 'abstract_title' | 'abstract_text' | '<url>'"
    )
    confidence_score: float = Field(default=0.0, description="Confidence score 0.0-1.0")
    extraction_details: list[ExtractionDetail] = Field(default_factory=list)


# =============================================================================
# STEP 2: GROUNDED SEARCH (Different format)
# =============================================================================

class GroundedSearchClassDetail(BaseModel):
    """Single drug class from grounded search.
    
    Grounded search returns drug_classes as array of objects (not strings).
    """
    class_name: str = Field(default="", description="Drug class name")
    class_type: str = Field(default="Therapeutic")
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
    
    Required fields (no defaults) ensure LLM response matches expected structure.
    If LLM returns malformed JSON, Pydantic will raise ValidationError.
    """
    # Required field - must be present in LLM response
    reasoning: str = Field(
        ...,  # Required - no default
        description="Step-by-step extraction reasoning"
    )
    
    # Optional fields - can be empty/missing
    drug_name: str = Field(default="")
    drug_classes: list[GroundedSearchClassDetail] = Field(default_factory=list)
    no_class_found: bool = Field(default=False)
    
    def to_extraction_result(self, drug: str) -> "DrugExtractionResult":
        """Convert to standard DrugExtractionResult for pipeline.
        
        Args:
            drug: Original drug name (fallback if drug_name is empty)
            
        Returns:
            DrugExtractionResult with normalized format
        """
        # Import here to avoid circular imports
        from src.agents.drug_class.schemas.outputs import DrugExtractionResult
        
        if self.no_class_found or not self.drug_classes:
            return DrugExtractionResult(
                drug_name=self.drug_name or drug,
                drug_classes=["NA"],
                extraction_method="grounded",
                reasoning=self.reasoning,
                success=False,
            )
        
        # Extract class names and build extraction details
        class_names = [c.class_name for c in self.drug_classes if c.class_name]
        sources = list(set(c.source_url for c in self.drug_classes if c.source_url))
        extraction_details = [
            ExtractionDetail(
                extracted_text=c.class_name,
                normalized_form=c.class_name,
                class_type=c.class_type,
                evidence=c.evidence,
                source=c.source_url,
                rules_applied=c.rules_applied,
            )
            for c in self.drug_classes
        ]
        
        return DrugExtractionResult(
            drug_name=self.drug_name or drug,
            drug_classes=class_names if class_names else ["NA"],
            selected_sources=sources,
            extraction_details=extraction_details,
            extraction_method="grounded",
            reasoning=self.reasoning,
            success=bool(class_names),
        )


# =============================================================================
# STEP 4: EXPLICIT EXTRACTION FROM TITLE
# =============================================================================

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
            rules_applied=self.rules_applied,
        )


class ExplicitLLMResponse(BaseModel):
    """LLM structured output for explicit drug class extraction from title.
    
    Matches the response format specified in DRUG_CLASS_EXTRACTION_FROM_TITLE prompt.
    
    Required fields (no defaults) ensure LLM response matches expected structure.
    If LLM returns malformed JSON, Pydantic will raise ValidationError.
    """
    # Required field - must be present in LLM response
    reasoning: str = Field(
        ...,  # Required - no default
        description="Step-by-step explanation of extraction process"
    )
    
    # Optional fields - can be empty/missing
    drug_classes: list[str] = Field(default_factory=list, description="Extracted drug classes")
    source: str = Field(default="abstract_title", description="Source of extraction")
    confidence_score: float = Field(default=0.0, description="Confidence score 0-1")
    extraction_details: list[ExplicitExtractionDetail] = Field(
        default_factory=list,
        description="Detailed extraction information for each class"
    )
    
    def to_step4_output(self) -> "Step4Output":
        """Convert LLM response to Step4Output for pipeline."""
        # Import here to avoid circular imports
        from src.agents.drug_class.schemas.outputs import Step4Output
        
        # Handle empty/no classes
        drug_classes = self.drug_classes if self.drug_classes else ["NA"]
        
        # Convert extraction details to standard format
        details = [d.to_extraction_detail() for d in self.extraction_details]
        
        return Step4Output(
            explicit_drug_classes=drug_classes,
            extraction_details=details,
            reasoning=self.reasoning,
        )


# =============================================================================
# STEP 5: CONSOLIDATION
# =============================================================================

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
    
    Required fields (no defaults) ensure LLM response matches expected structure.
    If LLM returns malformed JSON, Pydantic will raise ValidationError.
    """
    # Required field - must be present in LLM response
    reasoning: str = Field(
        ...,  # Required - no default
        description="Explanation of consolidation decisions"
    )
    
    # Optional fields - can be empty/missing
    refined_explicit_drug_classes: RefinedExplicitClasses = Field(
        default_factory=RefinedExplicitClasses,
        description="Nested structure with refined classes and removals"
    )
    
    def to_step5_output(self) -> "Step5Output":
        """Convert LLM response to Step5Output for pipeline."""
        # Import here to avoid circular imports
        from src.agents.drug_class.schemas.outputs import Step5Output
        
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

