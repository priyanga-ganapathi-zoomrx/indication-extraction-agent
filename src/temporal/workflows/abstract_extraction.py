"""Abstract Extraction Workflow - Temporal orchestration for medical abstract processing.

This workflow orchestrates the complete extraction pipeline for a single abstract:
1. Drug Extraction → Drug Validation
2. Drug Class Pipeline (steps 1-5) for each primary drug
3. Indication Extraction → Indication Validation

Best Practices Applied:
- Single dataclass input/output (Temporal recommendation)
- Activity execution with appropriate timeouts and retry policies
- Workflow.execute_activity for all external calls
- Separate task queues for different activity types
- Proper error handling with workflow.logger
- Query support for workflow state inspection
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

# Import activities using Temporal's safe import pattern
# This is required for proper replay behavior
with workflow.unsafe.imports_passed_through():
    # Activity function references
    from src.temporal.activities.drug import extract_drugs, validate_drugs
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
    
    # Input schemas for activities
    from src.agents.drug.schemas import DrugInput, ValidationInput as DrugValidationInput
    from src.agents.drug_class.schemas import (
        RegimenInput,
        DrugClassExtractionInput,
        SelectionInput,
        ExplicitExtractionInput,
        ConsolidationInput,
    )
    from src.agents.indication.schemas import IndicationInput
    
    # Config
    from src.temporal.config import (
        TaskQueues,
        Timeouts,
        RetryPolicies,
    )


# =============================================================================
# WORKFLOW INPUT/OUTPUT SCHEMAS
# =============================================================================

@dataclass
class AbstractExtractionInput:
    """Input for the abstract extraction workflow.
    
    Best Practice: Single dataclass input allows adding fields
    without breaking existing workflow executions.
    """
    abstract_id: str
    abstract_title: str
    session_title: str = ""
    full_abstract: str = ""
    firms: list[str] = field(default_factory=list)
    
    # Processing options
    skip_drug_validation: bool = False
    skip_indication_validation: bool = False
    skip_drug_class: bool = False
    
    # Storage path for search cache
    storage_base_path: str = ""


@dataclass
class DrugResult:
    """Result of drug extraction and validation."""
    extraction: dict = field(default_factory=dict)
    validation: Optional[dict] = None


@dataclass
class DrugClassResult:
    """Result of drug class pipeline for a single drug."""
    drug: str = ""
    components: list[str] = field(default_factory=list)  # Step 1
    extraction: Optional[dict] = None  # Step 2
    selection: Optional[dict] = None  # Step 3


@dataclass
class IndicationResult:
    """Result of indication extraction and validation."""
    extraction: dict = field(default_factory=dict)
    validation: Optional[dict] = None


@dataclass
class AbstractExtractionOutput:
    """Output from the abstract extraction workflow.
    
    Contains all extraction results for downstream processing.
    """
    abstract_id: str
    
    # Drug results
    drug: DrugResult = field(default_factory=DrugResult)
    
    # Drug class results (per drug)
    drug_classes: list[DrugClassResult] = field(default_factory=list)
    
    # Explicit drug classes (from title)
    explicit_classes: list[str] = field(default_factory=list)
    refined_explicit_classes: list[str] = field(default_factory=list)
    
    # Indication results
    indication: IndicationResult = field(default_factory=IndicationResult)
    
    # Processing metadata
    completed: bool = False
    errors: list[str] = field(default_factory=list)


# =============================================================================
# ACTIVITY OPTIONS
# =============================================================================

# Fast LLM activities (GPT-4, drug extraction, etc.)
FAST_LLM_OPTIONS = {
    "start_to_close_timeout": Timeouts.FAST_LLM,
    "retry_policy": RetryPolicies.FAST_LLM,
}

# Slow LLM activities (Sonnet 4.5, indication validation)
SLOW_LLM_OPTIONS = {
    "start_to_close_timeout": Timeouts.SLOW_LLM,
    "retry_policy": RetryPolicies.SLOW_LLM,
}

# Search activities (Tavily)
SEARCH_OPTIONS = {
    "start_to_close_timeout": Timeouts.SEARCH,
    "retry_policy": RetryPolicies.SEARCH,
}


# =============================================================================
# WORKFLOW DEFINITION
# =============================================================================

@workflow.defn(name="AbstractExtractionWorkflow")
class AbstractExtractionWorkflow:
    """Orchestrates complete extraction pipeline for a medical conference abstract.
    
    The workflow processes a single abstract through:
    1. Drug extraction and validation
    2. Drug class extraction (5-step pipeline per drug)
    3. Indication extraction and validation
    
    Each step uses appropriate task queues and timeout/retry configurations.
    """
    
    def __init__(self) -> None:
        """Initialize workflow state."""
        self._output: Optional[AbstractExtractionOutput] = None
        self._current_step: str = "initialized"
    
    @workflow.run
    async def run(self, input: AbstractExtractionInput) -> AbstractExtractionOutput:
        """Execute the complete abstract extraction pipeline.
        
        Args:
            input: AbstractExtractionInput with abstract details and options
            
        Returns:
            AbstractExtractionOutput with all extraction results
        """
        # Initialize output
        self._output = AbstractExtractionOutput(abstract_id=input.abstract_id)
        
        workflow.logger.info(f"Starting extraction for abstract {input.abstract_id}")
        
        try:
            # Step 1: Drug Extraction
            self._current_step = "drug_extraction"
            await self._extract_drugs(input)
            
            # Step 2: Drug Class Pipeline (if not skipped and drugs found)
            if not input.skip_drug_class:
                primary_drugs = self._output.drug.extraction.get("primary_drugs", [])
                if primary_drugs:
                    self._current_step = "drug_class_pipeline"
                    await self._extract_drug_classes(input, primary_drugs)
            
            # Step 3: Indication Extraction
            self._current_step = "indication_extraction"
            await self._extract_indication(input)
            
            self._output.completed = True
            workflow.logger.info(f"Completed extraction for abstract {input.abstract_id}")
            
        except Exception as e:
            workflow.logger.error(f"Workflow error for {input.abstract_id}: {e}")
            self._output.errors.append(str(e))
            # Don't re-raise - return partial results
        
        return self._output
    
    @workflow.query
    def current_step(self) -> str:
        """Query the current processing step."""
        return self._current_step
    
    @workflow.query
    def get_output(self) -> Optional[AbstractExtractionOutput]:
        """Query the current output state."""
        return self._output
    
    # =========================================================================
    # DRUG EXTRACTION
    # =========================================================================
    
    async def _extract_drugs(self, input: AbstractExtractionInput) -> None:
        """Extract drugs from abstract and optionally validate."""
        workflow.logger.info(f"Extracting drugs for abstract {input.abstract_id}")
        
        # Create drug input
        drug_input = DrugInput(
            abstract_id=input.abstract_id,
            abstract_title=input.abstract_title,
        )
        
        # Execute drug extraction activity
        extraction_result = await workflow.execute_activity(
            extract_drugs,
            drug_input,
            task_queue=TaskQueues.DRUG,
            **FAST_LLM_OPTIONS,
        )
        
        self._output.drug.extraction = extraction_result
        
        # Validate if not skipped
        if not input.skip_drug_validation:
            validation_input = DrugValidationInput(
                abstract_id=input.abstract_id,
                abstract_title=input.abstract_title,
                extraction_result=extraction_result,
            )
            
            validation_result = await workflow.execute_activity(
                validate_drugs,
                validation_input,
                task_queue=TaskQueues.DRUG,
                **FAST_LLM_OPTIONS,
            )
            
            self._output.drug.validation = validation_result
    
    # =========================================================================
    # DRUG CLASS PIPELINE
    # =========================================================================
    
    async def _extract_drug_classes(
        self,
        input: AbstractExtractionInput,
        primary_drugs: list[str],
    ) -> None:
        """Execute the 5-step drug class pipeline for each drug."""
        workflow.logger.info(
            f"Processing drug classes for {len(primary_drugs)} drugs "
            f"in abstract {input.abstract_id}"
        )
        
        # Process each drug through steps 1-3
        all_components: list[str] = []
        drug_selections: list[dict] = []
        
        for drug in primary_drugs:
            drug_result = DrugClassResult(drug=drug)
            
            try:
                # Step 1: Regimen identification
                components = await self._step1_regimen(input, drug)
                drug_result.components = components
                all_components.extend(components)
                
                # Step 2 & 3: For each component
                for component in components:
                    extraction = await self._step2_extraction(input, component)
                    drug_result.extraction = extraction
                    
                    # Step 3: Selection (if extraction has details)
                    if extraction and extraction.get("extraction_details"):
                        selection = await self._step3_selection(
                            input.abstract_id,
                            component,
                            extraction.get("extraction_details", []),
                        )
                        drug_result.selection = selection
                        
                        # Collect for step 5
                        if selection:
                            drug_selections.append({
                                "drug_name": component,
                                "selected_classes": selection.get("selected_drug_classes", []),
                            })
                
            except Exception as e:
                workflow.logger.error(f"Drug class pipeline error for {drug}: {e}")
                self._output.errors.append(f"Drug class error for {drug}: {e}")
            
            self._output.drug_classes.append(drug_result)
        
        # Step 4: Explicit extraction from title
        explicit_result = await self._step4_explicit(input)
        self._output.explicit_classes = explicit_result.get("explicit_drug_classes", [])
        
        # Step 5: Consolidation
        if self._output.explicit_classes and self._output.explicit_classes != ["NA"]:
            consolidation_result = await self._step5_consolidation(
                input,
                self._output.explicit_classes,
                drug_selections,
            )
            self._output.refined_explicit_classes = consolidation_result.get(
                "refined_explicit_classes", []
            )
        else:
            self._output.refined_explicit_classes = self._output.explicit_classes
    
    async def _step1_regimen(
        self,
        input: AbstractExtractionInput,
        drug: str,
    ) -> list[str]:
        """Step 1: Identify if drug is a regimen and get components."""
        regimen_input = RegimenInput(
            abstract_id=input.abstract_id,
            abstract_title=input.abstract_title,
            drug=drug,
        )
        
        result = await workflow.execute_activity(
            step1_regimen,
            regimen_input,
            task_queue=TaskQueues.DRUG_CLASS,
            **FAST_LLM_OPTIONS,
        )
        
        return result  # Returns list[str]
    
    async def _step2_extraction(
        self,
        input: AbstractExtractionInput,
        drug: str,
    ) -> dict:
        """Step 2: Extract drug classes (search + LLM extraction)."""
        # First fetch search results
        search_result = await workflow.execute_activity(
            step2_fetch_search_results,
            args=[drug, input.firms, input.storage_base_path],
            task_queue=TaskQueues.DRUG_CLASS,
            **SEARCH_OPTIONS,
        )
        
        # Build extraction input
        extraction_input = DrugClassExtractionInput(
            abstract_id=input.abstract_id,
            abstract_title=input.abstract_title,
            drug=drug,
            full_abstract=input.full_abstract,
            firms=input.firms,
            drug_class_results=search_result.get("drug_class_results", []),
            firm_search_results=search_result.get("firm_search_results", []),
        )
        
        # Try Tavily extraction first
        extraction_result = await workflow.execute_activity(
            step2_extract_with_tavily,
            extraction_input,
            task_queue=TaskQueues.DRUG_CLASS,
            **FAST_LLM_OPTIONS,
        )
        
        # If Tavily returns NA, try grounded search
        drug_classes = extraction_result.get("drug_classes", [])
        if not drug_classes or drug_classes == ["NA"]:
            workflow.logger.info(f"Tavily returned NA for {drug}, trying grounded search")
            extraction_result = await workflow.execute_activity(
                step2_extract_with_grounded,
                extraction_input,
                task_queue=TaskQueues.DRUG_CLASS,
                **FAST_LLM_OPTIONS,
            )
        
        return extraction_result
    
    async def _step3_selection(
        self,
        abstract_id: str,
        drug_name: str,
        extraction_details: list,
    ) -> dict:
        """Step 3: Select best drug class for multi-class drugs."""
        selection_input = SelectionInput(
            abstract_id=abstract_id,
            drug_name=drug_name,
            extraction_details=extraction_details,
        )
        
        result = await workflow.execute_activity(
            step3_selection,
            selection_input,
            task_queue=TaskQueues.DRUG_CLASS,
            **FAST_LLM_OPTIONS,
        )
        
        return result
    
    async def _step4_explicit(self, input: AbstractExtractionInput) -> dict:
        """Step 4: Extract explicit drug classes from title."""
        explicit_input = ExplicitExtractionInput(
            abstract_id=input.abstract_id,
            abstract_title=input.abstract_title,
        )
        
        result = await workflow.execute_activity(
            step4_explicit,
            explicit_input,
            task_queue=TaskQueues.DRUG_CLASS,
            **FAST_LLM_OPTIONS,
        )
        
        return result
    
    async def _step5_consolidation(
        self,
        input: AbstractExtractionInput,
        explicit_classes: list[str],
        drug_selections: list[dict],
    ) -> dict:
        """Step 5: Consolidate explicit and drug-derived classes."""
        consolidation_input = ConsolidationInput(
            abstract_id=input.abstract_id,
            abstract_title=input.abstract_title,
            explicit_drug_classes=explicit_classes,
            drug_selections=drug_selections,
        )
        
        result = await workflow.execute_activity(
            step5_consolidation,
            consolidation_input,
            task_queue=TaskQueues.DRUG_CLASS,
            **FAST_LLM_OPTIONS,
        )
        
        return result
    
    # =========================================================================
    # INDICATION EXTRACTION
    # =========================================================================
    
    async def _extract_indication(self, input: AbstractExtractionInput) -> None:
        """Extract indication from abstract and optionally validate."""
        workflow.logger.info(f"Extracting indication for abstract {input.abstract_id}")
        
        # Create indication input
        indication_input = IndicationInput(
            abstract_id=input.abstract_id,
            abstract_title=input.abstract_title,
            session_title=input.session_title,
        )
        
        # Execute indication extraction activity
        extraction_result = await workflow.execute_activity(
            extract_indication,
            indication_input,
            task_queue=TaskQueues.INDICATION_EXTRACTION,
            **FAST_LLM_OPTIONS,
        )
        
        self._output.indication.extraction = extraction_result
        
        # Validate if not skipped (uses slower Sonnet 4.5)
        if not input.skip_indication_validation:
            validation_result = await workflow.execute_activity(
                validate_indication,
                args=[indication_input, extraction_result],
                task_queue=TaskQueues.INDICATION_VALIDATION,
                **SLOW_LLM_OPTIONS,
            )
            
            self._output.indication.validation = validation_result
