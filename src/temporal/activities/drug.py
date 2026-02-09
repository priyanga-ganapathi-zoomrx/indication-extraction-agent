"""Temporal activities for drug extraction and validation.

These activities are thin wrappers around existing agent functions.
They:
- Accept the same input types (dataclasses) as the underlying functions
- Call the existing agent functions
- Serialize Pydantic outputs to dicts for Temporal serialization
- Let Temporal handle retries (configured in workflow execution)

Best Practices Applied:
- Activities are synchronous because underlying LLM calls use synchronous LangChain
- Activities are idempotent - same input produces same output
- Non-retryable errors (ValueError, ValidationError) configured in retry policy
- Timeouts and retries configured at workflow level, not in activities
"""

from temporalio import activity

from src.agents.drug.schemas import DrugInput, ValidationInput
from src.temporal.idle_shutdown import track_activity


@activity.defn(name="extract_drugs")
@track_activity
def extract_drugs(input_data: DrugInput) -> dict:
    """Extract drugs from an abstract title.
    
    This activity wraps the existing drug extraction agent function.
    
    Args:
        input_data: DrugInput dataclass containing:
            - abstract_id: Unique identifier for the abstract
            - abstract_title: The title text to extract drugs from
    
    Returns:
        dict: Serialized ExtractionResult containing:
            - primary_drugs: List of primary/investigational drugs
            - secondary_drugs: List of secondary/combination drugs
            - comparator_drugs: List of comparator/control drugs
            - reasoning: Step-by-step extraction reasoning
    
    Raises:
        DrugExtractionError: If LLM call fails (will trigger Temporal retry)
    
    Example:
        >>> input_data = DrugInput(
        ...     abstract_id="12345",
        ...     abstract_title="Phase 3 study of pembrolizumab vs placebo in NSCLC"
        ... )
        >>> result = extract_drugs(input_data)
        >>> result["primary_drugs"]
        ["pembrolizumab"]
    """
    # Import here to avoid circular imports and keep activity lightweight
    from src.agents.drug.extraction_agent import extract_drugs as _extract_drugs
    from src.agents.core.token_tracking import TokenUsageCallbackHandler
    
    # Log activity info for debugging
    activity.logger.info(
        f"Extracting drugs for abstract {input_data.abstract_id}"
    )
    
    # Create tracker and call agent with it
    tracker = TokenUsageCallbackHandler()
    result = _extract_drugs(input_data, callbacks=[tracker])
    
    # Serialize Pydantic model to dict with token metadata for workflow
    return {
        **result.model_dump(),
        "_token_usage": tracker.usage.to_dict(),
        "_llm_calls": tracker.llm_calls,
    }


@activity.defn(name="validate_drugs")
@track_activity
def validate_drugs(input_data: ValidationInput) -> dict:
    """Validate extracted drugs against rules.
    
    This activity wraps the existing drug validation agent function.
    
    Args:
        input_data: ValidationInput dataclass containing:
            - abstract_id: Unique identifier for the abstract
            - abstract_title: The original title text
            - extraction_result: Dict of extracted drugs to validate
    
    Returns:
        dict: Serialized ValidationResult containing:
            - validation_status: "PASS", "REVIEW", or "FAIL"
            - validation_confidence: Confidence score 0.0-1.0
            - missed_drugs: List of drugs that were missed
            - issues_found: List of validation issues
            - checks_performed: Results of all validation checks
            - validation_reasoning: Step-by-step reasoning
    
    Raises:
        DrugValidationError: If LLM call fails (will trigger Temporal retry)
    
    Example:
        >>> input_data = ValidationInput(
        ...     abstract_id="12345",
        ...     abstract_title="Phase 3 study of pembrolizumab vs placebo in NSCLC",
        ...     extraction_result={"primary_drugs": ["pembrolizumab"], ...}
        ... )
        >>> result = validate_drugs(input_data)
        >>> result["validation_status"]
        "PASS"
    """
    # Import here to avoid circular imports and keep activity lightweight
    from src.agents.drug.validation_agent import validate_drugs as _validate_drugs
    from src.agents.core.token_tracking import TokenUsageCallbackHandler
    
    # Log activity info for debugging
    activity.logger.info(
        f"Validating drugs for abstract {input_data.abstract_id}"
    )
    
    # Create tracker and call agent with it
    tracker = TokenUsageCallbackHandler()
    result = _validate_drugs(input_data, callbacks=[tracker])
    
    # Serialize Pydantic model to dict with token metadata for workflow
    return {
        **result.model_dump(),
        "_token_usage": tracker.usage.to_dict(),
        "_llm_calls": tracker.llm_calls,
    }
