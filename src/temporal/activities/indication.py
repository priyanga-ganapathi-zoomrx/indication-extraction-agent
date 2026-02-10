"""Temporal activities for indication extraction and validation.

This module provides activities for the indication extraction pipeline:
- extract_indication: Extract medical indication from abstract titles using LangGraph agent
- validate_indication: Validate extraction results against rules

Both agents use LangGraph with tool calling capabilities for rules retrieval.

Activities are thin wrappers around existing agent classes.
They:
- Accept IndicationInput dataclass as input
- Instantiate and invoke the agent classes
- Parse and serialize outputs for Temporal

Best Practices Applied:
- Activities are synchronous because underlying LangGraph agents use synchronous execution
- Activities are idempotent - same input produces same output
- Validation is a separate activity for independent scaling and retry configuration
- Agent instances are created per-invocation to avoid state leakage between activities
"""

import json
import re

from temporalio import activity

from src.agents.indication.schemas import (
    IndicationInput,
    ExtractionLLMResponse,
    ValidationLLMResponse,
)
from src.temporal.idle_shutdown import track_activity


class IndicationExtractionError(Exception):
    """Raised when indication extraction/validation fails.
    
    This error triggers Temporal retry when raised from activities.
    """
    pass


def _parse_json_from_message(content: str, model_class):
    """Parse JSON from LLM message content and validate against schema.
    
    Handles both raw JSON and markdown code blocks.
    
    Args:
        content: Message content containing JSON
        model_class: Pydantic model class to parse into
        
    Returns:
        Parsed Pydantic model instance
        
    Raises:
        ValueError: If JSON extraction or parsing fails
        IndicationExtractionError: If JSON is valid but doesn't match schema (missing required fields)
    """
    from pydantic import ValidationError
    
    # Try to extract JSON from markdown code blocks (take the last one,
    # since LLM may echo the input JSON before providing the output JSON)
    json_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if json_matches:
        json_str = json_matches[-1]
    else:
        # Try to find raw JSON object
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError(f"No JSON found in content: {content[:200]}...")
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e
    
    try:
        return model_class(**data)
    except ValidationError as e:
        # Schema validation failed - LLM returned wrong structure
        # This is a retryable error (LLM might return correct format on retry)
        raise IndicationExtractionError(
            f"LLM response doesn't match {model_class.__name__} schema. "
            f"Missing or invalid fields: {e.errors()}"
        ) from e


def _extract_result_from_messages(messages: list, model_class) -> dict:
    """Extract and parse the result from agent messages.
    
    Looks for the last AI message with JSON content and validates against schema.
    
    Error handling:
    - ValueError (no JSON found, JSON parse error): Try earlier messages
    - IndicationExtractionError (schema validation failed): Fail immediately and retry
    
    Args:
        messages: List of LangChain message objects
        model_class: Pydantic model class to parse into
        
    Returns:
        Parsed result as dict
        
    Raises:
        IndicationExtractionError: If no valid response found or schema validation fails
            (triggers Temporal retry)
    """
    from langchain_core.messages import AIMessage
    
    # Find last AI message with content
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            try:
                parsed = _parse_json_from_message(msg.content, model_class)
                return parsed.model_dump()
            except ValueError:
                # JSON extraction/parsing failed - try earlier messages
                continue
            # Note: IndicationExtractionError (schema validation) bubbles up immediately
            # We don't try other messages because LLM clearly returned wrong format
    
    # Raise error for Temporal to retry instead of returning empty result
    raise IndicationExtractionError(
        f"No valid {model_class.__name__} JSON found in agent response. "
        f"Messages count: {len(messages)}"
    )


# =============================================================================
# EXTRACTION ACTIVITY
# =============================================================================

@activity.defn(name="extract_indication")
@track_activity
def extract_indication(input_data: IndicationInput) -> dict:
    """Extract medical indication from abstract titles.
    
    Uses a LangGraph agent with tool calling for rules retrieval.
    The agent:
    1. Analyzes the abstract/session title
    2. Retrieves relevant extraction rules via tool calls
    3. Applies rules to identify and normalize indication components
    4. Returns structured extraction result
    
    Args:
        input_data: IndicationInput dataclass containing:
            - abstract_id: Unique identifier for the abstract
            - abstract_title: The abstract title to extract from
            - session_title: Optional session title (fallback source)
    
    Returns:
        dict: Serialized ExtractionLLMResponse containing:
            - selected_source: Source used (abstract_title, session_title, none)
            - generated_indication: The extracted medical indication
            - reasoning: Step-by-step extraction reasoning
            - rules_retrieved: Rules consulted during extraction
            - components_identified: Components identified in title
    
    Raises:
        Exception: If extraction fails after retries (triggers Temporal retry)
    
    Example:
        >>> input_data = IndicationInput(
        ...     abstract_id="12345",
        ...     abstract_title="Phase 3 study of Drug X in EGFR+ NSCLC",
        ...     session_title="Lung Cancer Oral Presentations"
        ... )
        >>> result = extract_indication(input_data)
        >>> result["generated_indication"]
        "EGFR-positive non-small cell lung cancer"
    """
    from src.agents.indication.extraction_agent import IndicationAgent
    from src.agents.core.token_tracking import TokenUsageCallbackHandler
    
    activity.logger.info(
        f"Extracting indication from abstract {input_data.abstract_id}"
    )
    
    # Create tracker and agent instance
    tracker = TokenUsageCallbackHandler()
    agent = IndicationAgent()
    
    # Invoke agent with token tracking callback
    raw_result = agent.invoke(
        abstract_title=input_data.abstract_title,
        session_title=input_data.session_title,
        abstract_id=input_data.abstract_id,
        callbacks=[tracker],
    )
    
    # Parse result from messages
    messages = raw_result.get("messages", [])
    result = _extract_result_from_messages(messages, ExtractionLLMResponse)
    
    activity.logger.info(
        f"Extracted indication: '{result.get('generated_indication', '')}' "
        f"from source: {result.get('selected_source', 'unknown')}"
    )
    
    # Embed token metadata for workflow
    result["_token_usage"] = tracker.usage.to_dict()
    result["_llm_calls"] = tracker.llm_calls
    
    return result


# =============================================================================
# VALIDATION ACTIVITY
# =============================================================================

@activity.defn(name="validate_indication")
@track_activity
def validate_indication(
    input_data: IndicationInput,
    extraction_result: dict,
) -> dict:
    """Validate an indication extraction result against rules.
    
    Uses a LangGraph agent with tool calling for rules retrieval.
    The agent performs 6 validation checks:
    1. Source selection - Was the correct source used?
    2. Hallucination check - Was anything added that's not in the source?
    3. Omission check - Was anything relevant omitted?
    4. Rule application - Were extraction rules followed correctly?
    5. Exclusion compliance - Were exclusion rules respected?
    6. Formatting compliance - Is the output formatted correctly?
    
    Args:
        input_data: IndicationInput dataclass containing:
            - abstract_id: Unique identifier for the abstract
            - abstract_title: The abstract title
            - session_title: Optional session title
        extraction_result: Dict from extract_indication containing:
            - generated_indication: The extracted indication to validate
            - selected_source: Source that was used
            - reasoning: Extraction reasoning
            - rules_retrieved: Rules consulted
            - components_identified: Components identified
    
    Returns:
        dict: Serialized ValidationLLMResponse containing:
            - validation_status: PASS, REVIEW, or FAIL
            - issues_found: List of issues with severity and evidence
            - checks_performed: Results of all 6 validation checks
            - validation_reasoning: Step-by-step validation explanation
    
    Raises:
        Exception: If validation fails after retries (triggers Temporal retry)
    
    Example:
        >>> input_data = IndicationInput(
        ...     abstract_id="12345",
        ...     abstract_title="Phase 3 study of Drug X in EGFR+ NSCLC",
        ...     session_title="Lung Cancer Oral Presentations"
        ... )
        >>> extraction_result = {"generated_indication": "EGFR+ NSCLC", ...}
        >>> result = validate_indication(input_data, extraction_result)
        >>> result["validation_status"]
        "PASS"
    """
    from src.agents.indication.validation_agent import IndicationValidationAgent
    from src.agents.core.token_tracking import TokenUsageCallbackHandler
    
    activity.logger.info(
        f"Validating indication extraction for abstract {input_data.abstract_id}"
    )
    
    # Create tracker and agent instance
    tracker = TokenUsageCallbackHandler()
    agent = IndicationValidationAgent()
    
    # Invoke agent with token tracking callback
    raw_result = agent.invoke(
        session_title=input_data.session_title,
        abstract_title=input_data.abstract_title,
        extraction_result=extraction_result,
        abstract_id=input_data.abstract_id,
        callbacks=[tracker],
    )
    
    # Parse result from messages
    messages = raw_result.get("messages", [])
    result = _extract_result_from_messages(messages, ValidationLLMResponse)
    
    activity.logger.info(
        f"Validation result for abstract {input_data.abstract_id}: "
        f"{result.get('validation_status', 'UNKNOWN')}"
    )
    
    # Embed token metadata for workflow
    result["_token_usage"] = tracker.usage.to_dict()
    result["_llm_calls"] = tracker.llm_calls
    
    return result
