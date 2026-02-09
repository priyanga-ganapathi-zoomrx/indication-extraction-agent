"""Step 3: Specific Drug Class Selection.

For drugs associated with multiple drug classes, select the most appropriate
drug class per drug by prioritizing class types (MoA > Chemical > Mode > Therapeutic)
unless the drug has multiple biological targets.

This module exports a SINGLE-DRUG function. Loop/checkpointing is in pipeline.py.
Uses with_structured_output for reliable JSON parsing.
Per-request timeout is 120s. Retries are handled by Temporal at the activity level.
"""

import json

from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.core import settings, create_llm, LLMConfig
from src.agents.core.langfuse_config import is_langfuse_enabled
from src.agents.drug_class.config import config
from src.agents.drug_class.prompts import (
    get_selection_prompt_parts,
    SELECTION_PROMPT_NAME,
)
from src.agents.drug_class.schemas import (
    SelectionInput,
    DrugSelectionResult,
    DrugClassExtractionError,
)


@observe(as_type="generation", name="drug-class-step3-selection")
def select_drug_class(input_data: SelectionInput, callbacks: list = None) -> DrugSelectionResult:
    """Select the best drug class(es) for a single drug.
    
    This is an atomic function for a SINGLE drug. For processing multiple
    drugs with checkpointing, use the pipeline orchestrator.
    
    Uses LangChain's with_structured_output for reliable JSON parsing.
    Per-request timeout is 120s. Retries are handled by Temporal at the activity level.
    
    Args:
        input_data: SelectionInput with drug and extraction details
        callbacks: Optional list of LangChain callback handlers (e.g., TokenUsageCallbackHandler)
        
    Returns:
        DrugSelectionResult with selected class(es)
        
    Raises:
        DrugClassExtractionError: If selection fails (triggers Temporal retry)
    """
    # Handle edge case: no classes to select from
    if not input_data.extraction_details:
        return DrugSelectionResult(
            drug_name=input_data.drug_name,
            selected_drug_classes=["NA"],
            reasoning="No extracted classes provided for selection.",
        )
    
    # Handle edge case: only one unique class - no LLM call needed
    # Support both Pydantic objects (direct calls) and dicts (Temporal pipeline)
    def _get(detail, field):
        if isinstance(detail, dict):
            return detail.get(field, "")
        return getattr(detail, field, "")

    unique_classes = list(set(
        _get(detail, "normalized_form") or _get(detail, "extracted_text")
        for detail in input_data.extraction_details
        if _get(detail, "normalized_form") or _get(detail, "extracted_text")
    ))
    
    if len(unique_classes) <= 1:
        selected = unique_classes if unique_classes else ["NA"]
        return DrugSelectionResult(
            drug_name=input_data.drug_name,
            selected_drug_classes=selected,
            reasoning="Only one unique class was extracted. No selection needed.",
        )
    
    # Load prompts
    selection_prompt, rules_message, prompt_version = get_selection_prompt_parts()
    
    # Format extraction details as JSON for input
    extracted_classes = []
    for detail in input_data.extraction_details:
        extracted_classes.append({
            "extracted_text": _get(detail, "extracted_text") or "",
            "class_type": _get(detail, "class_type") or "Therapeutic",
            "drug_class": _get(detail, "normalized_form") or _get(detail, "extracted_text") or "",
            "evidence": _get(detail, "evidence") or "",
            "source": _get(detail, "source") or "",
            "rules_applied": _get(detail, "rules_applied") or [],
        })
    
    input_json = json.dumps({
        "drug_name": input_data.drug_name,
        "extracted_classes": extracted_classes,
    }, indent=2, ensure_ascii=False)
    
    # Create LLM with structured output (120s timeout for long-running requests)
    base_llm = create_llm(LLMConfig(
        api_key=settings.llm.LLM_API_KEY,
        base_url=settings.llm.LLM_BASE_URL,
        model=config.SELECTION_MODEL,
        temperature=config.SELECTION_TEMPERATURE,
        max_tokens=config.SELECTION_MAX_TOKENS,
        timeout=120,  # 2 minute timeout
    ))
    llm = base_llm.with_structured_output(DrugSelectionResult)
    
    # Build messages - system prompt is static (enables prompt caching)
    if config.ENABLE_PROMPT_CACHING:
        system_message = SystemMessage(content=[{
            "type": "text",
            "text": selection_prompt,
            "cache_control": {"type": "ephemeral"}
        }])
        rules_msg = HumanMessage(content=[{
            "type": "text",
            "text": rules_message,
            "cache_control": {"type": "ephemeral"}
        }])
    else:
        system_message = SystemMessage(content=selection_prompt)
        rules_msg = HumanMessage(content=rules_message)
    
    # Input as a separate user message (not in system prompt for caching)
    input_msg = HumanMessage(
        content=f"""## INPUT

```json
{input_json}
```

Understand the extraction rules, analyze the evidence for each extracted class, and select the most appropriate drug class(es) based on the selection rules."""
    )
    
    messages = [system_message, rules_msg, input_msg]
    
    # Setup Langfuse callback and metadata if enabled
    langfuse_handler = None
    if is_langfuse_enabled():
        lf = get_client()
        lf.update_current_trace(
            tags=[
                f"abstract_id:{input_data.abstract_id}",
                f"drug:{input_data.drug_name}",
                f"prompt_version:{prompt_version}",
                f"model:{config.SELECTION_MODEL}",
                f"prompt_name:{SELECTION_PROMPT_NAME}",
            ],
        )
        lf.update_current_generation(
            model=config.SELECTION_MODEL,
            metadata={
                "abstract_id": input_data.abstract_id,
                "drug": input_data.drug_name,
                "prompt_version": prompt_version,
            },
        )
        langfuse_handler = CallbackHandler()
    
    all_callbacks = []
    if langfuse_handler:
        all_callbacks.append(langfuse_handler)
    if callbacks:
        all_callbacks.extend(callbacks)
    invoke_config = {"callbacks": all_callbacks} if all_callbacks else {}
    
    try:
        result: DrugSelectionResult = llm.invoke(messages, config=invoke_config)
        
        # Ensure drug_name is set
        if not result.drug_name:
            result.drug_name = input_data.drug_name
        
        # Ensure we have at least NA if no classes selected
        if not result.selected_drug_classes:
            result.selected_drug_classes = ["NA"]
        
        return result
        
    except Exception as e:
        raise DrugClassExtractionError(f"Selection failed for {input_data.drug_name}: {e}") from e


def needs_llm_selection(extraction_details: list) -> bool:
    """Check if LLM selection is needed for given extraction details.
    
    Returns False if there's 0 or 1 unique classes (no selection needed).
    
    Args:
        extraction_details: List of ExtractionDetail objects
        
    Returns:
        True if LLM call is needed, False otherwise
    """
    if not extraction_details:
        return False
    
    def _get(detail, field):
        if isinstance(detail, dict):
            return detail.get(field, "")
        return getattr(detail, field, "")

    unique_classes = set(
        _get(detail, "normalized_form") or _get(detail, "extracted_text")
        for detail in extraction_details
        if _get(detail, "normalized_form") or _get(detail, "extracted_text")
    )
    
    return len(unique_classes) > 1
