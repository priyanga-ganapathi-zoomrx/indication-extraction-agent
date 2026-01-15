"""Step 3: Specific Drug Class Selection.

For drugs associated with multiple drug classes, select the most appropriate
drug class per drug by prioritizing class types (MoA > Chemical > Mode > Therapeutic)
unless the drug has multiple biological targets.

This module exports a SINGLE-DRUG function. Loop/checkpointing is in pipeline.py.
Uses with_structured_output for reliable JSON parsing.
"""

import json

from langfuse import observe
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
def select_drug_class(input_data: SelectionInput) -> DrugSelectionResult:
    """Select the best drug class(es) for a single drug.
    
    This is an atomic function for a SINGLE drug. For processing multiple
    drugs with checkpointing, use the pipeline orchestrator.
    
    Uses with_structured_output for reliable parsing.
    
    Args:
        input_data: SelectionInput with drug and extraction details
        
    Returns:
        DrugSelectionResult with selected class(es)
        
    Raises:
        DrugClassExtractionError: If selection fails
    """
    # Handle edge case: no classes to select from
    if not input_data.extraction_details:
        return DrugSelectionResult(
            drug_name=input_data.drug_name,
            selected_drug_classes=["NA"],
            reasoning="No extracted classes provided for selection.",
        )
    
    # Handle edge case: only one unique class - no LLM call needed
    unique_classes = list(set(
        detail.get('normalized_form', detail.get('extracted_text', ''))
        for detail in input_data.extraction_details
        if detail.get('normalized_form') or detail.get('extracted_text')
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
            "extracted_text": detail.get("extracted_text", ""),
            "class_type": detail.get("class_type", "Therapeutic"),
            "drug_class": detail.get("normalized_form", detail.get("extracted_text", "")),
            "evidence": detail.get("evidence", ""),
            "source": detail.get("source", ""),
            "rules_applied": detail.get("rules_applied", []),
        })
    
    input_json = json.dumps({
        "drug_name": input_data.drug_name,
        "extracted_classes": extracted_classes,
    }, indent=2)
    
    # Substitute input into prompt
    prompt_with_input = selection_prompt.replace("{input_json}", input_json)
    
    # Create LLM with structured output
    base_llm = create_llm(LLMConfig(
        api_key=settings.llm.LLM_API_KEY,
        base_url=settings.llm.LLM_BASE_URL,
        model=config.SELECTION_MODEL,
        temperature=config.SELECTION_TEMPERATURE,
        max_tokens=config.SELECTION_MAX_TOKENS,
    ))
    llm = base_llm.with_structured_output(DrugSelectionResult)
    
    # Build messages
    if config.ENABLE_PROMPT_CACHING:
        system_message = SystemMessage(content=[{
            "type": "text",
            "text": prompt_with_input,
            "cache_control": {"type": "ephemeral"}
        }])
        rules_msg = HumanMessage(content=[{
            "type": "text",
            "text": rules_message,
            "cache_control": {"type": "ephemeral"}
        }])
    else:
        system_message = SystemMessage(content=prompt_with_input)
        rules_msg = HumanMessage(content=rules_message)
    
    instruction_msg = HumanMessage(
        content="Understand the extraction rules, analyze the evidence for each extracted class, "
                "and select the most appropriate drug class(es) based on the selection rules."
    )
    
    messages = [system_message, rules_msg, instruction_msg]
    
    # Setup Langfuse callback if enabled
    invoke_config = {}
    if is_langfuse_enabled():
        invoke_config["callbacks"] = [CallbackHandler()]
        invoke_config["metadata"] = {
            "langfuse_tags": [
                f"abstract_id:{input_data.abstract_id}",
                f"drug:{input_data.drug_name}",
                f"prompt_version:{prompt_version}",
                f"model:{config.SELECTION_MODEL}",
                f"prompt_name:{SELECTION_PROMPT_NAME}",
            ]
        }
    
    try:
        result: DrugSelectionResult = llm.invoke(messages, config=invoke_config)
        
        if result is None:
            raise DrugClassExtractionError(f"LLM returned None for {input_data.drug_name}")
        
        # Ensure drug_name is set
        if not result.drug_name:
            result.drug_name = input_data.drug_name
        
        # Ensure we have at least NA if no classes selected
        if not result.selected_drug_classes:
            result.selected_drug_classes = ["NA"]
        
        return result
        
    except DrugClassExtractionError:
        raise
    except Exception as e:
        raise DrugClassExtractionError(f"Selection failed for {input_data.drug_name}: {e}") from e


def needs_llm_selection(extraction_details: list[dict]) -> bool:
    """Check if LLM selection is needed for given extraction details.
    
    Returns False if there's 0 or 1 unique classes (no selection needed).
    
    Args:
        extraction_details: List of extraction detail dicts
        
    Returns:
        True if LLM call is needed, False otherwise
    """
    if not extraction_details:
        return False
    
    unique_classes = set(
        detail.get('normalized_form', detail.get('extracted_text', ''))
        for detail in extraction_details
        if detail.get('normalized_form') or detail.get('extracted_text')
    )
    
    return len(unique_classes) > 1
