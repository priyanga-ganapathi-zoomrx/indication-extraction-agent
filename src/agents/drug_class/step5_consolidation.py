"""Step 5: Refinement / Consolidation.

Compare explicit drug classes (from Step 4) with specific drug classes (from Step 3),
and remove duplicates as well as parent drug classes within the same hierarchy.

This module exports a single function. Orchestration is in pipeline.py.
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
    get_consolidation_prompt_parts,
    CONSOLIDATION_PROMPT_NAME,
)
from src.agents.drug_class.schemas import (
    ConsolidationInput,
    ConsolidationLLMResponse,
    Step5Output,
    DrugClassExtractionError,
)


@observe(as_type="generation", name="drug-class-step5-consolidation")
def consolidate_drug_classes(input_data: ConsolidationInput) -> Step5Output:
    """Consolidate explicit classes with drug-derived classes.
    
    Compares explicit drug classes (from Step 4) with drug-specific selections
    (from Step 3) and removes duplicates/parent classes.
    
    Uses with_structured_output for reliable parsing.
    
    Args:
        input_data: ConsolidationInput with explicit classes and drug selections
        
    Returns:
        Step5Output with refined explicit classes
        
    Raises:
        DrugClassExtractionError: If consolidation fails
    """
    # Handle empty explicit classes - no LLM call needed
    if not input_data.explicit_drug_classes or input_data.explicit_drug_classes == ["NA"]:
        return Step5Output(
            refined_explicit_classes=["NA"],
            removed_classes=[],
            reasoning="No explicit drug classes to consolidate.",
        )
    
    # Handle empty drug selections - nothing to compare against
    if not input_data.drug_selections:
        return Step5Output(
            refined_explicit_classes=input_data.explicit_drug_classes,
            removed_classes=[],
            reasoning="No drug selections to compare against.",
        )
    
    # Load prompts
    system_prompt, input_template, rules_message, prompt_version = get_consolidation_prompt_parts()
    
    # Format explicit drug classes as JSON
    explicit_data = {
        "drug_classes": input_data.explicit_drug_classes,
    }
    explicit_json = json.dumps(explicit_data, indent=2)
    
    # Format drug selections as JSON
    selections_json = json.dumps(input_data.drug_selections, indent=2)
    
    # Build input message
    if input_template:
        input_content = input_template.replace("{abstract_title}", input_data.abstract_title)
        input_content = input_content.replace("{explicit_drug_classes_json}", explicit_json)
        input_content = input_content.replace("{drug_selections_json}", selections_json)
    else:
        input_content = f"""# CONSOLIDATION INPUT

## Abstract Title
{input_data.abstract_title}

## Explicit Drug Classes (from Step 4)
{explicit_json}

## Drug-Specific Selections (from Step 3)
{selections_json}"""
    
    # Create LLM with structured output
    base_llm = create_llm(LLMConfig(
        api_key=settings.llm.LLM_API_KEY,
        base_url=settings.llm.LLM_BASE_URL,
        model=config.CONSOLIDATION_MODEL,
        temperature=config.CONSOLIDATION_TEMPERATURE,
        max_tokens=config.CONSOLIDATION_MAX_TOKENS,
    ))
    llm = base_llm.with_structured_output(ConsolidationLLMResponse)
    
    # Build messages
    if config.ENABLE_PROMPT_CACHING:
        system_message = SystemMessage(content=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"}
        }])
        rules_msg = HumanMessage(content=[{
            "type": "text",
            "text": rules_message,
            "cache_control": {"type": "ephemeral"}
        }])
    else:
        system_message = SystemMessage(content=system_prompt)
        rules_msg = HumanMessage(content=rules_message)
    
    input_msg = HumanMessage(content=input_content)
    messages = [system_message, rules_msg, input_msg]
    
    # Setup Langfuse callback if enabled
    invoke_config = {}
    if is_langfuse_enabled():
        invoke_config["callbacks"] = [CallbackHandler()]
        invoke_config["metadata"] = {
            "langfuse_tags": [
                f"abstract_id:{input_data.abstract_id}",
                f"prompt_version:{prompt_version}",
                f"model:{config.CONSOLIDATION_MODEL}",
                f"prompt_name:{CONSOLIDATION_PROMPT_NAME}",
                f"explicit_classes_count:{len(input_data.explicit_drug_classes)}",
                f"drug_selections_count:{len(input_data.drug_selections)}",
            ]
        }
    
    try:
        result: ConsolidationLLMResponse = llm.invoke(messages, config=invoke_config)
        
        if result is None:
            raise DrugClassExtractionError("LLM returned None for consolidation")
        
        # Convert to Step5Output
        output = result.to_step5_output()
        return output
        
    except DrugClassExtractionError:
        raise
    except Exception as e:
        raise DrugClassExtractionError(f"Consolidation failed: {e}") from e
