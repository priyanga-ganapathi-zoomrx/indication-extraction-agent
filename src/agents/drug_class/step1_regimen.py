"""Step 1: Regimen Identification.

Identifies if a drug is a regimen and extracts its component drugs.
For example, "FOLFOX" -> ["5-FU", "Leucovorin", "Oxaliplatin"]

This module exports a SINGLE-DRUG function. The loop and checkpointing
logic lives in pipeline.py for per-drug retry capability.

Uses structured output for reliable parsing.
"""

from langfuse import observe
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.core import settings, create_llm, LLMConfig
from src.agents.core.langfuse_config import is_langfuse_enabled
from src.agents.drug_class.config import config
from src.agents.drug_class.prompts import get_regimen_identification_prompt, REGIMEN_IDENTIFICATION_PROMPT_NAME
from src.agents.drug_class.schemas import (
    RegimenInput,
    RegimenLLMResponse,
    DrugClassExtractionError,
)


@observe(as_type="generation", name="drug-class-step1-regimen")
def identify_regimen(input_data: RegimenInput) -> list[str]:
    """Identify if a drug is a regimen and extract its components.
    
    This is an atomic function for a SINGLE drug. For processing multiple
    drugs with checkpointing, use the pipeline orchestrator.
    
    Uses with_structured_output for reliable JSON parsing.
    
    Args:
        input_data: RegimenInput with abstract_id, abstract_title, drug
        
    Returns:
        List of component drugs. If not a regimen, returns [drug].
        
    Raises:
        DrugClassExtractionError: If LLM call fails
    """
    # Load prompt
    system_prompt, prompt_version = get_regimen_identification_prompt()
    
    # Create LLM with structured output
    base_llm = create_llm(LLMConfig(
        api_key=settings.llm.LLM_API_KEY,
        base_url=settings.llm.LLM_BASE_URL,
        model=config.REGIMEN_MODEL,
        temperature=config.REGIMEN_TEMPERATURE,
        max_tokens=config.REGIMEN_MAX_TOKENS,
    ))
    llm = base_llm.with_structured_output(RegimenLLMResponse)
    
    # Build messages
    if config.ENABLE_PROMPT_CACHING:
        system_message = SystemMessage(content=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"}
        }])
    else:
        system_message = SystemMessage(content=system_prompt)
    
    user_content = f"""Abstract Title: {input_data.abstract_title or "Not provided"}
Drug: {input_data.drug}"""
    
    user_message = HumanMessage(content=user_content)
    
    # Setup Langfuse callback if enabled
    invoke_config = {}
    if is_langfuse_enabled():
        invoke_config["callbacks"] = [CallbackHandler()]
        invoke_config["metadata"] = {
            "langfuse_tags": [
                f"abstract_id:{input_data.abstract_id}",
                f"drug:{input_data.drug}",
                f"prompt_version:{prompt_version}",
                f"model:{config.REGIMEN_MODEL}",
                f"prompt_name:{REGIMEN_IDENTIFICATION_PROMPT_NAME}",
            ]
        }
    
    try:
        result: RegimenLLMResponse = llm.invoke([system_message, user_message], config=invoke_config)
        
        if result is None:
            raise DrugClassExtractionError(f"LLM returned None response for {input_data.drug}")
        
        # If no components, return the original drug
        components = result.components if result.components else [input_data.drug]
        return components
        
    except DrugClassExtractionError:
        raise
    except Exception as e:
        raise DrugClassExtractionError(f"Regimen identification failed for {input_data.drug}: {e}") from e
