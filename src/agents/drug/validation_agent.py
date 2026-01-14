"""Drug validation function.

Simple function that validates extracted drugs.
Uses structured output - raises errors on failure for Temporal retry.
"""

import json

from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.core import settings, create_llm, LLMConfig
from src.agents.core.langfuse_config import is_langfuse_enabled
from src.agents.drug.config import config
from src.agents.drug.prompts import get_validation_prompt_parts, VALIDATION_PROMPT_NAME
from src.agents.drug.schemas import ValidationInput, ValidationResult


class DrugValidationError(Exception):
    """Raised when drug validation fails."""
    pass


@observe(as_type="generation", name="drug-validation")
def validate_drugs(input_data: ValidationInput) -> ValidationResult:
    """Validate extracted drugs against rules.
    
    Args:
        input_data: ValidationInput containing extraction result to validate
        
    Returns:
        ValidationResult with validation status and issues
        
    Raises:
        DrugValidationError: If validation or parsing fails
    """
    # Load validation prompt parts (instructions + rules)
    validation_instructions, extraction_rules, validation_version = get_validation_prompt_parts()
    
    # Update trace metadata if Langfuse is enabled
    langfuse_handler = None
    if is_langfuse_enabled():
        lf = get_client()
        lf.update_current_trace(
            tags=[
                f"abstract_id:{input_data.abstract_id}",
                f"prompt_version:{validation_version}",
                f"model:{config.VALIDATION_MODEL}",
                f"prompt_name:{VALIDATION_PROMPT_NAME}",
            ],
        )
        lf.update_current_generation(
            model=config.VALIDATION_MODEL,
            metadata={
                "abstract_id": input_data.abstract_id,
                "prompt_version": validation_version,
            },
        )
        # Create LangChain callback handler linked to current trace
        langfuse_handler = CallbackHandler()
    
    # Create LLM with structured output
    base_llm = create_llm(LLMConfig(
        api_key=settings.llm.LLM_API_KEY,
        base_url=settings.llm.LLM_BASE_URL,
        model=config.VALIDATION_MODEL,
        temperature=config.VALIDATION_TEMPERATURE,
        max_tokens=config.VALIDATION_MAX_TOKENS,
    ))
    llm = base_llm.with_structured_output(ValidationResult)
    
    # Build messages (3-message pattern: system + rules + input)
    if config.ENABLE_PROMPT_CACHING:
        system_message = SystemMessage(content=[{
            "type": "text",
            "text": validation_instructions,
            "cache_control": {"type": "ephemeral"}
        }])
        rules_message = HumanMessage(content=[{
            "type": "text",
            "text": f"# REFERENCE RULES DOCUMENT\n\n{extraction_rules}",
            "cache_control": {"type": "ephemeral"}
        }])
    else:
        system_message = SystemMessage(content=validation_instructions)
        rules_message = HumanMessage(
            content=f"# REFERENCE RULES DOCUMENT\n\n{extraction_rules}"
        )
    
    validation_input = f"""Validate the extracted drugs for the following:

**Abstract Title:** {input_data.abstract_title}

**Extraction Result:**
{json.dumps(input_data.extraction_result, indent=2)}"""
    
    user_message = HumanMessage(content=validation_input)
    
    try:
        # Invoke LLM with structured output and Langfuse callback
        invoke_config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
        result: ValidationResult = llm.invoke(
            [system_message, rules_message, user_message], 
            config=invoke_config
        )
        
        if result is None:
            raise DrugValidationError("LLM returned None response")
        
        return result
        
    except Exception as e:
        # Re-raise for Temporal retry
        raise DrugValidationError(f"Drug validation failed: {e}") from e
