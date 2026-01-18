"""Drug extraction function.

Simple function that extracts drugs from abstract titles.
Uses structured output - raises errors on failure for Temporal retry.
"""

from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.core import settings, create_llm, LLMConfig
from src.agents.core.langfuse_config import is_langfuse_enabled
from src.agents.drug.config import config
from src.agents.drug.prompts import get_extraction_prompt, EXTRACTION_PROMPT_NAME
from src.agents.drug.schemas import DrugInput, ExtractionResult


class DrugExtractionError(Exception):
    """Raised when drug extraction fails."""
    pass


@observe(as_type="generation", name="drug-extraction")
def extract_drugs(input_data: DrugInput) -> ExtractionResult:
    """Extract drugs from an abstract title.
    
    Args:
        input_data: DrugInput containing abstract_id and abstract_title
        
    Returns:
        ExtractionResult with extracted drugs
        
    Raises:
        DrugExtractionError: If extraction or parsing fails
    """
    # Load prompt
    system_prompt, prompt_version = get_extraction_prompt()
    
    # Update trace metadata if Langfuse is enabled
    langfuse_handler = None
    if is_langfuse_enabled():
        lf = get_client()
        lf.update_current_trace(
            tags=[
                f"abstract_id:{input_data.abstract_id}",
                f"prompt_version:{prompt_version}",
                f"model:{config.EXTRACTION_MODEL}",
                f"prompt_name:{EXTRACTION_PROMPT_NAME}",
            ],
        )
        lf.update_current_generation(
            model=config.EXTRACTION_MODEL,
            metadata={
                "abstract_id": input_data.abstract_id,
                "prompt_version": prompt_version,
            },
        )
        # Create LangChain callback handler linked to current trace
        langfuse_handler = CallbackHandler()
    
    # Create LLM with structured output
    base_llm = create_llm(LLMConfig(
        api_key=settings.llm.LLM_API_KEY,
        base_url=settings.llm.LLM_BASE_URL,
        model=config.EXTRACTION_MODEL,
        temperature=config.EXTRACTION_TEMPERATURE,
        max_tokens=config.EXTRACTION_MAX_TOKENS,
    ))
    llm = base_llm.with_structured_output(ExtractionResult)
    
    # Build messages
    if config.ENABLE_PROMPT_CACHING:
        system_message = SystemMessage(content=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"}
        }])
    else:
        system_message = SystemMessage(content=system_prompt)
    
    user_message = HumanMessage(
        content=f"Extract drugs from the following:\n\nabstract_title: {input_data.abstract_title}"
    )
    
    try:
        # Invoke LLM with structured output and Langfuse callback
        invoke_config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
        result: ExtractionResult = llm.invoke([system_message, user_message], config=invoke_config)
        
        if result is None:
            raise DrugExtractionError("LLM returned None response")
        
        return result
        
    except Exception as e:
        # Re-raise for Temporal retry
        raise DrugExtractionError(f"Drug extraction failed: {e}") from e
