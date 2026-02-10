"""Step 4: Explicit Drug Class Extraction.

Extract drug classes directly mentioned in the abstract title.
These are explicit class mentions, not inferred from drug names.

This module exports a single function. Orchestration is in pipeline.py.
Uses with_structured_output for reliable JSON parsing.
Per-request timeout is 120s. Retries are handled by Temporal at the activity level.
"""

from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.core import settings, create_llm, LLMConfig
from src.agents.core.langfuse_config import is_langfuse_enabled
from src.agents.drug_class.config import config
from src.agents.drug_class.prompts import (
    get_explicit_extraction_prompt_parts,
    EXTRACTION_TITLE_PROMPT_NAME,
)
from src.agents.drug_class.schemas import (
    ExplicitExtractionInput,
    ExplicitLLMResponse,
    Step4Output,
    DrugClassExtractionError,
)


@observe(as_type="generation", name="drug-class-step4-explicit")
def extract_explicit_classes(input_data: ExplicitExtractionInput, callbacks: list = None) -> Step4Output:
    """Extract explicit drug classes from abstract title.
    
    This is an atomic function for extracting drug classes directly mentioned
    in the abstract title (not inferred from drug names).
    
    Uses LangChain's with_structured_output for reliable JSON parsing.
    Per-request timeout is 120s. Retries are handled by Temporal at the activity level.
    
    Args:
        input_data: ExplicitExtractionInput with abstract_id and abstract_title
        callbacks: Optional list of LangChain callback handlers (e.g., TokenUsageCallbackHandler)
        
    Returns:
        Step4Output with extracted explicit drug classes
        
    Raises:
        DrugClassExtractionError: If extraction fails (triggers Temporal retry)
    """
    # Handle empty title
    if not input_data.abstract_title or not input_data.abstract_title.strip():
        return Step4Output(
            explicit_drug_classes=["NA"],
            reasoning="Empty abstract title provided",
        )
    
    # Load prompts
    system_prompt, input_template, rules_message, prompt_version = get_explicit_extraction_prompt_parts()
    
    # Format input message
    if input_template:
        input_content = input_template.replace("{abstract_title}", input_data.abstract_title)
    else:
        input_content = f"""# EXTRACTION INPUT

## Abstract Title
{input_data.abstract_title}"""
    
    # Create LLM with structured output (120s timeout for long-running requests)
    base_llm = create_llm(LLMConfig(
        api_key=settings.llm.LLM_API_KEY,
        base_url=settings.llm.LLM_BASE_URL,
        model=config.EXPLICIT_MODEL,
        temperature=config.EXPLICIT_TEMPERATURE,
        max_tokens=config.EXPLICIT_MAX_TOKENS,
        timeout=120,  # 2 minute timeout
    ))
    llm = base_llm.with_structured_output(ExplicitLLMResponse)
    
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
    
    # Setup Langfuse callback and metadata if enabled
    langfuse_handler = None
    if is_langfuse_enabled():
        lf = get_client()
        lf.update_current_trace(
            tags=[
                f"abstract_id:{input_data.abstract_id}",
                f"prompt_version:{prompt_version}",
                f"model:{config.EXPLICIT_MODEL}",
                f"prompt_name:{EXTRACTION_TITLE_PROMPT_NAME}",
            ],
        )
        lf.update_current_generation(
            model=config.EXPLICIT_MODEL,
            metadata={
                "abstract_id": input_data.abstract_id,
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
        result: ExplicitLLMResponse = llm.invoke(messages, config=invoke_config)
        
        # Convert to Step4Output
        output = result.to_step4_output()
        return output
        
    except Exception as e:
        raise DrugClassExtractionError(f"Explicit extraction failed: {e}") from e
