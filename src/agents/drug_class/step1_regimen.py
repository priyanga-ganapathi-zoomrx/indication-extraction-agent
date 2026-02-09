from langfuse import observe, get_client
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
def identify_regimen(input_data: RegimenInput, callbacks: list = None) -> list[str]:
    """Identify if a drug is a regimen and extract its components.
    
    Uses LangChain's with_structured_output for reliable JSON parsing.
    Per-request timeout is 120s. Retries are handled by Temporal at the activity level.
    
    Args:
        input_data: RegimenInput with abstract_id, abstract_title, drug
        callbacks: Optional list of LangChain callback handlers (e.g., TokenUsageCallbackHandler)
        
    Returns:
        List of component drugs. If not a regimen, returns [drug].
        
    Raises:
        DrugClassExtractionError: If LLM call fails (triggers Temporal retry)
    """
    # Load prompt
    system_prompt, prompt_version = get_regimen_identification_prompt()
    
    # Create LLM with structured output (120s timeout for long-running requests)
    base_llm = create_llm(LLMConfig(
        api_key=settings.llm.LLM_API_KEY,
        base_url=settings.llm.LLM_BASE_URL,
        model=config.REGIMEN_MODEL,
        temperature=config.REGIMEN_TEMPERATURE,
        max_tokens=config.REGIMEN_MAX_TOKENS,
        timeout=120,  # 2 minute timeout
    ))
    llm = base_llm.with_structured_output(RegimenLLMResponse)
    
    # Build messages
    system_message = SystemMessage(content=system_prompt)
    
    user_content = f"""Abstract Title: {input_data.abstract_title or "Not provided"}
Drug: {input_data.drug}"""
    
    user_message = HumanMessage(content=user_content)
    
    # Setup Langfuse callback and metadata if enabled
    langfuse_handler = None
    if is_langfuse_enabled():
        lf = get_client()
        lf.update_current_trace(
            tags=[
                f"abstract_id:{input_data.abstract_id}",
                f"drug:{input_data.drug}",
                f"prompt_version:{prompt_version}",
                f"model:{config.REGIMEN_MODEL}",
                f"prompt_name:{REGIMEN_IDENTIFICATION_PROMPT_NAME}",
            ],
        )
        lf.update_current_generation(
            model=config.REGIMEN_MODEL,
            metadata={
                "abstract_id": input_data.abstract_id,
                "drug": input_data.drug,
                "prompt_version": prompt_version,
            },
        )
        # Create LangChain callback handler linked to current trace
        langfuse_handler = CallbackHandler()
    
    # Invoke LLM with structured output and callbacks (Langfuse + token tracking)
    all_callbacks = []
    if langfuse_handler:
        all_callbacks.append(langfuse_handler)
    if callbacks:
        all_callbacks.extend(callbacks)
    invoke_config = {"callbacks": all_callbacks} if all_callbacks else {}
    
    try:
        result: RegimenLLMResponse = llm.invoke([system_message, user_message], config=invoke_config)
        
        return result.components or [input_data.drug]
        
    except Exception as e:
        raise DrugClassExtractionError(f"Regimen identification failed for {input_data.drug}: {e}") from e
