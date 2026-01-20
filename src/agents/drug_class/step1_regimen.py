from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_fixed

from src.agents.core import settings, create_llm, LLMConfig
from src.agents.core.langfuse_config import is_langfuse_enabled
from src.agents.drug_class.config import config
from src.agents.drug_class.prompts import get_regimen_identification_prompt, REGIMEN_IDENTIFICATION_PROMPT_NAME
from src.agents.drug_class.schemas import (
    RegimenInput,
    RegimenLLMResponse,
    DrugClassExtractionError,
)


@retry(
    stop=stop_after_attempt(2),  # 1 initial + 1 retry
    wait=wait_fixed(1),  # 1 second between retries
    retry=retry_if_exception_type((TimeoutError, ConnectionError, Exception)),
    reraise=True,
)
@observe(as_type="generation", name="drug-class-step1-regimen")
def identify_regimen(input_data: RegimenInput) -> list[str]:
    """Identify if a drug is a regimen and extract its components.
    Args:
        input_data: RegimenInput with abstract_id, abstract_title, drug
        
    Returns:
        List of component drugs. If not a regimen, returns [drug].
        
    Raises:
        DrugClassExtractionError: If LLM call fails
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
    
    # Invoke LLM with structured output and Langfuse callback
    invoke_config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    
    try:
        result: RegimenLLMResponse = llm.invoke([system_message, user_message], config=invoke_config)
        
        return result.components or [input_data.drug]
        
    except Exception as e:
        raise DrugClassExtractionError(f"Regimen identification failed for {input_data.drug}: {e}") from e
