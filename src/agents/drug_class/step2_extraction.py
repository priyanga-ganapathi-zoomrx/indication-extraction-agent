"""Step 2 Extraction: Drug class extraction using LLM.

Extracts drug classes from search results using:
a) Tavily-based extraction - uses pre-fetched search results
b) Grounded search - fallback using LLM's web_search_preview

This module exports SINGLE-DRUG functions. Loop/checkpointing is in pipeline.py.
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
    get_extraction_rules_prompt_parts,
    get_grounded_search_prompt_parts,
    EXTRACTION_RULES_PROMPT_NAME,
    GROUNDED_SEARCH_PROMPT_NAME,
)
from src.agents.drug_class.schemas import (
    DrugClassExtractionInput,
    DrugExtractionResult,
    DrugClassLLMResponse,
    GroundedSearchLLMResponse,
    DrugClassExtractionError,
)


# =============================================================================
# HELPERS
# =============================================================================

def _format_search_results(
    drug_class_results: list[dict],
    firm_results: list[dict],
) -> str:
    """Format combined search results for the LLM prompt.
    
    Args:
        drug_class_results: Results from drug class search
        firm_results: Results from firm search
        
    Returns:
        Formatted string with all search results
    """
    parts = []
    
    if drug_class_results:
        parts.append("## Drug Class Search Results\n")
        for i, result in enumerate(drug_class_results, 1):
            content = result.get("raw_content") or result.get("content", "")
            if len(content) > 5000:
                content = content[:5000] + "... [truncated]"
            parts.append(f"### Result {i}")
            parts.append(f"**URL**: {result.get('url', 'Unknown')}")
            parts.append(f"**Content**: {content}\n")
    
    if firm_results:
        parts.append("\n## Firm Search Results\n")
        for i, result in enumerate(firm_results, 1):
            content = result.get("raw_content") or result.get("content", "")
            if len(content) > 5000:
                content = content[:5000] + "... [truncated]"
            parts.append(f"### Result {i}")
            parts.append(f"**URL**: {result.get('url', 'Unknown')}")
            parts.append(f"**Content**: {content}\n")
    
    if not parts:
        return "No search results available."
    
    return "\n".join(parts)


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

@observe(as_type="generation", name="drug-class-step2-tavily")
def extract_with_tavily(input_data: DrugClassExtractionInput, callbacks: list = None) -> DrugExtractionResult:
    """Extract drug classes using Tavily search results.
    
    Uses LangChain's with_structured_output for reliable JSON parsing.
    Per-request timeout is 120s. Retries are handled by Temporal at the activity level.
    
    Args:
        input_data: DrugClassExtractionInput with drug and search results
        callbacks: Optional list of LangChain callback handlers (e.g., TokenUsageCallbackHandler)
        
    Returns:
        DrugExtractionResult with extracted classes
        
    Raises:
        DrugClassExtractionError: If extraction fails (triggers Temporal retry)
    """
    # Load and parse prompt
    system_prompt, rules_message, prompt_version = get_extraction_rules_prompt_parts()
    
    if not system_prompt:
        raise DrugClassExtractionError("Could not extract SYSTEM_PROMPT from prompt file")
    
    # Format search results
    formatted_results = _format_search_results(
        input_data.drug_class_results,
        input_data.firm_search_results,
    )
    
    # Build input message
    input_content = f"""# EXTRACTION INPUT

## Drug
{input_data.drug}

## Abstract Title
{input_data.abstract_title or "Not provided"}

## Full Abstract
{input_data.full_abstract or "Not provided"}

## Search Results

{formatted_results}"""
    
    # Create LLM with structured output (120s timeout for long-running requests)
    base_llm = create_llm(LLMConfig(
        api_key=settings.llm.LLM_API_KEY,
        base_url=settings.llm.LLM_BASE_URL,
        model=config.EXTRACTION_MODEL,
        temperature=config.EXTRACTION_TEMPERATURE,
        max_tokens=config.EXTRACTION_MAX_TOKENS,
        timeout=120,  # 2 minute timeout
    ))
    llm = base_llm.with_structured_output(DrugClassLLMResponse)
    
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
                f"drug:{input_data.drug}",
                f"prompt_version:{prompt_version}",
                f"model:{config.EXTRACTION_MODEL}",
                f"prompt_name:{EXTRACTION_RULES_PROMPT_NAME}",
                "source_type:tavily",
            ],
        )
        lf.update_current_generation(
            model=config.EXTRACTION_MODEL,
            metadata={
                "abstract_id": input_data.abstract_id,
                "drug": input_data.drug,
                "prompt_version": prompt_version,
            },
        )
        # Create LangChain callback handler linked to current trace
        langfuse_handler = CallbackHandler()
    
    all_callbacks = []
    if langfuse_handler:
        all_callbacks.append(langfuse_handler)
    if callbacks:
        all_callbacks.extend(callbacks)
    invoke_config = {"callbacks": all_callbacks} if all_callbacks else {}
    
    try:
        result: DrugClassLLMResponse = llm.invoke(messages, config=invoke_config)
        
        # Build extraction result
        drug_classes = result.drug_classes or ["NA"]
        success = bool(result.drug_classes and result.drug_classes != ["NA"])
        
        return DrugExtractionResult(
            drug_name=result.drug_name or input_data.drug,
            drug_classes=drug_classes,
            selected_sources=result.selected_sources,
            confidence_score=result.confidence_score,
            extraction_details=result.extraction_details,
            extraction_method="tavily",
            reasoning=result.reasoning,
            success=success,
        )
        
    except Exception as e:
        raise DrugClassExtractionError(f"Tavily extraction failed for {input_data.drug}: {e}") from e


@observe(as_type="generation", name="drug-class-step2-grounded")
def extract_with_grounded(input_data: DrugClassExtractionInput, callbacks: list = None) -> DrugExtractionResult:
    """Extract drug classes using LLM's grounded search (web_search_preview).
    
    Fallback method when Tavily returns no results or NA.
    Uses LangChain's with_structured_output for reliable JSON parsing.
    Per-request timeout is 120s. Retries are handled by Temporal at the activity level.
    
    Args:
        input_data: DrugClassExtractionInput with drug info
        callbacks: Optional list of LangChain callback handlers (e.g., TokenUsageCallbackHandler)
        
    Returns:
        DrugExtractionResult with extracted classes
        
    Raises:
        DrugClassExtractionError: If extraction fails (triggers Temporal retry)
    """
    # Load and parse prompt (includes fallback logic)
    system_prompt, rules_message, prompt_version = get_grounded_search_prompt_parts()
    
    # Build input message
    input_content = f"""# EXTRACTION INPUT

## Drug Name
{input_data.drug}

## Abstract Title
{input_data.abstract_title or "Not provided"}

## Full Abstract
{input_data.full_abstract or "Not provided"}"""
    
    # Create LLM with web search enabled and structured output (120s timeout)
    # Note: Uses GroundedSearchLLMResponse which has different format than Tavily
    base_llm = create_llm(
        LLMConfig(
            api_key=settings.llm.LLM_API_KEY,
            base_url=settings.llm.LLM_BASE_URL,
            model=config.GROUNDED_MODEL,
            temperature=config.GROUNDED_TEMPERATURE,
            max_tokens=config.GROUNDED_MAX_TOKENS,
            timeout=120,  # 2 minute timeout
        ),
        model_kwargs={"tools": [{"type": "web_search_preview"}]}
    )
    llm = base_llm.with_structured_output(GroundedSearchLLMResponse)
    
    # Build messages
    if config.ENABLE_PROMPT_CACHING:
        system_message = SystemMessage(content=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"}
        }])
        messages = [system_message]
        
        if rules_message:
            rules_msg = HumanMessage(content=[{
                "type": "text",
                "text": rules_message,
                "cache_control": {"type": "ephemeral"}
            }])
            messages.append(rules_msg)
    else:
        system_message = SystemMessage(content=system_prompt)
        messages = [system_message]
        
        if rules_message:
            rules_msg = HumanMessage(content=rules_message)
            messages.append(rules_msg)
    
    messages.append(HumanMessage(content=input_content))
    
    # Setup Langfuse callback and metadata if enabled
    langfuse_handler = None
    if is_langfuse_enabled():
        lf = get_client()
        lf.update_current_trace(
            tags=[
                f"abstract_id:{input_data.abstract_id}",
                f"drug:{input_data.drug}",
                f"prompt_version:{prompt_version}",
                f"model:{config.GROUNDED_MODEL}",
                f"prompt_name:{GROUNDED_SEARCH_PROMPT_NAME}",
                "source_type:grounded",
            ],
        )
        lf.update_current_generation(
            model=config.GROUNDED_MODEL,
            metadata={
                "abstract_id": input_data.abstract_id,
                "drug": input_data.drug,
                "prompt_version": prompt_version,
            },
        )
        # Create LangChain callback handler linked to current trace
        langfuse_handler = CallbackHandler()
    
    all_callbacks = []
    if langfuse_handler:
        all_callbacks.append(langfuse_handler)
    if callbacks:
        all_callbacks.extend(callbacks)
    invoke_config = {"callbacks": all_callbacks} if all_callbacks else {}
    
    try:
        result: GroundedSearchLLMResponse = llm.invoke(messages, config=invoke_config)
        
        # Convert grounded search format to standard DrugExtractionResult
        return result.to_extraction_result(input_data.drug)
        
    except Exception as e:
        raise DrugClassExtractionError(f"Grounded extraction failed for {input_data.drug}: {e}") from e
