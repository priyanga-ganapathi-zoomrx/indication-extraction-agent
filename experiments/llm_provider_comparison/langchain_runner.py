"""LangChain ChatOpenAI runner for LLM Provider Comparison Experiment.

Uses LangChain's ChatOpenAI with Google's OpenAI-compatible endpoint to call Gemini models.
Reference: https://docs.langchain.com/oss/python/integrations/chat/openai

Google Search Grounding:
- Uses googleSearch tool for real-time web search grounding via LiteLLM proxy
- Reference: https://docs.litellm.ai/docs/providers/gemini#google-search-tool

Langfuse Integration:
- Uses langfuse.langchain.CallbackHandler for tracing
- Set enable_langfuse=True to enable observability
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from experiments.llm_provider_comparison.config import (
    DEFAULT_ENABLE_LANGFUSE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    get_api_key,
    get_base_url,
    setup_langfuse_env,
)
from experiments.llm_provider_comparison.prompts import SYSTEM_PROMPT, format_user_message

# Track if Langfuse has been initialized for LangChain
_langfuse_initialized = False


@dataclass
class RunResult:
    """Result from a single LLM run."""
    
    content: str
    response_time_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    provider: str
    success: bool
    error: Optional[str] = None
    raw_response: Optional[Any] = None
    langfuse_enabled: bool = False


def _init_langfuse_for_langchain() -> bool:
    """Initialize Langfuse for LangChain tracing.
    
    Sets up environment variables required by langfuse.langchain.CallbackHandler.
    
    Returns:
        bool: True if Langfuse is configured, False otherwise
    """
    global _langfuse_initialized
    if _langfuse_initialized:
        return True
    
    if setup_langfuse_env():
        _langfuse_initialized = True
        print("✓ Langfuse configured for LangChain (CallbackHandler)")
        return True
    else:
        print("ℹ Langfuse not configured (missing keys) - LangChain tracing disabled")
        return False


def create_langchain_client(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_web_search: bool = True,
) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI client using configurable base URL.
    
    Uses ChatOpenAI with base_url from LLM_BASE_URL env variable.
    Falls back to Google's OpenAI-compatible API endpoint if not set.
    Reference: https://docs.langchain.com/oss/python/integrations/chat/openai
    
    Google Search grounding is enabled by default via model_kwargs.
    Reference: https://docs.litellm.ai/docs/providers/gemini#google-search-tool
    
    Args:
        model: Model name (e.g., "gemini/gemini-3-flash-preview")
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        enable_web_search: Enable Google Search grounding (default: True)
        
    Returns:
        Configured ChatOpenAI instance
    """
    api_key = get_api_key()
    base_url = get_base_url()
    
    # Extract model name without provider prefix for the API call
    model_name = model.split("/")[-1] if "/" in model else model
    
    # Google Search tool for web grounding (passed through LiteLLM proxy)
    # Reference: https://docs.litellm.ai/docs/providers/gemini#google-search-tool
    model_kwargs = {}
    if enable_web_search:
        model_kwargs["tools"] = [{"type": "web_search_preview"}]
    
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs,
    )


def run_langchain(
    drug: str,
    context: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_langfuse: bool = DEFAULT_ENABLE_LANGFUSE,
) -> RunResult:
    """Run a single request using LangChain ChatOpenAI.
    
    Args:
        drug: Drug name to classify
        context: Clinical context for the drug
        model: Model name to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        enable_langfuse: Enable Langfuse tracing (default: True if configured)
        
    Returns:
        RunResult with metrics and response content
    """
    langfuse_active = False
    
    try:
        # Initialize Langfuse if enabled
        if enable_langfuse:
            langfuse_active = _init_langfuse_for_langchain()
        
        # Create client
        llm = create_langchain_client(model, temperature, max_tokens)
        
        # Build messages
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=format_user_message(drug, context)),
        ]
        
        # Setup RunnableConfig with Langfuse callback if enabled
        config = None
        if langfuse_active:
            # Tags and metadata are passed via RunnableConfig, not CallbackHandler constructor
            langfuse_tags = ["llm_comparison", "langchain", drug, "web_search"]
            langfuse_metadata = {
                "drug": drug,
                "context": context,
                "model": model,
                "provider": "langchain",
                "experiment": "llm_provider_comparison",
                "web_search_enabled": True,
            }
            config = RunnableConfig(
                callbacks=[LangfuseCallbackHandler()],
                metadata={
                    "langfuse_tags": langfuse_tags,
                    **langfuse_metadata,
                },
            )
        
        # Time the request
        start_time = time.perf_counter()
        response = llm.invoke(messages, config=config)
        end_time = time.perf_counter()
        
        response_time_ms = (end_time - start_time) * 1000
        
        # Extract token usage from ChatOpenAI response
        # ChatOpenAI returns usage_metadata with input_tokens, output_tokens, total_tokens
        # or response_metadata with token_usage dict
        usage_metadata = getattr(response, "usage_metadata", {}) or {}
        response_metadata = getattr(response, "response_metadata", {}) or {}
        token_usage = response_metadata.get("token_usage", {})
        
        # Try usage_metadata first (LangChain standard), then response_metadata
        prompt_tokens = (
            usage_metadata.get("input_tokens") or 
            token_usage.get("prompt_tokens") or 
            0
        )
        completion_tokens = (
            usage_metadata.get("output_tokens") or 
            token_usage.get("completion_tokens") or 
            0
        )
        total_tokens = (
            usage_metadata.get("total_tokens") or 
            token_usage.get("total_tokens") or 
            prompt_tokens + completion_tokens
        )
        
        return RunResult(
            content=response.content,
            response_time_ms=response_time_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            provider="langchain",
            success=True,
            raw_response=response,
            langfuse_enabled=langfuse_active,
        )
        
    except Exception as e:
        return RunResult(
            content="",
            response_time_ms=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            model=model,
            provider="langchain",
            success=False,
            error=str(e),
            langfuse_enabled=langfuse_active,
        )


def run_langchain_batch(
    inputs: list[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_langfuse: bool = DEFAULT_ENABLE_LANGFUSE,
) -> list[RunResult]:
    """Run multiple requests using LangChain.
    
    Args:
        inputs: List of dicts with 'drug' and 'context' keys
        model: Model name to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        enable_langfuse: Enable Langfuse tracing (default: True if configured)
        
    Returns:
        List of RunResult objects
    """
    results = []
    for input_data in inputs:
        result = run_langchain(
            drug=input_data["drug"],
            context=input_data["context"],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_langfuse=enable_langfuse,
        )
        results.append(result)
    return results


if __name__ == "__main__":
    # Quick test
    from experiments.llm_provider_comparison.prompts import TEST_INPUTS
    
    print("Testing LangChain runner...")
    result = run_langchain(
        drug=TEST_INPUTS[0]["drug"],
        context=TEST_INPUTS[0]["context"],
    )
    
    print(f"\nProvider: {result.provider}")
    print(f"Success: {result.success}")
    print(f"Response time: {result.response_time_ms:.2f} ms")
    print(f"Prompt tokens: {result.prompt_tokens}")
    print(f"Completion tokens: {result.completion_tokens}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"\nContent preview: {result.content[:500]}...")

