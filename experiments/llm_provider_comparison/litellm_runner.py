"""LiteLLM runner for LLM Provider Comparison Experiment.

Uses LiteLLM's completion API with the LiteLLM proxy server.
Reference: https://docs.litellm.ai/docs/providers/gemini

Google Search Grounding:
- Uses googleSearch tool for real-time web search grounding
- Reference: https://docs.litellm.ai/docs/providers/gemini#google-search-tool

Langfuse Integration:
- Uses litellm.callbacks = ["langfuse_otel"] for OTEL tracing
- Set enable_langfuse=True to enable observability

Response Schema:
- Uses Pydantic model with response_format to enforce structured JSON output
- Reference: https://docs.litellm.ai/docs/providers/gemini#response-schema
"""

import time
from typing import Dict, List, Literal

import litellm
from litellm import completion
from pydantic import BaseModel, Field

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
from experiments.llm_provider_comparison.langchain_runner import RunResult


# =============================================================================
# Pydantic Response Schema (matches the JSON structure in SYSTEM_PROMPT)
# Reference: https://docs.litellm.ai/docs/providers/gemini#response-schema
# =============================================================================

class PrimaryClassification(BaseModel):
    """Primary drug class classification."""
    drug_class: str = Field(description="Main drug class")
    class_type: Literal["MoA", "Chemical", "Therapeutic", "Mode"] = Field(
        description="Type of classification"
    )
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level")
    evidence: str = Field(description="Supporting evidence")


class SecondaryClassification(BaseModel):
    """Secondary drug class classification."""
    drug_class: str = Field(description="Additional class")
    class_type: str = Field(description="Type of classification")
    relationship: str = Field(description="How it relates to primary")


class MechanismOfAction(BaseModel):
    """Mechanism of action details."""
    target: str = Field(description="Molecular target")
    pathway: str = Field(description="Biological pathway")
    effect: str = Field(description="Therapeutic effect")


class ClinicalContext(BaseModel):
    """Clinical context information."""
    approved_indications: List[str] = Field(description="Approved indications")
    common_combinations: List[str] = Field(description="Common drug combinations")
    administration_route: Literal["oral", "IV", "subcutaneous", "intramuscular", "topical", "other"] = Field(
        description="Route of administration"
    )


class Analysis(BaseModel):
    """Complete drug class analysis."""
    primary_classification: PrimaryClassification
    secondary_classifications: List[SecondaryClassification] = Field(default_factory=list)
    mechanism_of_action: MechanismOfAction
    clinical_context: ClinicalContext


class QualityAssessment(BaseModel):
    """Quality assessment of the analysis."""
    completeness: float = Field(ge=0.0, le=1.0, description="Completeness score 0.0-1.0")
    source_reliability: Literal["high", "medium", "low"] = Field(description="Source reliability")
    classification_certainty: Literal["definitive", "probable", "uncertain"] = Field(
        description="Classification certainty"
    )


class Reference(BaseModel):
    """Source reference from web search grounding."""
    title: str = Field(description="Source title")
    url: str = Field(description="Source URL")
    snippet: str = Field(description="Relevant excerpt from the source")


class DrugClassificationResponse(BaseModel):
    """Complete drug classification response schema.
    
    This Pydantic model enforces the exact JSON structure expected from the LLM.
    Used with LiteLLM's response_format parameter for Gemini models.
    Includes references and search_evidence fields for web search grounding.
    """
    drug_name: str = Field(description="Name of the drug")
    analysis: Analysis = Field(description="Complete drug class analysis")
    references: List[Reference] = Field(default_factory=list, description="Source references from web search")
    search_evidence: str = Field(default="", description="Summary of key findings from web search")
    quality_assessment: QualityAssessment = Field(description="Quality assessment")
    reasoning: str = Field(description="Step-by-step reasoning with citations")

# Track if Langfuse has been initialized for LiteLLM
_langfuse_initialized = False


def _init_langfuse_for_litellm() -> bool:
    """Initialize Langfuse for LiteLLM OTEL tracing.
    
    Sets up environment variables and configures litellm.callbacks for tracing.
    
    Returns:
        bool: True if Langfuse is configured, False otherwise
    """
    global _langfuse_initialized
    if _langfuse_initialized:
        return True
    
    if setup_langfuse_env():
        litellm.callbacks = ["langfuse_otel"]
        _langfuse_initialized = True
        print("✓ Langfuse OTEL integration configured for LiteLLM")
        return True
    else:
        print("ℹ Langfuse not configured (missing keys) - LiteLLM tracing disabled")
        return False


def run_litellm(
    drug: str,
    context: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_langfuse: bool = DEFAULT_ENABLE_LANGFUSE,
) -> RunResult:
    """Run a single request using LiteLLM completion via proxy.
    
    Uses the LiteLLM proxy server configured in LLM_BASE_URL.
    
    Args:
        drug: Drug name to classify
        context: Clinical context for the drug
        model: Model name to use (e.g., "gemini/gemini-3-flash-preview")
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
            langfuse_active = _init_langfuse_for_litellm()
        
        # Get API key and base URL from project settings
        api_key = get_api_key()
        base_url = get_base_url()
        
        # Build messages
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_message(drug, context)},
        ]
        
        # Build metadata for Langfuse tracing
        metadata = {
            "drug": drug,
            "context": context,
            "provider": "litellm",
            "experiment": "llm_provider_comparison",
            "tags": ["llm_comparison", "litellm", drug, "web_search"],
        }
        
        # Google Search tool for web grounding
        # Reference: https://docs.litellm.ai/docs/providers/gemini#google-search-tool
        tools = [{"googleSearch": {}}]
        
        # Time the request
        # Use response_format with Pydantic model to enforce structured JSON output
        # Reference: https://docs.litellm.ai/docs/providers/gemini#response-schema
        start_time = time.perf_counter()
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_base=base_url,
            tools=tools,  # Enable Google Search grounding
            # response_format=DrugClassificationResponse,  # Enforce JSON schema
            metadata=metadata if langfuse_active else None,
        )
        end_time = time.perf_counter()
        
        response_time_ms = (end_time - start_time) * 1000
        
        # Extract content
        content = response.choices[0].message.content if response.choices else ""
        
        # Extract token usage - handle both object and dict formats
        usage = getattr(response, "usage", None)
        
        if usage is not None:
            # Try object attribute access first, then dict access
            if hasattr(usage, "prompt_tokens"):
                prompt_tokens = usage.prompt_tokens or 0
            elif isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0)
            else:
                prompt_tokens = 0
                
            if hasattr(usage, "completion_tokens"):
                completion_tokens = usage.completion_tokens or 0
            elif isinstance(usage, dict):
                completion_tokens = usage.get("completion_tokens", 0)
            else:
                completion_tokens = 0
                
            if hasattr(usage, "total_tokens"):
                total_tokens = usage.total_tokens or 0
            elif isinstance(usage, dict):
                total_tokens = usage.get("total_tokens", 0)
            else:
                total_tokens = prompt_tokens + completion_tokens
        else:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
        
        return RunResult(
            content=content,
            response_time_ms=response_time_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            provider="litellm",
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
            provider="litellm",
            success=False,
            error=str(e),
            langfuse_enabled=langfuse_active,
        )


def run_litellm_batch(
    inputs: list[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    enable_langfuse: bool = DEFAULT_ENABLE_LANGFUSE,
) -> list[RunResult]:
    """Run multiple requests using LiteLLM.
    
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
        result = run_litellm(
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
    
    print("Testing LiteLLM runner...")
    result = run_litellm(
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

