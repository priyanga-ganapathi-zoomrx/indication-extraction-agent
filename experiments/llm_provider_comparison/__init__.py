"""LLM Provider Comparison Experiment.

This module compares Gemini model performance between:
1. LangChain ChatOpenAI - Using OpenAI-compatible interface
2. LiteLLM completion - Direct LiteLLM API

Metrics compared:
- Response time
- Token consumption (prompt, completion, total)
"""

from experiments.llm_provider_comparison.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from experiments.llm_provider_comparison.prompts import SYSTEM_PROMPT, TEST_INPUTS

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "SYSTEM_PROMPT",
    "TEST_INPUTS",
]

