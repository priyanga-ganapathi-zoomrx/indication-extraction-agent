"""Simplified LLM setup for creating language model instances.

This module provides a simple way to create LLM instances with proper configuration
and multi-provider support (OpenAI, Anthropic, Google Gemini, etc.).
"""

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from src.agents.core.config import settings


class LLMConfig(BaseModel):
    """Configuration for a language model instance.

    Attributes:
        api_key: API key for the LLM service
        model: Model name (e.g., 'anthropic/claude-sonnet-4-20250514', 'openai/gpt-4')
        temperature: Temperature for generation (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        base_url: Base URL for the LLM API
        timeout: Timeout in seconds for API requests
    """

    api_key: str = Field(..., description="API key for the LLM service")
    model: str = Field(..., description="Name of the model to use")
    temperature: float = Field(default=0.0, description="Temperature (0.0 to 2.0)")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    base_url: str = Field(
        default=settings.llm.LLM_BASE_URL, description="Base URL for the API"
    )
    timeout: int = Field(default=90, description="Timeout in seconds")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0.0 and 2.0."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Validate max_tokens is in reasonable range."""
        if v <= 0 or v > 100000:
            raise ValueError("max_tokens must be between 1 and 100000")
        return v

    @property
    def provider(self) -> str:
        """Extract provider from model name.
        
        Examples:
            'anthropic/claude-sonnet-4' -> 'anthropic'
            'openai/gpt-4' -> 'openai'
            'google/gemini-pro' -> 'google'
            'gpt-4' -> 'openai' (default)
        """
        model_value = str(self.model)
        return model_value.split("/")[0].lower() if "/" in model_value else "openai"
    
    @property
    def model_name(self) -> str:
        """Extract model name without provider prefix.
        
        Examples:
            'anthropic/claude-sonnet-4' -> 'claude-sonnet-4'
            'google/gemini-pro' -> 'gemini-pro'
            'gpt-4' -> 'gpt-4'
        """
        model_value = str(self.model)
        return model_value.split("/")[1] if "/" in model_value else model_value


def create_llm(
    llm_config: LLMConfig,
    model_kwargs: dict = None,
) -> ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI:
    """Create an LLM instance based on the configuration.

    Args:
        llm_config: Configuration for the LLM
        model_kwargs: Optional dictionary of additional model arguments
                      (e.g., {"tools": [{"type": "web_search_preview"}]} for OpenAI grounded search)

    Returns:
        ChatOpenAI, ChatAnthropic, or ChatGoogleGenerativeAI instance

    Raises:
        Exception: If LLM creation fails
    """
    try:
        if llm_config.provider == "anthropic":
            return ChatAnthropic(
                model=llm_config.model,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                timeout=llm_config.timeout,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
            )
        elif llm_config.provider == "google":
            # Google Gemini
            return ChatGoogleGenerativeAI(
                model=llm_config.model_name,
                google_api_key=llm_config.api_key,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
            )
        else:
            # OpenAI or OpenAI-compatible providers
            return ChatOpenAI(
                model=llm_config.model,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                timeout=llm_config.timeout,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                model_kwargs=model_kwargs or {},
            )
    except Exception as e:
        print(f"✗ Error creating LLM: {e}")
        raise

