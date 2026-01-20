"""Configuration for drug extraction agent.

Centralizes all drug-specific settings with environment variable support.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DrugConfig(BaseSettings):
    """Drug extraction and validation configuration settings.
    
    All settings can be overridden via environment variables with DRUG_ prefix.
    Example: DRUG_EXTRACTION_MODEL="anthropic/claude-sonnet-4-20250514"
             DRUG_VALIDATION_MODEL="openai/gpt-4o"
    """
    
    model_config = SettingsConfigDict(
        env_prefix="DRUG_",
        env_file=".env",
        extra="ignore",
    )
    
    # ==========================================================================
    # Extraction LLM settings
    # ==========================================================================
    EXTRACTION_MODEL: str = Field(
        default="gemini/gemini-2.5-pro",
        description="LLM model for drug extraction"
    )
    EXTRACTION_TEMPERATURE: float = Field(
        default=0,
        description="Temperature for extraction LLM"
    )
    EXTRACTION_MAX_TOKENS: int = Field(
        default=50000,
        description="Maximum tokens for extraction LLM"
    )
    
    # ==========================================================================
    # Validation LLM settings
    # ==========================================================================
    VALIDATION_MODEL: str = Field(
        default="gemini/gemini-3-flash-preview",
        description="LLM model for drug validation"
    )
    VALIDATION_TEMPERATURE: float = Field(
        default=1,
        description="Temperature for validation LLM"
    )
    VALIDATION_MAX_TOKENS: int = Field(
        default=50000,
        description="Maximum tokens for validation LLM"
    )
    
    # ==========================================================================
    # Prompt settings
    # ==========================================================================
    ENABLE_PROMPT_CACHING: bool = Field(
        default=True,
        description="Enable prompt caching (reduces LLM costs for Anthropic)"
    )


# Global config instance
config = DrugConfig()

