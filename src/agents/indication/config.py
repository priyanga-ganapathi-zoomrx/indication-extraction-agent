"""Configuration for indication extraction agent.

Centralizes all indication-specific settings with environment variable support.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IndicationConfig(BaseSettings):
    """Indication extraction and validation configuration settings.
    
    All settings can be overridden via environment variables with INDICATION_ prefix.
    Example: INDICATION_LLM_MODEL="gemini/gemini-2.5-pro"
             INDICATION_VALIDATION_LLM_MODEL="anthropic/claude-sonnet-4-20250514"
    """
    
    model_config = SettingsConfigDict(
        env_prefix="INDICATION_",
        env_file=".env",
        extra="ignore",
    )
    
    # ==========================================================================
    # Extraction LLM settings
    # ==========================================================================
    LLM_MODEL: str = Field(
        default="gemini/gemini-2.5-pro",
        description="LLM model for indication extraction"
    )
    LLM_TEMPERATURE: float = Field(
        default=0,
        description="Temperature for extraction LLM"
    )
    LLM_MAX_TOKENS: int = Field(
        default=50000,
        description="Maximum tokens for extraction LLM"
    )
    
    # ==========================================================================
    # Validation LLM settings
    # ==========================================================================
    VALIDATION_LLM_MODEL: str = Field(
        default="anthropic/claude-sonnet-4-5-20250929",
        description="LLM model for indication validation"
    )
    VALIDATION_LLM_TEMPERATURE: float = Field(
        default=0,
        description="Temperature for validation LLM"
    )
    VALIDATION_LLM_MAX_TOKENS: int = Field(
        default=50000,
        description="Maximum tokens for validation LLM"
    )
    
    # ==========================================================================
    # Paths
    # ==========================================================================
    RULES_PATH: Path = Field(
        default=Path("data/indication/rules/indication_extraction_rules.csv"),
        description="Path to indication extraction rules CSV"
    )
    
    # ==========================================================================
    # Agent settings
    # ==========================================================================
    ENABLE_PROMPT_CACHING: bool = Field(
        default=True,
        description="Enable prompt caching (reduces LLM costs)"
    )
    
    RECURSION_LIMIT: int = Field(
        default=100,
        description="LangGraph recursion limit"
    )


# Global config instance
config = IndicationConfig()

