"""Configuration for drug class extraction.

Centralizes all settings with environment variable support.
Each step can have its own model configuration.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DrugClassConfig(BaseSettings):
    """Drug class extraction configuration settings.
    
    Override via environment variables with DRUG_CLASS_ prefix.
    Example: DRUG_CLASS_REGIMEN_MODEL="gemini/gemini-2.5-pro"
    """
    
    model_config = SettingsConfigDict(
        env_prefix="DRUG_CLASS_",
        env_file=".env",
        extra="ignore",
    )
    
    # ==========================================================================
    # Step 1: Regimen Identification
    # ==========================================================================
    REGIMEN_MODEL: str = Field(
        default="gemini/gemini-2.5-flash",
        description="Model for regimen identification (fast, simple task)"
    )
    REGIMEN_TEMPERATURE: float = Field(default=0)
    REGIMEN_MAX_TOKENS: int = Field(default=4096)
    
    # ==========================================================================
    # Step 2: Drug Class Extraction (Tavily + Grounded)
    # ==========================================================================
    EXTRACTION_MODEL: str = Field(
        default="gemini/gemini-2.5-pro",
        description="Model for drug class extraction (complex reasoning)"
    )
    EXTRACTION_TEMPERATURE: float = Field(default=0)
    EXTRACTION_MAX_TOKENS: int = Field(default=16384)
    
    GROUNDED_MODEL: str = Field(
        default="openai/gpt-4o-search-preview",
        description="Model for grounded search fallback (OpenAI with web search)"
    )
    GROUNDED_TEMPERATURE: float = Field(default=0)
    GROUNDED_MAX_TOKENS: int = Field(default=8192)
    
    # ==========================================================================
    # Step 3: Selection
    # ==========================================================================
    SELECTION_MODEL: str = Field(
        default="gemini/gemini-2.5-flash",
        description="Model for drug class selection"
    )
    SELECTION_TEMPERATURE: float = Field(default=0)
    SELECTION_MAX_TOKENS: int = Field(default=4096)
    
    # ==========================================================================
    # Step 4: Explicit Extraction
    # ==========================================================================
    EXPLICIT_MODEL: str = Field(
        default="gemini/gemini-2.5-flash",
        description="Model for explicit drug class extraction from title"
    )
    EXPLICIT_TEMPERATURE: float = Field(default=0)
    EXPLICIT_MAX_TOKENS: int = Field(default=4096)
    
    # ==========================================================================
    # Step 5: Consolidation
    # ==========================================================================
    CONSOLIDATION_MODEL: str = Field(
        default="gemini/gemini-2.5-flash",
        description="Model for consolidation"
    )
    CONSOLIDATION_TEMPERATURE: float = Field(default=0)
    CONSOLIDATION_MAX_TOKENS: int = Field(default=4096)
    
    # ==========================================================================
    # Validation
    # ==========================================================================
    VALIDATION_MODEL: str = Field(
        default="gemini/gemini-2.5-pro",
        description="Model for validation (requires strong reasoning)"
    )
    VALIDATION_TEMPERATURE: float = Field(default=0)
    VALIDATION_MAX_TOKENS: int = Field(default=16384)
    
    # ==========================================================================
    # Prompt settings
    # ==========================================================================
    ENABLE_PROMPT_CACHING: bool = Field(
        default=True,
        description="Enable prompt caching for Anthropic models"
    )
    
    # ==========================================================================
    # Tavily Search settings
    # ==========================================================================
    TAVILY_MAX_RESULTS: int = Field(default=3)
    TAVILY_SEARCH_DEPTH: str = Field(default="advanced")


# Global config instance
config = DrugClassConfig()

