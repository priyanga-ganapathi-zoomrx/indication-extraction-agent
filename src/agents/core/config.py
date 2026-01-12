"""Configuration settings for the calculator agent."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LangfuseSettings(BaseSettings):
    """Langfuse configuration settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LANGFUSE_PUBLIC_KEY: str = Field(default="", description="Langfuse public key")
    LANGFUSE_SECRET_KEY: str = Field(default="", description="Langfuse secret key")
    LANGFUSE_HOST: str = Field(
        default="https://cloud.langfuse.com", description="Langfuse host URL"
    )


class LLMSettings(BaseSettings):
    """LLM configuration settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_API_KEY: str = Field(default="", description="API key for LLM service")
    LLM_BASE_URL: str = Field(
        default="https://api.openai.com/v1", description="Base URL for LLM API"
    )
    LLM_MODEL: str = Field(
        default="anthropic/claude-sonnet-4-20250514",
        description="Model name (e.g., anthropic/claude-sonnet-4-20250514, openai/gpt-4)",
    )
    LLM_TEMPERATURE: float = Field(default=0.0, description="Temperature for LLM")
    LLM_MAX_TOKENS: int = Field(default=4096, description="Maximum tokens for LLM")


class TavilySettings(BaseSettings):
    """Tavily search API configuration settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    TAVILY_API_KEY: str = Field(default="", description="API key for Tavily search")
    TAVILY_MAX_RESULTS: int = Field(default=5, description="Maximum search results to return")


class AppSettings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Nested settings
    langfuse: LangfuseSettings = LangfuseSettings()
    llm: LLMSettings = LLMSettings()
    tavily: TavilySettings = TavilySettings()


# Global settings instance
settings = AppSettings()

