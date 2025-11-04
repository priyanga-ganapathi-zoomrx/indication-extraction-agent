"""Langfuse configuration module."""

from pydantic import BaseModel, Field

from src.config import settings


class LangfuseConfig(BaseModel):
    """A data class for storing Langfuse configuration settings.

    This class centralizes the configuration for the Langfuse client, used for
    observability and tracing.

    Attributes:
        public_key (str): The public key for Langfuse.
        secret_key (str): The secret key for Langfuse.
        host (str): The host URL for the Langfuse server.
    """

    public_key: str = Field(..., description="Langfuse public key")
    secret_key: str = Field(..., description="Langfuse secret key")
    host: str = Field(
        default=settings.langfuse.LANGFUSE_HOST, description="Langfuse host URL"
    )


def get_langfuse_config() -> LangfuseConfig | None:
    """Get Langfuse configuration from environment settings.

    Returns:
        LangfuseConfig: Configured Langfuse settings, or None if not configured
    """
    if not settings.langfuse.LANGFUSE_PUBLIC_KEY or not settings.langfuse.LANGFUSE_SECRET_KEY:
        return None

    return LangfuseConfig(
        public_key=settings.langfuse.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.langfuse.LANGFUSE_SECRET_KEY,
        host=settings.langfuse.LANGFUSE_HOST,
    )

