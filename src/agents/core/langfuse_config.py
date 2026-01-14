"""Langfuse configuration and singleton client.

Provides a single Langfuse instance for use across all agents.
"""

from pydantic import BaseModel, Field
from langfuse import Langfuse

from src.agents.core.config import settings


# =============================================================================
# Singleton Langfuse Client
# =============================================================================

def _create_langfuse_client() -> Langfuse | None:
    """Create Langfuse client if configured.
    
    Returns:
        Langfuse client instance, or None if not configured
    """
    if not settings.langfuse.LANGFUSE_PUBLIC_KEY or not settings.langfuse.LANGFUSE_SECRET_KEY:
        return None
    
    return Langfuse(
        public_key=settings.langfuse.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.langfuse.LANGFUSE_SECRET_KEY,
        host=settings.langfuse.LANGFUSE_HOST,
    )


# Single global Langfuse instance
# Import this wherever you need Langfuse:
#   from src.agents.core.langfuse_config import langfuse
langfuse = _create_langfuse_client()


def is_langfuse_enabled() -> bool:
    """Check if Langfuse is configured and available."""
    return langfuse is not None


# =============================================================================
# Backward Compatibility (for drug_class and other modules)
# =============================================================================

class LangfuseConfig(BaseModel):
    """A data class for storing Langfuse configuration settings.
    
    DEPRECATED: Use `langfuse` singleton instead.
    Kept for backward compatibility with drug_class and other modules.
    """
    public_key: str = Field(..., description="Langfuse public key")
    secret_key: str = Field(..., description="Langfuse secret key")
    host: str = Field(
        default=settings.langfuse.LANGFUSE_HOST, description="Langfuse host URL"
    )


def get_langfuse_config() -> LangfuseConfig | None:
    """Get Langfuse configuration from environment settings.
    
    DEPRECATED: Use `langfuse` singleton and `is_langfuse_enabled()` instead.
    Kept for backward compatibility with drug_class and other modules.
    
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
