"""Configuration for LLM Provider Comparison Experiment.

Uses the same settings as the main project from src/config.py.
Reads LLM_API_KEY, LLM_BASE_URL, and Langfuse settings from .env file.
"""

import sys
import os

# Add project root to path to import src.config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import settings
from src.langfuse_config import get_langfuse_config, LangfuseConfig

# Default model to use for comparison
DEFAULT_MODEL = "gemini/gemini-3-flash-preview"

# LLM parameters from project settings
DEFAULT_TEMPERATURE = settings.llm.LLM_TEMPERATURE
DEFAULT_MAX_TOKENS = settings.llm.LLM_MAX_TOKENS

# Enable Langfuse by default if configured
DEFAULT_ENABLE_LANGFUSE = True


def get_api_key() -> str:
    """Get the LLM API key from project settings.
    
    Uses LLM_API_KEY from .env file via src.config.settings.
    
    Returns:
        str: The API key
        
    Raises:
        ValueError: If no API key is found
    """
    api_key = settings.llm.LLM_API_KEY
    if not api_key:
        raise ValueError(
            "No API key found. Set LLM_API_KEY in your .env file."
        )
    return api_key


def get_base_url() -> str:
    """Get the LLM base URL from project settings.
    
    Uses LLM_BASE_URL from .env file via src.config.settings.
    This typically points to the LiteLLM proxy server.
    
    Returns:
        str: The base URL
    """
    return settings.llm.LLM_BASE_URL


def get_langfuse_settings() -> LangfuseConfig | None:
    """Get Langfuse configuration from project settings.
    
    Returns:
        LangfuseConfig: Langfuse settings, or None if not configured
    """
    return get_langfuse_config()


def setup_langfuse_env() -> bool:
    """Setup Langfuse environment variables for tracing.
    
    Sets LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST
    environment variables required by both LangChain and LiteLLM integrations.
    
    Returns:
        bool: True if Langfuse is configured and env vars are set, False otherwise
    """
    langfuse_config = get_langfuse_settings()
    if not langfuse_config:
        return False
    
    os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_config.public_key
    os.environ["LANGFUSE_SECRET_KEY"] = langfuse_config.secret_key
    os.environ["LANGFUSE_HOST"] = langfuse_config.host
    return True

