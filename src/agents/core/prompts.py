"""Generic prompt loading utilities for all agents."""

from pathlib import Path
from typing import Optional

from langfuse import Langfuse

from src.agents.core.config import settings


def load_prompt(
    prompt_name: str,
    prompts_dir: Path,
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load prompt from Langfuse or fallback to local file.

    Args:
        prompt_name: Name of the prompt in Langfuse (also used as filename without extension)
        prompts_dir: Directory containing local prompt files
        langfuse_client: Optional Langfuse client. If None, creates one using settings.
        fallback_to_file: If True, fallback to local file on Langfuse failure

    Returns:
        tuple[str, str]: (prompt_content, version) - version is "local" if from file

    Raises:
        Exception: If both Langfuse and file loading fail
    """
    # Skip Langfuse if not configured
    if not langfuse_client and not settings.langfuse.LANGFUSE_PUBLIC_KEY:
        return _load_from_file(prompt_name, prompts_dir)

    # Try Langfuse
    try:
        client = langfuse_client or Langfuse(
            public_key=settings.langfuse.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.langfuse.LANGFUSE_SECRET_KEY,
            host=settings.langfuse.LANGFUSE_HOST,
        )
        print(f"ℹ Fetching prompt '{prompt_name}' from Langfuse...")
        langfuse_prompt = client.get_prompt(prompt_name)

        if hasattr(langfuse_prompt, "prompt"):
            content = langfuse_prompt.prompt
        elif hasattr(langfuse_prompt, "get_langchain_prompt"):
            content = langfuse_prompt.get_langchain_prompt()
        else:
            content = str(langfuse_prompt)

        version = str(getattr(langfuse_prompt, "version", "unknown"))
        print(f"✓ Loaded prompt from Langfuse (version: {version})")
        return content.strip(), version

    except Exception as e:
        print(f"✗ Langfuse error: {e}")

        if not fallback_to_file:
            raise

        return _load_from_file(prompt_name, prompts_dir)


def _load_from_file(prompt_name: str, prompts_dir: Path) -> tuple[str, str]:
    """Load prompt from local file."""
    prompt_file = prompts_dir / f"{prompt_name}.md"
    print(f"ℹ Loading prompt from {prompt_file}...")
    content = prompt_file.read_text(encoding="utf-8")
    print("✓ Loaded prompt from local file")
    return content.strip(), "local"
