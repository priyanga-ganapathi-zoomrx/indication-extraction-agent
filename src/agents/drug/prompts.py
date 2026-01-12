"""Prompts module for drug extraction, validation, and verification agents."""

import os
from pathlib import Path
from typing import Optional

from langfuse import Langfuse


# Available prompt names
EXTRACTION_PROMPT_NAME = "DRUG_EXTRACTION_SYSTEM_PROMPT"
VALIDATION_PROMPT_NAME = "DRUG_VALIDATION_SYSTEM_PROMPT"
VERIFICATION_PROMPT_NAME = "DRUG_VERIFICATION_SYSTEM_PROMPT"

# Default prompts directory (relative to this file)
PROMPTS_DIR = Path(__file__).parent / "prompts"


def get_system_prompt(
    langfuse_client: Optional[Langfuse] = None,
    prompt_name: str = EXTRACTION_PROMPT_NAME,
    fallback_to_file: bool = True,
    prompt_dir: Optional[Path] = None,
) -> tuple[str, str]:
    """Load the system prompt from Langfuse or fallback to local file.

    Args:
        langfuse_client: Optional Langfuse client instance. If not provided, will create one using env vars.
        prompt_name: Name of the prompt in Langfuse (default: EXTRACTION_PROMPT_NAME)
        fallback_to_file: If True, fallback to reading from local file (prompt_name.md) if Langfuse fetch fails
        prompt_dir: Optional directory to look for prompt files (default: prompts/ in same folder)

    Returns:
        tuple[str, str]: A tuple of (prompt_content, prompt_version)

    Raises:
        Exception: If Langfuse fetch fails and fallback_to_file is False
    """
    # Try to fetch from Langfuse
    try:
        # Use provided client or create a new one
        client = langfuse_client or Langfuse()
        
        # Fetch the prompt from Langfuse
        print(f"ℹ Fetching prompt '{prompt_name}' from Langfuse...")
        langfuse_prompt = client.get_prompt(prompt_name)
        
        # Get the prompt content
        if hasattr(langfuse_prompt, 'prompt'):
            content = langfuse_prompt.prompt
        elif hasattr(langfuse_prompt, 'get_langchain_prompt'):
            content = langfuse_prompt.get_langchain_prompt()
        else:
            content = str(langfuse_prompt)
        
        # Get the version
        version = str(langfuse_prompt.version) if hasattr(langfuse_prompt, 'version') else "unknown"
        
        print(f"✓ Successfully fetched prompt from Langfuse (version: {version})")
        return content.strip(), version
        
    except Exception as e:
        print(f"✗ Error fetching prompt from Langfuse: {e}")
        
        if not fallback_to_file:
            raise
        
        # Fallback to local file
        prompt_filename = f"{prompt_name}.md"
        prompts_directory = prompt_dir or PROMPTS_DIR
        print(f"ℹ Falling back to local {prompt_filename} file...")
        
        try:
            prompt_file = prompts_directory / prompt_filename
            
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print("✓ Successfully loaded prompt from local file")
            return content.strip(), "local"
        except Exception as file_error:
            raise Exception(
                f"Failed to fetch prompt from Langfuse and local file: "
                f"Langfuse error: {e}, File error: {file_error}"
            )


def get_extraction_prompt(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load the drug extraction system prompt.

    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails

    Returns:
        tuple[str, str]: A tuple of (prompt_content, prompt_version)
    """
    return get_system_prompt(
        langfuse_client=langfuse_client,
        prompt_name=EXTRACTION_PROMPT_NAME,
        fallback_to_file=fallback_to_file,
    )


def get_validation_prompt(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load the drug validation system prompt.

    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails

    Returns:
        tuple[str, str]: A tuple of (prompt_content, prompt_version)
    """
    return get_system_prompt(
        langfuse_client=langfuse_client,
        prompt_name=VALIDATION_PROMPT_NAME,
        fallback_to_file=fallback_to_file,
    )


def get_verification_prompt(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load the drug verification system prompt.

    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails

    Returns:
        tuple[str, str]: A tuple of (prompt_content, prompt_version)
    """
    return get_system_prompt(
        langfuse_client=langfuse_client,
        prompt_name=VERIFICATION_PROMPT_NAME,
        fallback_to_file=fallback_to_file,
    )

