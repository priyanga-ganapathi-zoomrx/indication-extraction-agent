"""Prompts package for indication extraction."""

import os
from typing import Optional

from langfuse import Langfuse


def get_system_prompt(
    langfuse_client: Optional[Langfuse] = None,
    prompt_name: str = "MEDICAL_INDICATION_EXTRACTION_SYSTEM_PROMPT",
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load the system prompt from Langfuse or fallback to local file.

    Args:
        langfuse_client: Optional Langfuse client instance. If not provided, will create one using env vars.
        prompt_name: Name of the prompt in Langfuse (default: "MEDICAL_INDICATION_EXTRACTION_SYSTEM_PROMPT")
        fallback_to_file: If True, fallback to reading from local file (prompt_name.md) if Langfuse fetch fails

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
        print(f"ℹ Falling back to local {prompt_filename} file...")
        try:
            prompt_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_file = os.path.join(prompt_dir, prompt_filename)
            
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print("✓ Successfully loaded prompt from local file")
            return content.strip(), "local"
        except Exception as file_error:
            raise Exception(
                f"Failed to fetch prompt from Langfuse and local file: "
                f"Langfuse error: {e}, File error: {file_error}"
            )
