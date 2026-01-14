"""Prompts for drug extraction and validation."""

from pathlib import Path
from typing import Optional

from langfuse import Langfuse

from src.agents.core import load_prompt


# Prompt names
EXTRACTION_PROMPT_NAME = "DRUG_EXTRACTION_SYSTEM_PROMPT"
VALIDATION_PROMPT_NAME = "DRUG_VALIDATION_SYSTEM_PROMPT"

# Prompts directory
PROMPTS_DIR = Path(__file__).parent / "prompts"


def get_extraction_prompt(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load drug extraction prompt."""
    return load_prompt(EXTRACTION_PROMPT_NAME, PROMPTS_DIR, langfuse_client, fallback_to_file)


def get_validation_prompt_parts(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str, str]:
    """Load and parse validation prompt into instructions and rules.
    
    Returns:
        Tuple of (instructions, rules, version)
    """
    full_prompt, version = load_prompt(VALIDATION_PROMPT_NAME, PROMPTS_DIR, langfuse_client, fallback_to_file)
    
    # Parse MESSAGE_1 (instructions) and MESSAGE_2 (rules)
    instructions = ""
    rules = ""
    
    # Extract MESSAGE_1: VALIDATION_INSTRUCTIONS
    msg1_start = "<!-- MESSAGE_1_START: VALIDATION_INSTRUCTIONS -->"
    msg1_end = "<!-- MESSAGE_1_END: VALIDATION_INSTRUCTIONS -->"
    if msg1_start in full_prompt and msg1_end in full_prompt:
        start_idx = full_prompt.index(msg1_start) + len(msg1_start)
        end_idx = full_prompt.index(msg1_end)
        instructions = full_prompt[start_idx:end_idx].strip()
    
    # Extract MESSAGE_2: EXTRACTION_RULES
    msg2_start = "<!-- MESSAGE_2_START: EXTRACTION_RULES -->"
    msg2_end = "<!-- MESSAGE_2_END: EXTRACTION_RULES -->"
    if msg2_start in full_prompt and msg2_end in full_prompt:
        start_idx = full_prompt.index(msg2_start) + len(msg2_start)
        end_idx = full_prompt.index(msg2_end)
        rules = full_prompt[start_idx:end_idx].strip()
    
    return instructions, rules, version
