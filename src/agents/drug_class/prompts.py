"""Prompts module for drug class extraction, validation, and related agents."""

import re
from pathlib import Path
from typing import Optional

from langfuse import Langfuse

from src.agents.core.langfuse_config import langfuse as langfuse_singleton


# Available prompt names
EXTRACTION_TITLE_PROMPT_NAME = "DRUG_CLASS_EXTRACTION_FROM_TITLE"
EXTRACTION_RULES_PROMPT_NAME = "DRUG_CLASS_EXTRACTION_FROM_SEARCH_REACT_PATTERN"
VALIDATION_PROMPT_NAME = "DRUG_CLASS_VALIDATION_SYSTEM_PROMPT"
SELECTION_PROMPT_NAME = "DRUG_CLASS_SELECTION_SYSTEM_PROMPT"
GROUNDED_SEARCH_PROMPT_NAME = "DRUG_CLASS_GROUNDED_SEARCH_PROMPT"
CONSOLIDATION_PROMPT_NAME = "DRUG_CLASS_CONSOLIDATION_PROMPT"
REGIMEN_IDENTIFICATION_PROMPT_NAME = "REGIMEN_IDENTIFICATION_PROMPT"

# Default prompts directory (relative to this file)
PROMPTS_DIR = Path(__file__).parent / "prompts"

# Prompt cache to avoid repeated fetching
_prompt_cache: dict[str, tuple[str, str]] = {}


# =============================================================================
# SECTION EXTRACTION HELPER
# =============================================================================

def extract_section(content: str, section_name: str) -> str:
    """Extract content between MESSAGE markers in prompt file.
    
    Args:
        content: Full prompt file content
        section_name: Section name to extract (e.g., "SYSTEM_PROMPT")
        
    Returns:
        Extracted section content, or empty string if not found
    """
    pattern = rf'<!-- MESSAGE_\d+_START: {section_name} -->\s*(.*?)\s*<!-- MESSAGE_\d+_END: {section_name} -->'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        section = match.group(1).strip()
        # Remove the section header if present
        section = re.sub(rf'^##\s*{section_name}\s*\n+', '', section)
        return section
    return ""


# =============================================================================
# PROMPT LOADING FUNCTIONS
# =============================================================================

def get_system_prompt(
    langfuse_client: Optional[Langfuse] = None,
    prompt_name: str = EXTRACTION_TITLE_PROMPT_NAME,
    fallback_to_file: bool = True,
    prompt_dir: Optional[Path] = None,
) -> tuple[str, str]:
    """Load the system prompt from Langfuse or fallback to local file.

    Args:
        langfuse_client: Optional Langfuse client instance. If not provided, uses the singleton.
        prompt_name: Name of the prompt in Langfuse (default: EXTRACTION_TITLE_PROMPT_NAME)
        fallback_to_file: If True, fallback to reading from local file (prompt_name.md) if Langfuse fetch fails
        prompt_dir: Optional directory to look for prompt files (default: prompts/ in same folder)

    Returns:
        tuple[str, str]: A tuple of (prompt_content, prompt_version)

    Raises:
        Exception: If Langfuse fetch fails and fallback_to_file is False
    """
    # Check cache first
    if prompt_name in _prompt_cache:
        return _prompt_cache[prompt_name]
    
    prompts_directory = prompt_dir or PROMPTS_DIR
    
    # Use provided client, singleton, or skip Langfuse if not enabled
    client = langfuse_client or langfuse_singleton
    
    # If no Langfuse client available, go straight to file
    if client is None:
        result = _load_prompt_from_file(prompt_name, prompts_directory)
        _prompt_cache[prompt_name] = result
        return result
    
    # Try to fetch from Langfuse
    try:
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
        result = (content.strip(), version)
        _prompt_cache[prompt_name] = result
        return result
        
    except Exception as e:
        print(f"✗ Error fetching prompt from Langfuse: {e}")
        
        if not fallback_to_file:
            raise
        
        result = _load_prompt_from_file(prompt_name, prompts_directory)
        _prompt_cache[prompt_name] = result
        return result


def _load_prompt_from_file(prompt_name: str, prompts_directory: Path) -> tuple[str, str]:
    """Load prompt from local file.
    
    Args:
        prompt_name: Name of the prompt (used as filename without extension)
        prompts_directory: Directory containing prompt files
        
    Returns:
        tuple[str, str]: (prompt_content, "local")
    """
    prompt_filename = f"{prompt_name}.md"
    print(f"ℹ Loading prompt from local {prompt_filename} file...")
        
    prompt_file = prompts_directory / prompt_filename
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("✓ Successfully loaded prompt from local file")
    return content.strip(), "local"


def get_extraction_title_prompt(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load the drug class extraction from title prompt.

    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails

    Returns:
        tuple[str, str]: A tuple of (prompt_content, prompt_version)
    """
    return get_system_prompt(
        langfuse_client=langfuse_client,
        prompt_name=EXTRACTION_TITLE_PROMPT_NAME,
        fallback_to_file=fallback_to_file,
    )


def get_extraction_rules_prompt(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load the drug class extraction rules prompt.

    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails

    Returns:
        tuple[str, str]: A tuple of (prompt_content, prompt_version)
    """
    return get_system_prompt(
        langfuse_client=langfuse_client,
        prompt_name=EXTRACTION_RULES_PROMPT_NAME,
        fallback_to_file=fallback_to_file,
    )


def get_validation_prompt(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load the drug class validation prompt.

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


def get_selection_prompt(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load the drug class selection prompt.

    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails

    Returns:
        tuple[str, str]: A tuple of (prompt_content, prompt_version)
    """
    return get_system_prompt(
        langfuse_client=langfuse_client,
        prompt_name=SELECTION_PROMPT_NAME,
        fallback_to_file=fallback_to_file,
    )


def get_grounded_search_prompt(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load the drug class grounded search prompt.

    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails

    Returns:
        tuple[str, str]: A tuple of (prompt_content, prompt_version)
    """
    return get_system_prompt(
        langfuse_client=langfuse_client,
        prompt_name=GROUNDED_SEARCH_PROMPT_NAME,
        fallback_to_file=fallback_to_file,
    )


def get_consolidation_prompt(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load the drug class consolidation prompt.

    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails

    Returns:
        tuple[str, str]: A tuple of (prompt_content, prompt_version)
    """
    return get_system_prompt(
        langfuse_client=langfuse_client,
        prompt_name=CONSOLIDATION_PROMPT_NAME,
        fallback_to_file=fallback_to_file,
    )


def get_regimen_identification_prompt(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str]:
    """Load the regimen identification prompt.

    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails

    Returns:
        tuple[str, str]: A tuple of (prompt_content, prompt_version)
    """
    return get_system_prompt(
        langfuse_client=langfuse_client,
        prompt_name=REGIMEN_IDENTIFICATION_PROMPT_NAME,
        fallback_to_file=fallback_to_file,
    )


# =============================================================================
# PARSED PROMPT FUNCTIONS (return sections)
# =============================================================================

def get_extraction_rules_prompt_parts(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str, str]:
    """Load and parse extraction rules prompt into system prompt and rules.
    
    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails
    
    Returns:
        Tuple of (system_prompt, rules_message, version)
    """
    full_prompt, version = get_extraction_rules_prompt(langfuse_client, fallback_to_file)
    
    system_prompt = extract_section(full_prompt, "SYSTEM_PROMPT")
    rules_message = extract_section(full_prompt, "RULES_MESSAGE")
    
    return system_prompt, rules_message, version


def get_grounded_search_prompt_parts(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str, str]:
    """Load and parse grounded search prompt into system prompt and rules.
    
    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails
    
    Returns:
        Tuple of (system_prompt, rules_message, version)
    """
    full_prompt, version = get_grounded_search_prompt(langfuse_client, fallback_to_file)
    
    system_prompt = extract_section(full_prompt, "SYSTEM_PROMPT")
    rules_message = extract_section(full_prompt, "RULES_MESSAGE")
    
    # Fallback to using the whole prompt as system prompt if no sections found
    if not system_prompt:
        system_prompt = full_prompt
        rules_message = ""
    
    return system_prompt, rules_message, version


def get_selection_prompt_parts(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str, str]:
    """Load selection prompt and extraction rules for selection step.
    
    The selection step uses the selection prompt as system message and
    the extraction rules (RULES_MESSAGE section) to guide selection decisions.
    
    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails
    
    Returns:
        Tuple of (selection_prompt, rules_message, version)
    """
    selection_prompt, selection_version = get_selection_prompt(langfuse_client, fallback_to_file)
    rules_content, _ = get_extraction_rules_prompt(langfuse_client, fallback_to_file)
    
    rules_message = extract_section(rules_content, "RULES_MESSAGE")
    
    return selection_prompt, rules_message, selection_version


def get_explicit_extraction_prompt_parts(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str, str, str]:
    """Load and parse explicit extraction (from title) prompt into sections.
    
    The explicit extraction step extracts drug classes directly from the
    abstract title. It uses the title extraction prompt and extraction rules.
    
    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails
    
    Returns:
        Tuple of (system_prompt, input_template, rules_message, version)
    """
    title_prompt, title_version = get_extraction_title_prompt(langfuse_client, fallback_to_file)
    rules_content, _ = get_extraction_rules_prompt(langfuse_client, fallback_to_file)
    
    # Extract sections from title prompt
    system_prompt = extract_section(title_prompt, "SYSTEM_PROMPT")
    input_template = extract_section(title_prompt, "INPUT_TEMPLATE")
    
    # Extract rules section
    rules_message = extract_section(rules_content, "RULES_MESSAGE")
    
    # Fallback to whole prompt if no sections found
    if not system_prompt:
        system_prompt = title_prompt
    
    return system_prompt, input_template, rules_message, title_version


def get_consolidation_prompt_parts(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str, str, str]:
    """Load and parse consolidation prompt into sections.
    
    The consolidation step compares explicit drug classes (from Step 4) with
    drug-specific selections (from Step 3) and removes duplicates/parents.
    
    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails
    
    Returns:
        Tuple of (system_prompt, input_template, rules_message, version)
    """
    consolidation_prompt, version = get_consolidation_prompt(langfuse_client, fallback_to_file)
    rules_content, _ = get_extraction_rules_prompt(langfuse_client, fallback_to_file)
    
    # Extract sections from consolidation prompt
    system_prompt = extract_section(consolidation_prompt, "SYSTEM_PROMPT")
    input_template = extract_section(consolidation_prompt, "INPUT_TEMPLATE")
    
    # Extract rules section
    rules_message = extract_section(rules_content, "RULES_MESSAGE")
    
    # Fallback to whole prompt if no sections found
    if not system_prompt:
        system_prompt = consolidation_prompt
    
    return system_prompt, input_template, rules_message, version


def get_validation_prompt_parts(
    langfuse_client: Optional[Langfuse] = None,
    fallback_to_file: bool = True,
) -> tuple[str, str, str]:
    """Load validation prompt and extraction rules for reference.
    
    The validation step validates drug class extractions against the 
    extraction rules. It needs the validation prompt and the extraction
    rules (as reference for the validator).
    
    Args:
        langfuse_client: Optional Langfuse client instance
        fallback_to_file: If True, fallback to local file if Langfuse fails
    
    Returns:
        Tuple of (validation_prompt, extraction_rules, version)
    """
    validation_prompt, version = get_validation_prompt(langfuse_client, fallback_to_file)
    extraction_rules, _ = get_extraction_rules_prompt(langfuse_client, fallback_to_file)
    
    return validation_prompt, extraction_rules, version

