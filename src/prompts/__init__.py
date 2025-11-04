"""Prompts package for indication extraction."""

import os


def get_system_prompt() -> str:
    """Load the system prompt from the markdown file.

    Returns:
        str: The system prompt content from system_prompt.md
    """
    # Get the directory of this file
    prompt_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(prompt_dir, "system_prompt.md")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    return content.strip()
