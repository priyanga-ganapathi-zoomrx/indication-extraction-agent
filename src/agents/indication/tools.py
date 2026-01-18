"""Tools for indication extraction rules retrieval.

Uses configuration for paths and caching for performance.
"""

import csv
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from src.agents.indication.config import config


@lru_cache(maxsize=1)
def _load_rules(rules_path: str) -> list[dict]:
    """Load and cache rules from CSV.
    
    Uses utf-8-sig encoding to handle BOM automatically.
    """
    path = Path(rules_path)
    if not path.exists():
        return []
    
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [{k.strip(): v.strip() for k, v in row.items()} for row in reader]


@tool
def get_indication_rules(category: str, subcategories: list[str]) -> str:
    """Retrieve clinical rules for indication extraction.

    Args:
        category: Main category (e.g., "Common Check points", "Gene type")
        subcategories: List of subcategories to filter by

    Returns:
        Formatted string with matching rules
    """
    rules = _load_rules(str(config.RULES_PATH))
    
    if not rules:
        return f"Error: Rules file not found at {config.RULES_PATH}"
    
    matches = [
        r for r in rules
        if r.get("Category", "") == category.strip()
        and any(s.strip() == r.get("Sub Category", "") for s in subcategories)
    ]
    
    if not matches:
        return f"No rules found for category '{category}' with subcategories {subcategories}"
    
    lines = [f"Found {len(matches)} rule(s):\n"]
    for i, r in enumerate(matches, 1):
        lines.append(f"Rule {i} (ID: {r.get('ID', 'N/A')}):")
        lines.append(f"  Keyword: {r.get('Keyword', '')}")
        action_key = "Do/Don't"
        lines.append(f"  Action: {r.get(action_key, '')}")
        lines.append(f"  Generated Rule: {r.get('Generated_Rule', '')}")
        lines.append("")
    
    return "\n".join(lines)


def get_tools() -> list:
    """Get all indication extraction tools."""
    return [get_indication_rules]
