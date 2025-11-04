"""Indication extraction tools for the agent."""

from src.rule_tool import get_indication_tools


def get_tools():
    """Returns a list of indication extraction tools.

    Returns:
        list: List of indication extraction tool functions
    """
    return get_indication_tools()

