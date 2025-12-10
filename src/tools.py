"""Indication and drug class extraction tools for the agent."""

from src.rule_tool import get_indication_tools, get_drug_class_tools


def get_tools():
    """Returns a list of indication extraction tools.

    Returns:
        list: List of indication extraction tool functions
    """
    return get_indication_tools()


def get_drug_class_extraction_tools():
    """Returns a list of drug class extraction tools.

    Returns:
        list: List of drug class extraction tool functions
    """
    return get_drug_class_tools()

