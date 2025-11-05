"""Tool for retrieving indication extraction rules."""

import csv
from typing import List, Dict, Any
from langchain_core.tools import tool


@tool
def get_indication_rules(category: str, subcategories: List[str]) -> str:
    """Retrieves clinical rules for indication extraction based on category and subcategories.

    This tool searches the indication extraction rules database to find relevant rules
    that apply to specific categories and subcategories of medical indication processing.

    Args:
        category: The main category of rules to retrieve (e.g., "Common Check points", "Patient Sub-Group")
        subcategories: List of subcategories within the category (e.g., ["Casing", "General"])

    Returns:
        str: Formatted string containing all matching rules with their details

    Example:
        getRule("Common Check points", ["Casing", "General"])
    """
    try:
        rules_found = []
        rules_file = "data/indication_extraction_rules.csv"

        with open(rules_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if this rule matches the requested category and any of the subcategories
                if (row['Category'].strip() == category.strip() and
                    any(subcat.strip() == row['Sub Category'].strip() for subcat in subcategories)):

                    # Handle BOM in ID column
                    id_key = 'ID' if 'ID' in row else '\ufeffID'
                    rule_info = {
                        'id': row[id_key],
                        'category': row['Category'].strip(),
                        'subcategory': row['Sub Category'].strip(),
                        'keyword': row['Keyword'].strip(),
                        'do_dont': row['Do/Don\'t'].strip(),
                        'action': row['Action'].strip(),
                        'example_title': row['Example_Title'].strip(),
                        'output_indication': row['Output_Indication'].strip(),
                        'generated_rule': row['Generated_Rule'].strip(),
                        'rule_status': row['Rule_Generation_Status'].strip()
                    }
                    rules_found.append(rule_info)

        if not rules_found:
            return f"No rules found for category '{category}' with subcategories {subcategories}"

        # Format the results
        result = f"Found {len(rules_found)} rule(s) for category '{category}' and subcategories {subcategories}:\n\n"

        for i, rule in enumerate(rules_found, 1):
            result += f"Rule {i} (ID: {rule['id']}):\n"
            result += f"  Category: {rule['category']}\n"
            result += f"  Subcategory: {rule['subcategory']}\n"
            result += f"  Keyword: {rule['keyword']}\n"
            result += f"  Action: {rule['do_dont']}\n"
            result += f"  Example Title: {rule['example_title']}\n"
            result += f"  Expected Output: {rule['output_indication']}\n"
            result += f"  Generated Rule: {rule['generated_rule']}\n"
            result += f"  Status: {rule['rule_status']}\n\n"

        return result

    except FileNotFoundError:
        return f"Error: Rules file '{rules_file}' not found."
    except Exception as e:
        return f"Error retrieving rules: {str(e)}"


def get_indication_tools():
    """Returns a list of indication extraction tools.

    Returns:
        list: List of indication extraction tool functions
    """
    return [get_indication_rules]
