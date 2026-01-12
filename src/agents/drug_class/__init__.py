# src/agents/drug_class/__init__.py
"""Drug class extraction agent package.

This package contains all drug class-related agents:
- DrugClassConsolidationOnlyAgent: Consolidates multiple extraction results
- DrugClassExtractionTitleAgent: Extracts drug classes from titles
- DrugClassGroundedSearchAgent: Grounded search-based extraction
- DrugClassReActAgent: ReAct pattern-based extraction
- DrugClassSelectionAgent: Selects best drug classes from candidates
- DrugClassValidationAgent: Validates extraction results
"""

from src.agents.drug_class.consolidation_agent import DrugClassConsolidationOnlyAgent
from src.agents.drug_class.extraction_title_agent import DrugClassExtractionTitleAgent
from src.agents.drug_class.grounded_search_agent import DrugClassGroundedSearchAgent
from src.agents.drug_class.react_agent import DrugClassReActAgent
from src.agents.drug_class.regimen_identification_agent import RegimenIdentificationAgent
from src.agents.drug_class.selection_agent import DrugClassSelectionAgent
from src.agents.drug_class.validation_agent import DrugClassValidationAgent
from src.agents.drug_class.prompts import (
    get_system_prompt,
    get_extraction_title_prompt,
    get_extraction_rules_prompt,
    get_validation_prompt,
    get_selection_prompt,
    get_grounded_search_prompt,
    get_consolidation_prompt,
    get_regimen_identification_prompt,
    EXTRACTION_TITLE_PROMPT_NAME,
    EXTRACTION_RULES_PROMPT_NAME,
    VALIDATION_PROMPT_NAME,
    SELECTION_PROMPT_NAME,
    GROUNDED_SEARCH_PROMPT_NAME,
    CONSOLIDATION_PROMPT_NAME,
    REGIMEN_IDENTIFICATION_PROMPT_NAME,
)

__all__ = [
    # Agents
    "DrugClassConsolidationOnlyAgent",
    "DrugClassExtractionTitleAgent",
    "DrugClassGroundedSearchAgent",
    "DrugClassReActAgent",
    "DrugClassSelectionAgent",
    "DrugClassValidationAgent",
    "RegimenIdentificationAgent",
    # Prompts
    "get_system_prompt",
    "get_extraction_title_prompt",
    "get_extraction_rules_prompt",
    "get_validation_prompt",
    "get_selection_prompt",
    "get_grounded_search_prompt",
    "get_consolidation_prompt",
    "get_regimen_identification_prompt",
    "EXTRACTION_TITLE_PROMPT_NAME",
    "EXTRACTION_RULES_PROMPT_NAME",
    "VALIDATION_PROMPT_NAME",
    "SELECTION_PROMPT_NAME",
    "GROUNDED_SEARCH_PROMPT_NAME",
    "CONSOLIDATION_PROMPT_NAME",
    "REGIMEN_IDENTIFICATION_PROMPT_NAME",
]
