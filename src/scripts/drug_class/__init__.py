# src/scripts/drug_class/__init__.py
"""Drug class processing scripts package.

This package contains batch processors for drug class extraction:
- consolidation_processor: Consolidates multiple extraction results
- extraction_title_processor: Extracts drug classes from titles
- grounded_search_processor: Grounded search-based extraction
- react_batch_processor: ReAct pattern-based batch processing
- regimen_batch_processor: Identifies regimens and extracts component drugs
- selection_processor: Selects best drug classes from candidates
- validation_processor: Validates extraction results
"""

from src.scripts.drug_class.consolidation_processor import main as consolidation_main
from src.scripts.drug_class.extraction_title_processor import main as extraction_title_main
from src.scripts.drug_class.grounded_search_processor import main as grounded_search_main
from src.scripts.drug_class.react_batch_processor import main as react_batch_main
from src.scripts.drug_class.regimen_batch_processor import main as regimen_batch_main
from src.scripts.drug_class.selection_processor import main as selection_main
from src.scripts.drug_class.validation_processor import main as validation_main

__all__ = [
    "consolidation_main",
    "extraction_title_main",
    "grounded_search_main",
    "react_batch_main",
    "regimen_batch_main",
    "selection_main",
    "validation_main",
]
