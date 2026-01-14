# src/scripts/drug/__init__.py
"""Drug processing scripts package.

This package contains batch processors for drug extraction and validation:
- extraction_processor: Drug extraction using extract_drugs()
- validation_processor: Drug validation using validate_drugs()
"""

from src.scripts.drug.extraction_processor import (
    load_abstracts,
    process_single as process_extraction,
    main as extraction_main,
)
from src.scripts.drug.validation_processor import (
    load_extractions,
    process_single as process_validation,
    main as validation_main,
)

__all__ = [
    # Extraction processor
    "load_abstracts",
    "process_extraction",
    "extraction_main",
    # Validation processor
    "load_extractions",
    "process_validation",
    "validation_main",
]
