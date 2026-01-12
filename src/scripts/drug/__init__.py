# src/scripts/drug/__init__.py
"""Drug processing scripts package.

This package contains batch processors and CLI scripts for drug extraction:
- batch_processor: Full batch processing using DrugExtractionAgent
- extraction_processor: Step 1 - Drug extraction only
- validation_processor: Step 2 - Drug validation only
- verification_processor: Step 3 - Drug verification via Tavily
"""

from src.scripts.drug.batch_processor import (
    load_abstracts_from_csv,
    process_abstracts_batch,
    extract_drugs_from_response,
    process_single_abstract,
    main as batch_main,
)
from src.scripts.drug.extraction_processor import (
    DrugExtractionProcessor,
    load_abstracts_from_csv as load_abstracts_for_extraction,
    parse_extraction_response,
    process_single_abstract as process_extraction_abstract,
    main as extraction_main,
)
from src.scripts.drug.validation_processor import (
    DrugValidationProcessor,
    load_extraction_results,
    parse_validation_response,
    process_single_row as process_validation_row,
    main as validation_main,
)
from src.scripts.drug.verification_processor import (
    DrugVerificationProcessor,
    load_extraction_results as load_results_for_verification,
    process_single_row as process_verification_row,
    main as verification_main,
)

__all__ = [
    # Batch processor
    "load_abstracts_from_csv",
    "process_abstracts_batch",
    "extract_drugs_from_response",
    "process_single_abstract",
    "batch_main",
    # Extraction processor
    "DrugExtractionProcessor",
    "load_abstracts_for_extraction",
    "parse_extraction_response",
    "process_extraction_abstract",
    "extraction_main",
    # Validation processor
    "DrugValidationProcessor",
    "load_extraction_results",
    "parse_validation_response",
    "process_validation_row",
    "validation_main",
    # Verification processor
    "DrugVerificationProcessor",
    "load_results_for_verification",
    "process_verification_row",
    "verification_main",
]
