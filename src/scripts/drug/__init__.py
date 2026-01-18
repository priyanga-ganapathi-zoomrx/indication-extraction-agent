# src/scripts/drug/__init__.py
"""Drug processing scripts package.

This package contains batch processors for drug extraction and validation:
- extraction_processor: Drug extraction using extract_drugs()
- validation_processor: Drug validation using validate_drugs()

Features:
- Parallel processing with configurable workers
- Per-abstract status tracking (status.json)
- Batch-level status summary (batch_status.json)
- Retry logic for failed extractions/validations
- Real-time progress monitoring with tqdm
- Duration tracking with accumulation across retries
"""

from src.scripts.drug.extraction_processor import (
    load_abstracts,
    process_single as process_extraction,
    save_extraction_result,
    save_results_csv as save_extraction_csv,
    save_batch_status as save_extraction_batch_status,
    run_extraction_batch,
    get_abstract_status as get_extraction_status,
    should_process_extraction,
    StatusEntry,
    ProcessResult as ExtractionProcessResult,
    main as extraction_main,
)
from src.scripts.drug.validation_processor import (
    load_abstracts_for_validation,
    process_single as process_validation,
    save_validation_result,
    save_results_csv as save_validation_csv,
    save_batch_status as save_validation_batch_status,
    run_validation_batch,
    get_abstract_status as get_validation_status,
    should_process_validation,
    ValidationInput,
    ProcessResult as ValidationProcessResult,
    main as validation_main,
)

__all__ = [
    # Extraction processor
    "load_abstracts",
    "process_extraction",
    "save_extraction_result",
    "save_extraction_csv",
    "save_extraction_batch_status",
    "run_extraction_batch",
    "get_extraction_status",
    "should_process_extraction",
    "StatusEntry",
    "ExtractionProcessResult",
    "extraction_main",
    # Validation processor
    "load_abstracts_for_validation",
    "process_validation",
    "save_validation_result",
    "save_validation_csv",
    "save_validation_batch_status",
    "run_validation_batch",
    "get_validation_status",
    "should_process_validation",
    "ValidationInput",
    "ValidationProcessResult",
    "validation_main",
]
