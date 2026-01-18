"""Indication processing scripts and batch processors."""

from src.scripts.indication.extraction_processor import (
    load_abstracts,
    process_single as extraction_process_single,
    save_extraction_result,
    save_batch_status as extraction_save_batch_status,
    save_results_csv as extraction_save_results_csv,
    run_extraction_batch,
    ProcessResult as ExtractionResult,
    StatusEntry,
)
from src.scripts.indication.validation_processor import (
    load_abstracts_for_validation,
    process_single as validation_process_single,
    save_validation_result,
    save_batch_status as validation_save_batch_status,
    save_results_csv as validation_save_results_csv,
    run_validation_batch,
    ProcessResult as ValidationResult,
    ValidationInput,
)

__all__ = [
    # Batch processor (extraction)
    "load_abstracts",
    "extraction_process_single",
    "save_extraction_result",
    "extraction_save_batch_status",
    "extraction_save_results_csv",
    "run_extraction_batch",
    "ExtractionResult",
    "StatusEntry",
    # Validation processor
    "load_abstracts_for_validation",
    "validation_process_single",
    "save_validation_result",
    "validation_save_batch_status",
    "validation_save_results_csv",
    "run_validation_batch",
    "ValidationResult",
    "ValidationInput",
]
