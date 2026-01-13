"""Indication processing scripts and batch processors."""

from src.scripts.indication.batch_processor import (
    load_abstracts,
    process_single as extraction_process_single,
    save_results as extraction_save_results,
    ProcessResult as ExtractionResult,
)
from src.scripts.indication.validation_processor import (
    load_extractions,
    process_single as validation_process_single,
    save_results as validation_save_results,
    ProcessResult as ValidationResult,
)

__all__ = [
    # Batch processor (extraction)
    "load_abstracts",
    "extraction_process_single",
    "extraction_save_results",
    "ExtractionResult",
    # Validation processor
    "load_extractions",
    "validation_process_single",
    "validation_save_results",
    "ValidationResult",
]
