"""Temporal workflow orchestration for abstract extraction.

This module provides:
- Activities: Thin wrappers around existing agents
- Workflows: Orchestration logic for extraction pipeline
- Workers: Separate workers per task queue
- Client: Batch workflow execution utilities
- Config: Task queues, timeouts, retry policies

For workflow status, results, retries, and cancellation, use Temporal UI or CLI.
"""

from src.temporal.workflows import (
    AbstractExtractionWorkflow,
    AbstractExtractionInput,
    AbstractExtractionOutput,
)

from src.temporal.client import (
    generate_workflow_id,
    load_batch_items,
    start_batch_extraction,
    BatchItem,
    BatchResult,
)

__all__ = [
    # Workflows
    "AbstractExtractionWorkflow",
    "AbstractExtractionInput",
    "AbstractExtractionOutput",
    # Client utilities
    "generate_workflow_id",
    "load_batch_items",
    "start_batch_extraction",
    "BatchItem",
    "BatchResult",
]
