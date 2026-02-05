"""Temporal client utilities for starting and managing workflows.

This module provides:
- Client creation and connection management
- Workflow execution helpers (single and batch)
- Workflow status and result retrieval
- Query utilities for workflow state inspection

Usage:
    from src.temporal.client import (
        get_client,
        start_extraction,
        execute_extraction,
        start_batch_extraction,
        get_extraction_status,
        get_extraction_result,
    )
"""

from src.temporal.client.extraction import (
    get_client,
    start_extraction,
    execute_extraction,
    start_batch_extraction,
    get_extraction_status,
    get_extraction_result,
    query_extraction_step,
    cancel_extraction,
)

__all__ = [
    # Client
    "get_client",
    # Single workflow
    "start_extraction",
    "execute_extraction",
    # Batch workflow
    "start_batch_extraction",
    # Status and results
    "get_extraction_status",
    "get_extraction_result",
    "query_extraction_step",
    "cancel_extraction",
]
