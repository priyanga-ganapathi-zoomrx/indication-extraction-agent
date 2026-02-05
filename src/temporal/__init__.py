"""Temporal workflow orchestration for abstract extraction.

This module provides:
- Activities: Thin wrappers around existing agents
- Workflows: Orchestration logic for extraction pipeline
- Workers: Separate workers per task queue
- Client: Utilities for starting and managing workflows
- Config: Task queues, timeouts, retry policies
"""

from src.temporal.config import (
    TASK_QUEUES,
    TIMEOUTS,
    RETRY_POLICIES,
    WORKER_SETTINGS,
)

from src.temporal.workflows import (
    AbstractExtractionWorkflow,
    AbstractExtractionInput,
    AbstractExtractionOutput,
)

from src.temporal.client import (
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
    # Config
    "TASK_QUEUES",
    "TIMEOUTS",
    "RETRY_POLICIES",
    "WORKER_SETTINGS",
    # Workflows
    "AbstractExtractionWorkflow",
    "AbstractExtractionInput",
    "AbstractExtractionOutput",
    # Client utilities
    "get_client",
    "start_extraction",
    "execute_extraction",
    "start_batch_extraction",
    "get_extraction_status",
    "get_extraction_result",
    "query_extraction_step",
    "cancel_extraction",
]
