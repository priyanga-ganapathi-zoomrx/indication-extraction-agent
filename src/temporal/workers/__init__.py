"""Temporal workers for abstract extraction.

This module provides workers for each task queue:
- WorkflowWorker: Orchestrates extraction workflows (lightweight, no activities)
- CheckpointWorker: Storage operations for status/checkpoints (shared by all pipelines)
- DrugWorker: Executes drug extraction/validation activities
- DrugClassWorker: Executes drug class pipeline activities
- IndicationExtractionWorker: Executes indication extraction activities
- IndicationValidationWorker: Executes indication validation (slow, Sonnet 4.5)

Best Practice: Run separate workers per task queue to:
- Enable independent scaling
- Isolate slow activities from fast ones
- Apply different concurrency settings per workload

Worker Architecture:
                    ┌─────────────────────┐
                    │   Workflow Worker   │
                    │   (orchestration)   │
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Checkpoint Worker  │
                    │  (storage, shared)  │
                    └─────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────┐                    ┌───────────────────┐
│   Drug Workers    │                    │ Indication Workers│
│ - Drug            │                    │ - Extraction      │
│ - Drug Class      │                    │ - Validation      │
└───────────────────┘                    └───────────────────┘
"""

from src.temporal.workers.base import create_temporal_client, run_worker
from src.temporal.workers.workflow_worker import run_workflow_worker
from src.temporal.workers.checkpoint_worker import run_checkpoint_worker
from src.temporal.workers.drug_worker import run_drug_worker
from src.temporal.workers.drug_class_worker import run_drug_class_worker
from src.temporal.workers.indication_extraction_worker import run_indication_extraction_worker
from src.temporal.workers.indication_validation_worker import run_indication_validation_worker

__all__ = [
    # Utilities
    "create_temporal_client",
    "run_worker",
    # Worker runners
    "run_workflow_worker",
    "run_checkpoint_worker",
    "run_drug_worker",
    "run_drug_class_worker",
    "run_indication_extraction_worker",
    "run_indication_validation_worker",
]
