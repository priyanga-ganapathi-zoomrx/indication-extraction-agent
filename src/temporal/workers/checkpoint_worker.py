"""Checkpoint Worker - Handles storage/persistence activities.

This worker:
- Polls the CHECKPOINT task queue
- Executes load/save activities for status and step outputs
- Fast I/O operations to GCS or local filesystem

Activities:
- load_workflow_status: Load workflow status from storage
- save_workflow_status: Save workflow status to storage
- load_step_output: Load step checkpoint from storage
- save_step_output: Save step checkpoint to storage

Note: This worker is shared by all pipelines (drug, indication, etc.)
since checkpoint activities are cross-cutting storage operations.

Usage:
    python -m src.temporal.workers.checkpoint_worker
"""

import asyncio
import logging

from src.temporal.config import TaskQueues, WorkerSettings
from src.temporal.activities.checkpoint import (
    load_workflow_status,
    save_workflow_status,
    load_step_output,
    save_step_output,
)
from src.temporal.workers.base import run_worker

logger = logging.getLogger(__name__)


async def run_checkpoint_worker() -> None:
    """Run the checkpoint/storage worker.
    
    This worker handles all checkpoint/persistence activities.
    These are fast I/O operations (typically <1s) shared by all
    entity pipelines.
    
    Configuration from WorkerSettings.CHECKPOINT:
    - max_concurrent_activities: 50 (I/O bound, can be high)
    """
    settings = WorkerSettings.CHECKPOINT
    
    logger.info("Starting Checkpoint Worker")
    
    await run_worker(
        task_queue=TaskQueues.CHECKPOINT,
        workflows=None,  # No workflows - activities only
        activities=[
            load_workflow_status,
            save_workflow_status,
            load_step_output,
            save_step_output,
        ],
        max_concurrent_activities=settings.get("max_concurrent_activities", 50),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run_checkpoint_worker())
