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
    
    # With idle shutdown (env var)
    IDLE_SHUTDOWN_MINUTES=5 python -m src.temporal.workers.checkpoint_worker
"""

import asyncio
import logging
import os

from src.temporal.config import TaskQueues, WorkerSettings
from src.temporal.activities.checkpoint import (
    load_workflow_status,
    save_workflow_status,
    load_step_output,
    save_step_output,
)
from src.temporal.workers.base import run_worker

logger = logging.getLogger(__name__)


async def run_checkpoint_worker(idle_shutdown_minutes: float | None = None) -> None:
    """Run the checkpoint/storage worker.
    
    Args:
        idle_shutdown_minutes: Auto-shutdown after N minutes of inactivity
    """
    settings = WorkerSettings.CHECKPOINT
    
    logger.info("Starting Checkpoint Worker")
    
    await run_worker(
        task_queue=TaskQueues.CHECKPOINT,
        workflows=None,
        activities=[
            load_workflow_status,
            save_workflow_status,
            load_step_output,
            save_step_output,
        ],
        max_concurrent_activities=settings.get("max_concurrent_activities", 50),
        idle_shutdown_minutes=idle_shutdown_minutes,
    )


def main():
    """Entry point."""
    idle_shutdown = os.getenv("IDLE_SHUTDOWN_MINUTES")
    idle_minutes = float(idle_shutdown) if idle_shutdown else None
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run_checkpoint_worker(idle_shutdown_minutes=idle_minutes))


if __name__ == "__main__":
    main()
