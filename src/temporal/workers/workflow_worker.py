"""Workflow Worker - Handles workflow orchestration tasks.

This worker:
- Polls the WORKFLOWS task queue
- Executes the AbstractExtractionWorkflow (single flat workflow)
- Lightweight orchestration only (no activities)

Best Practice: Workflow workers are separate from activity workers
to allow independent scaling and avoid resource contention.

Note: Checkpoint activities run on a dedicated CHECKPOINT queue,
not on this workflow queue.

Note: Idle shutdown is not useful for workflow-only workers since
they don't run activities (idle tracking is activity-based).

Usage:
    python -m src.temporal.workers.workflow_worker
"""

import asyncio
import logging

from src.temporal.config import TaskQueues, WorkerSettings
from src.temporal.workflows import AbstractExtractionWorkflow
from src.temporal.workers.base import run_worker

logger = logging.getLogger(__name__)


async def run_workflow_worker() -> None:
    """Run the workflow worker.

    This worker handles workflow orchestration for:
    - AbstractExtractionWorkflow: Main extraction workflow (flat, no child workflows)

    No activities run on this worker - it's purely for orchestration.
    Checkpoint activities have their own dedicated worker.

    Configuration from WorkerSettings.WORKFLOWS:
    - max_concurrent_workflow_tasks: 100
    - max_cached_workflows: 50
    """
    settings = WorkerSettings.WORKFLOWS

    logger.info("Starting Workflow Worker")

    await run_worker(
        task_queue=TaskQueues.WORKFLOWS,
        workflows=[
            AbstractExtractionWorkflow,
        ],
        activities=None,  # No activities - orchestration only
        max_concurrent_workflow_tasks=settings.get("max_concurrent_workflow_tasks", 100),
        max_cached_workflows=settings.get("max_cached_workflows", 50),
        # Note: idle_shutdown_minutes not set because workflow workers
        # don't track activity execution (idle tracking is activity-based)
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run_workflow_worker())
