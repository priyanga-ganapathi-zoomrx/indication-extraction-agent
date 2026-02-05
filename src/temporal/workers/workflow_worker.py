"""Workflow Worker - Handles workflow orchestration tasks.

This worker:
- Polls the WORKFLOWS task queue
- Executes AbstractExtractionWorkflow orchestration
- Lightweight - no activities, just workflow coordination

Best Practice: Workflow workers are separate from activity workers
to allow independent scaling and avoid resource contention.

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
    
    This worker handles workflow orchestration for the
    AbstractExtractionWorkflow. It coordinates activity execution
    across different task queues but doesn't run activities itself.
    
    Configuration from WorkerSettings.WORKFLOWS:
    - max_concurrent_workflow_tasks: 100
    - max_cached_workflows: 50
    """
    settings = WorkerSettings.WORKFLOWS
    
    logger.info("Starting Workflow Worker")
    
    await run_worker(
        task_queue=TaskQueues.WORKFLOWS,
        workflows=[AbstractExtractionWorkflow],
        activities=None,  # No activities - workflows only
        max_concurrent_workflow_tasks=settings.get("max_concurrent_workflow_tasks", 100),
        max_cached_workflows=settings.get("max_cached_workflows", 50),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run_workflow_worker())
