"""Indication Extraction Worker - Handles indication extraction activities.

This worker:
- Polls the INDICATION_EXTRACTION task queue
- Executes extract_indication activity
- Uses fast LLM (GPT-4) for extraction

Activities:
- extract_indication: Extract medical indication from abstract titles

Usage:
    python -m src.temporal.workers.indication_extraction_worker
    
    # With idle shutdown (env var)
    IDLE_SHUTDOWN_MINUTES=5 python -m src.temporal.workers.indication_extraction_worker
"""

import asyncio
import logging
import os

from src.temporal.config import TaskQueues, WorkerSettings
from src.temporal.activities.indication import extract_indication
from src.temporal.workers.base import run_worker

logger = logging.getLogger(__name__)


async def run_indication_extraction_worker(
    idle_shutdown_minutes: float | None = None,
) -> None:
    """Run the indication extraction worker.
    
    Args:
        idle_shutdown_minutes: Auto-shutdown after N minutes of inactivity
    """
    settings = WorkerSettings.INDICATION_EXTRACTION
    
    logger.info("Starting Indication Extraction Worker")
    
    await run_worker(
        task_queue=TaskQueues.INDICATION_EXTRACTION,
        workflows=None,
        activities=[extract_indication],
        max_concurrent_activities=settings.get("max_concurrent_activities", 20),
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
    asyncio.run(run_indication_extraction_worker(idle_shutdown_minutes=idle_minutes))


if __name__ == "__main__":
    main()
