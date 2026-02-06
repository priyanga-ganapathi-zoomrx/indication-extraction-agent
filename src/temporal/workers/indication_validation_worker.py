"""Indication Validation Worker - Handles indication validation activities.

This worker:
- Polls the INDICATION_VALIDATION task queue
- Executes validate_indication activity
- Uses slow LLM (Sonnet 4.5) for thorough validation

Activities:
- validate_indication: Validate extracted indication against rules

Note: This worker is separated from extraction because validation uses
a slower, more expensive LLM model (Sonnet 4.5) with typical response
times of 30-60s. Separating ensures fast extraction isn't blocked.

Usage:
    python -m src.temporal.workers.indication_validation_worker
    
    # With idle shutdown (env var)
    IDLE_SHUTDOWN_MINUTES=5 python -m src.temporal.workers.indication_validation_worker
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from src.temporal.config import TaskQueues, WorkerSettings
from src.temporal.activities.indication import validate_indication
from src.temporal.workers.base import run_worker

logger = logging.getLogger(__name__)


async def run_indication_validation_worker(
    idle_shutdown_minutes: float | None = None,
) -> None:
    """Run the indication validation worker.
    
    Args:
        idle_shutdown_minutes: Auto-shutdown after N minutes of inactivity
    """
    settings = WorkerSettings.INDICATION_VALIDATION
    
    logger.info("Starting Indication Validation Worker")
    
    await run_worker(
        task_queue=TaskQueues.INDICATION_VALIDATION,
        workflows=None,
        activities=[validate_indication],
        max_concurrent_activities=settings.get("max_concurrent_activities", 5),
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
    asyncio.run(run_indication_validation_worker(idle_shutdown_minutes=idle_minutes))


if __name__ == "__main__":
    main()
