"""Drug Worker - Handles drug extraction and validation activities.

This worker:
- Polls the DRUG task queue
- Executes extract_drugs and validate_drugs activities
- Uses fast LLM (GPT-4) for quick responses

Activities:
- extract_drugs: Extract drugs from abstract title
- validate_drugs: Validate drug extraction results

Usage:
    python -m src.temporal.workers.drug_worker
    
    # With idle shutdown (env var)
    IDLE_SHUTDOWN_MINUTES=5 python -m src.temporal.workers.drug_worker
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from src.temporal.config import TaskQueues, WorkerSettings
from src.temporal.activities.drug import extract_drugs, validate_drugs
from src.temporal.workers.base import run_worker

logger = logging.getLogger(__name__)


async def run_drug_worker(idle_shutdown_minutes: float | None = None) -> None:
    """Run the drug activities worker.
    
    Args:
        idle_shutdown_minutes: Auto-shutdown after N minutes of inactivity
    """
    settings = WorkerSettings.DRUG
    
    logger.info("Starting Drug Worker")
    
    await run_worker(
        task_queue=TaskQueues.DRUG,
        workflows=None,
        activities=[extract_drugs, validate_drugs],
        max_concurrent_activities=settings.get("max_concurrent_activities", 15),
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
    asyncio.run(run_drug_worker(idle_shutdown_minutes=idle_minutes))


if __name__ == "__main__":
    main()
