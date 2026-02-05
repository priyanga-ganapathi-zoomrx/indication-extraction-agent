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
"""

import asyncio
import logging

from src.temporal.config import TaskQueues, WorkerSettings
from src.temporal.activities.drug import extract_drugs, validate_drugs
from src.temporal.workers.base import run_worker

logger = logging.getLogger(__name__)


async def run_drug_worker() -> None:
    """Run the drug activities worker.
    
    This worker handles drug extraction and validation activities.
    Both use fast LLMs (GPT-4) with typical response times of 5-15s.
    
    Configuration from WorkerSettings.DRUG:
    - max_concurrent_activities: 15
    """
    settings = WorkerSettings.DRUG
    
    logger.info("Starting Drug Worker")
    
    await run_worker(
        task_queue=TaskQueues.DRUG,
        workflows=None,  # No workflows - activities only
        activities=[extract_drugs, validate_drugs],
        max_concurrent_activities=settings.get("max_concurrent_activities", 15),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run_drug_worker())
