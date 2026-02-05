"""Indication Extraction Worker - Handles indication extraction activities.

This worker:
- Polls the INDICATION_EXTRACTION task queue
- Executes extract_indication activity
- Uses LangGraph agent with tool calling

Activities:
- extract_indication: Extract medical indication from abstract title

Usage:
    python -m src.temporal.workers.indication_extraction_worker
"""

import asyncio
import logging

from src.temporal.config import TaskQueues, WorkerSettings
from src.temporal.activities.indication import extract_indication
from src.temporal.workers.base import run_worker

logger = logging.getLogger(__name__)


async def run_indication_extraction_worker() -> None:
    """Run the indication extraction worker.
    
    This worker handles indication extraction using a LangGraph
    agent with tool calling for rules retrieval. Uses fast LLM.
    
    Configuration from WorkerSettings.INDICATION_EXTRACTION:
    - max_concurrent_activities: 20
    """
    settings = WorkerSettings.INDICATION_EXTRACTION
    
    logger.info("Starting Indication Extraction Worker")
    
    await run_worker(
        task_queue=TaskQueues.INDICATION_EXTRACTION,
        workflows=None,  # No workflows - activities only
        activities=[extract_indication],
        max_concurrent_activities=settings.get("max_concurrent_activities", 20),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run_indication_extraction_worker())
