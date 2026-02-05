"""Indication Validation Worker - Handles indication validation activities.

This worker:
- Polls the INDICATION_VALIDATION task queue
- Executes validate_indication activity
- Uses SLOW LLM (Sonnet 4.5) - isolated for independent scaling

Activities:
- validate_indication: Validate indication extraction results

Note: This worker is separate because indication validation uses
Claude Sonnet 4.5 which has higher latency (30-60s typical).
Isolation prevents slow validation from blocking other activities.

Usage:
    python -m src.temporal.workers.indication_validation_worker
"""

import asyncio
import logging

from src.temporal.config import TaskQueues, WorkerSettings
from src.temporal.activities.indication import validate_indication
from src.temporal.workers.base import run_worker

logger = logging.getLogger(__name__)


async def run_indication_validation_worker() -> None:
    """Run the indication validation worker.
    
    This worker handles indication validation using a LangGraph
    agent with tool calling. Uses slow LLM (Sonnet 4.5) with
    30-60s typical response times.
    
    Configuration from WorkerSettings.INDICATION_VALIDATION:
    - max_concurrent_activities: 5 (limited due to slow LLM)
    """
    settings = WorkerSettings.INDICATION_VALIDATION
    
    logger.info("Starting Indication Validation Worker (Slow)")
    
    await run_worker(
        task_queue=TaskQueues.INDICATION_VALIDATION,
        workflows=None,  # No workflows - activities only
        activities=[validate_indication],
        max_concurrent_activities=settings.get("max_concurrent_activities", 5),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run_indication_validation_worker())
