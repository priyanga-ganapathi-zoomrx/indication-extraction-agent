"""Drug Class Worker - Handles drug class extraction pipeline activities.

This worker:
- Polls the DRUG_CLASS task queue
- Executes the 5-step drug class extraction pipeline + validation
- Handles search (Tavily) and LLM extraction activities

Activities:
- step1_regimen: Identify regimen components
- step2_fetch_search_results: Fetch Tavily search results
- step2_extract_with_tavily: Extract drug classes from search results
- step2_extract_with_grounded: Fallback extraction using LLM web search
- step3_selection: Select best drug class for multi-class drugs
- step4_explicit: Extract explicit drug classes from title
- step5_consolidation: Consolidate and deduplicate classes
- validate_drug_class_activity: Validate extraction results

Usage:
    python -m src.temporal.workers.drug_class_worker
    
    # With idle shutdown (env var)
    IDLE_SHUTDOWN_MINUTES=5 python -m src.temporal.workers.drug_class_worker
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from src.temporal.config import TaskQueues, WorkerSettings
from src.temporal.activities.drug_class import (
    step1_regimen,
    step2_fetch_search_results,
    step2_extract_with_tavily,
    step2_extract_with_grounded,
    step3_selection,
    step4_explicit,
    step5_consolidation,
    validate_drug_class_activity,
)
from src.temporal.workers.base import run_worker

logger = logging.getLogger(__name__)


async def run_drug_class_worker(idle_shutdown_minutes: float | None = None) -> None:
    """Run the drug class activities worker.
    
    Args:
        idle_shutdown_minutes: Auto-shutdown after N minutes of inactivity
    """
    settings = WorkerSettings.DRUG_CLASS
    
    logger.info("Starting Drug Class Worker")
    
    await run_worker(
        task_queue=TaskQueues.DRUG_CLASS,
        workflows=None,
        activities=[
            step1_regimen,
            step2_fetch_search_results,
            step2_extract_with_tavily,
            step2_extract_with_grounded,
            step3_selection,
            step4_explicit,
            step5_consolidation,
            validate_drug_class_activity,
        ],
        max_concurrent_activities=settings.get("max_concurrent_activities", 10),
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
    asyncio.run(run_drug_class_worker(idle_shutdown_minutes=idle_minutes))


if __name__ == "__main__":
    main()
