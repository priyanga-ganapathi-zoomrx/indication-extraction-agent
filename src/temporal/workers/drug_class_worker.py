"""Drug Class Worker - Handles drug class extraction pipeline activities.

This worker:
- Polls the DRUG_CLASS task queue
- Executes the 5-step drug class extraction pipeline
- Handles search (Tavily) and LLM extraction activities

Activities:
- step1_regimen: Identify regimen components
- step2_fetch_search_results: Fetch Tavily search results
- step2_extract_with_tavily: Extract drug classes from search results
- step2_extract_with_grounded: Fallback extraction using LLM web search
- step3_selection: Select best drug class for multi-class drugs
- step4_explicit: Extract explicit drug classes from title
- step5_consolidation: Consolidate and deduplicate classes

Usage:
    python -m src.temporal.workers.drug_class_worker
"""

import asyncio
import logging

from src.temporal.config import TaskQueues, WorkerSettings
from src.temporal.activities.drug_class import (
    step1_regimen,
    step2_fetch_search_results,
    step2_extract_with_tavily,
    step2_extract_with_grounded,
    step3_selection,
    step4_explicit,
    step5_consolidation,
)
from src.temporal.workers.base import run_worker

logger = logging.getLogger(__name__)


async def run_drug_class_worker() -> None:
    """Run the drug class activities worker.
    
    This worker handles the 5-step drug class extraction pipeline.
    It processes multiple steps per abstract (heavier workload),
    including search operations and LLM calls.
    
    Configuration from WorkerSettings.DRUG_CLASS:
    - max_concurrent_activities: 10
    """
    settings = WorkerSettings.DRUG_CLASS
    
    logger.info("Starting Drug Class Worker")
    
    await run_worker(
        task_queue=TaskQueues.DRUG_CLASS,
        workflows=None,  # No workflows - activities only
        activities=[
            step1_regimen,
            step2_fetch_search_results,
            step2_extract_with_tavily,
            step2_extract_with_grounded,
            step3_selection,
            step4_explicit,
            step5_consolidation,
        ],
        max_concurrent_activities=settings.get("max_concurrent_activities", 10),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run_drug_class_worker())
