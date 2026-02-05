"""Run All Workers - Development helper to run all workers in one process.

This script runs all workers concurrently in a single process.
Useful for local development and testing.

For production, run each worker separately for:
- Independent scaling
- Better resource isolation
- Easier monitoring and deployment

Usage:
    python -m src.temporal.workers.run_all
    
    # Or with poetry
    poetry run python -m src.temporal.workers.run_all

Environment Variables:
    TEMPORAL_HOST: Temporal server address (default: localhost:7233)
    TEMPORAL_NAMESPACE: Temporal namespace (default: default)
"""

import asyncio
import concurrent.futures
import logging
import signal
from typing import Optional

from temporalio.client import Client
from temporalio.worker import Worker

from src.temporal.config import (
    TaskQueues,
    WorkerSettings,
    TEMPORAL_HOST,
    TEMPORAL_NAMESPACE,
)
from src.temporal.workflows import AbstractExtractionWorkflow
from src.temporal.activities import (
    DRUG_ACTIVITIES,
    DRUG_CLASS_ACTIVITIES,
    INDICATION_ACTIVITIES,
)

logger = logging.getLogger(__name__)


async def run_all_workers(client: Optional[Client] = None) -> None:
    """Run all workers concurrently in a single process.
    
    Note: For production, run each worker separately using the individual
    worker scripts. This combined approach is for development only.
    
    Args:
        client: Optional pre-existing Temporal client
    """
    # Create client if not provided
    if client is None:
        logger.info(f"Connecting to Temporal at {TEMPORAL_HOST}")
        client = await Client.connect(TEMPORAL_HOST, namespace=TEMPORAL_NAMESPACE)
        logger.info("Connected to Temporal")
    
    # Setup shutdown event
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create thread pool for activities
    # Size should accommodate all workers' max concurrent activities
    total_max_activities = (
        WorkerSettings.DRUG.get("max_concurrent_activities", 15) +
        WorkerSettings.DRUG_CLASS.get("max_concurrent_activities", 10) +
        WorkerSettings.INDICATION_EXTRACTION.get("max_concurrent_activities", 20) +
        WorkerSettings.INDICATION_VALIDATION.get("max_concurrent_activities", 5)
    )
    
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=total_max_activities
    ) as activity_executor:
        
        # Create workers for each task queue
        workers = []
        
        # 1. Workflow worker (no activities)
        workflow_settings = WorkerSettings.WORKFLOWS
        workflow_worker = Worker(
            client,
            task_queue=TaskQueues.WORKFLOWS,
            workflows=[AbstractExtractionWorkflow],
            max_concurrent_workflow_tasks=workflow_settings.get(
                "max_concurrent_workflow_tasks", 100
            ),
            max_cached_workflows=workflow_settings.get("max_cached_workflows", 50),
        )
        workers.append(("workflows", workflow_worker))
        
        # 2. Drug activities worker
        drug_settings = WorkerSettings.DRUG
        drug_worker = Worker(
            client,
            task_queue=TaskQueues.DRUG,
            activities=DRUG_ACTIVITIES,
            activity_executor=activity_executor,
            max_concurrent_activities=drug_settings.get(
                "max_concurrent_activities", 15
            ),
        )
        workers.append(("drug", drug_worker))
        
        # 3. Drug class activities worker
        drug_class_settings = WorkerSettings.DRUG_CLASS
        drug_class_worker = Worker(
            client,
            task_queue=TaskQueues.DRUG_CLASS,
            activities=DRUG_CLASS_ACTIVITIES,
            activity_executor=activity_executor,
            max_concurrent_activities=drug_class_settings.get(
                "max_concurrent_activities", 10
            ),
        )
        workers.append(("drug_class", drug_class_worker))
        
        # 4. Indication extraction worker
        indication_extraction_settings = WorkerSettings.INDICATION_EXTRACTION
        indication_extraction_worker = Worker(
            client,
            task_queue=TaskQueues.INDICATION_EXTRACTION,
            activities=[INDICATION_ACTIVITIES[0]],  # extract_indication only
            activity_executor=activity_executor,
            max_concurrent_activities=indication_extraction_settings.get(
                "max_concurrent_activities", 20
            ),
        )
        workers.append(("indication_extraction", indication_extraction_worker))
        
        # 5. Indication validation worker (slow)
        indication_validation_settings = WorkerSettings.INDICATION_VALIDATION
        indication_validation_worker = Worker(
            client,
            task_queue=TaskQueues.INDICATION_VALIDATION,
            activities=[INDICATION_ACTIVITIES[1]],  # validate_indication only
            activity_executor=activity_executor,
            max_concurrent_activities=indication_validation_settings.get(
                "max_concurrent_activities", 5
            ),
        )
        workers.append(("indication_validation", indication_validation_worker))
        
        # Log worker configuration
        logger.info("=" * 60)
        logger.info("Starting All Workers (Development Mode)")
        logger.info("=" * 60)
        for name, worker in workers:
            logger.info(f"  - {name}: {worker.task_queue}")
        logger.info(f"  Total thread pool workers: {total_max_activities}")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to shutdown gracefully")
        
        # Run all workers concurrently
        async def run_single_worker(name: str, worker: Worker):
            async with worker:
                logger.info(f"Worker '{name}' started")
                await shutdown_event.wait()
                logger.info(f"Worker '{name}' shutting down...")
        
        # Start all workers
        await asyncio.gather(
            *[run_single_worker(name, worker) for name, worker in workers]
        )
        
        logger.info("All workers shut down gracefully")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run_all_workers())
