"""Base utilities for Temporal workers.

Provides shared functionality for all worker types:
- Client creation with environment-based configuration
- Worker configuration helpers
- Graceful shutdown handling

Best Practices Applied:
- Reusable client creation
- ThreadPoolExecutor only for activity workers (not workflow-only)
- Proper shutdown handling with signal handlers
"""

import asyncio
import concurrent.futures
import logging
import signal
from typing import Callable, Optional, Sequence

from temporalio.client import Client
from temporalio.worker import Worker

from src.temporal.config import TEMPORAL_HOST, TEMPORAL_NAMESPACE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def create_temporal_client(
    host: Optional[str] = None,
    namespace: Optional[str] = None,
) -> Client:
    """Create a Temporal client connection.
    
    Args:
        host: Temporal server host (default from environment/config)
        namespace: Temporal namespace (default from environment/config)
        
    Returns:
        Connected Temporal Client
        
    Example:
        >>> client = await create_temporal_client()
        >>> # Use client to start workflows or create workers
    """
    target_host = host or TEMPORAL_HOST
    target_namespace = namespace or TEMPORAL_NAMESPACE
    
    logger.info(f"Connecting to Temporal at {target_host}, namespace: {target_namespace}")
    
    client = await Client.connect(
        target_host,
        namespace=target_namespace,
    )
    
    logger.info("Successfully connected to Temporal")
    return client


async def run_worker(
    task_queue: str,
    workflows: Optional[Sequence[type]] = None,
    activities: Optional[Sequence[Callable]] = None,
    max_concurrent_activities: int = 10,
    max_concurrent_workflow_tasks: int = 100,
    max_cached_workflows: int = 50,
    activity_executor_max_workers: Optional[int] = None,
    client: Optional[Client] = None,
) -> None:
    """Run a Temporal worker with proper configuration.
    
    This function:
    1. Creates a Temporal client (if not provided)
    2. Sets up a ThreadPoolExecutor for sync activities (if activities provided)
    3. Configures the worker with appropriate concurrency settings
    4. Handles graceful shutdown on SIGINT/SIGTERM
    
    Args:
        task_queue: Task queue name to poll
        workflows: Workflow classes to register (optional)
        activities: Activity functions to register (optional)
        max_concurrent_activities: Max activities running concurrently
        max_concurrent_workflow_tasks: Max workflow tasks running concurrently
        max_cached_workflows: Max workflows cached in memory
        activity_executor_max_workers: Thread pool size (defaults to max_concurrent_activities)
        client: Pre-existing Temporal client (optional)
        
    Example:
        >>> # Activity worker
        >>> await run_worker(
        ...     task_queue="drug-activities",
        ...     activities=[extract_drugs, validate_drugs],
        ...     max_concurrent_activities=15,
        ... )
        >>> 
        >>> # Workflow worker (no thread pool needed)
        >>> await run_worker(
        ...     task_queue="extraction-workflows",
        ...     workflows=[AbstractExtractionWorkflow],
        ... )
    """
    # Create client if not provided
    if client is None:
        client = await create_temporal_client()
    
    # Setup shutdown event
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        shutdown_event.set()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Build worker kwargs
    worker_kwargs = {
        "client": client,
        "task_queue": task_queue,
    }
    
    if workflows:
        worker_kwargs["workflows"] = workflows
        worker_kwargs["max_concurrent_workflow_tasks"] = max_concurrent_workflow_tasks
        worker_kwargs["max_cached_workflows"] = max_cached_workflows
    
    if activities:
        worker_kwargs["activities"] = activities
        worker_kwargs["max_concurrent_activities"] = max_concurrent_activities
    
    # Only create thread pool if we have activities
    # Workflow-only workers don't need a thread pool
    if activities:
        executor_workers = activity_executor_max_workers or max_concurrent_activities
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=executor_workers
        ) as activity_executor:
            worker_kwargs["activity_executor"] = activity_executor
            
            worker = Worker(**worker_kwargs)
            
            _log_worker_info(task_queue, workflows, activities, max_concurrent_activities, executor_workers)
            
            async with worker:
                await shutdown_event.wait()
    else:
        # Workflow-only worker - no thread pool needed
        worker = Worker(**worker_kwargs)
        
        _log_worker_info(task_queue, workflows, activities, max_concurrent_activities, None)
        
        async with worker:
            await shutdown_event.wait()
    
    logger.info(f"Worker for '{task_queue}' shut down gracefully")


def _log_worker_info(
    task_queue: str,
    workflows: Optional[Sequence[type]],
    activities: Optional[Sequence[Callable]],
    max_concurrent_activities: int,
    executor_workers: Optional[int],
) -> None:
    """Log worker startup information."""
    logger.info(
        f"Starting worker for task queue '{task_queue}' "
        f"(workflows: {len(workflows or [])}, activities: {len(activities or [])})"
    )
    
    if workflows:
        logger.info(f"  Registered workflows: {[w.__name__ for w in workflows]}")
    if activities:
        logger.info(f"  Registered activities: {[a.__name__ for a in activities]}")
        logger.info(f"  Max concurrent activities: {max_concurrent_activities}")
        if executor_workers:
            logger.info(f"  Activity executor workers: {executor_workers}")
