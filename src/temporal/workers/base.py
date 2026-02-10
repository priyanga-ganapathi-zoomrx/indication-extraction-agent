"""Base utilities for Temporal workers.

Provides shared functionality:
- Worker configuration and lifecycle
- Graceful shutdown handling (SIGINT/SIGTERM)
- Idle shutdown support (optional)
"""

import asyncio
import concurrent.futures
import logging
import signal
from datetime import timedelta
from typing import Callable, Optional, Sequence

from temporalio.client import Client
from temporalio.worker import Worker

from src.temporal.config import TEMPORAL_HOST, TEMPORAL_NAMESPACE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_worker(
    task_queue: str,
    workflows: Optional[Sequence[type]] = None,
    activities: Optional[Sequence[Callable]] = None,
    max_concurrent_activities: int = 10,
    max_concurrent_workflow_tasks: int = 100,
    max_cached_workflows: int = 50,
    idle_shutdown_minutes: Optional[float] = None,
    graceful_shutdown_timeout: timedelta = timedelta(seconds=60),
) -> None:
    """Run a Temporal worker with proper configuration.
    
    Handles client creation, thread pool for sync activities,
    signal handlers for graceful shutdown, and optional idle shutdown.
    """
    # Connect to Temporal
    logger.info(f"Connecting to Temporal at {TEMPORAL_HOST}")
    client = await Client.connect(TEMPORAL_HOST, namespace=TEMPORAL_NAMESPACE)
    
    # Build worker config
    worker_kwargs = {
        "client": client,
        "task_queue": task_queue,
        "graceful_shutdown_timeout": graceful_shutdown_timeout,
    }
    
    if workflows:
        worker_kwargs["workflows"] = workflows
        worker_kwargs["max_concurrent_workflow_tasks"] = max_concurrent_workflow_tasks
        worker_kwargs["max_cached_workflows"] = max_cached_workflows
    
    if activities:
        worker_kwargs["activities"] = activities
        worker_kwargs["max_concurrent_activities"] = max_concurrent_activities
    
    # Log startup info
    logger.info(
        f"Starting worker for '{task_queue}' "
        f"(workflows: {len(workflows or [])}, activities: {len(activities or [])})"
    )
    if idle_shutdown_minutes:
        logger.info(f"  Idle shutdown: {idle_shutdown_minutes} minutes")
    
    # Run with thread pool if we have activities
    if activities:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent_activities
        ) as executor:
            worker_kwargs["activity_executor"] = executor
            worker = Worker(**worker_kwargs)
            await _run_with_shutdown(worker, idle_shutdown_minutes)
    else:
        worker = Worker(**worker_kwargs)
        await _run_with_shutdown(worker, idle_shutdown_minutes)
    
    logger.info(f"Worker for '{task_queue}' shut down gracefully")


async def _run_with_shutdown(
    worker: Worker,
    idle_shutdown_minutes: Optional[float] = None,
) -> None:
    """Run worker with signal handling and optional idle shutdown."""
    loop = asyncio.get_running_loop()
    
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(_shutdown(worker, s.name))
        )
    
    # Start idle watchdog if configured
    watchdog_task = None
    if idle_shutdown_minutes and idle_shutdown_minutes > 0:
        from src.temporal.idle_shutdown import idle_watchdog
        watchdog_task = asyncio.create_task(
            idle_watchdog(worker, idle_shutdown_minutes)
        )
    
    try:
        await worker.run()
    finally:
        if watchdog_task and not watchdog_task.done():
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass


async def _shutdown(worker: Worker, signal_name: str) -> None:
    """Handle shutdown signal."""
    logger.info(f"Received {signal_name}, shutting down...")
    await worker.shutdown()
