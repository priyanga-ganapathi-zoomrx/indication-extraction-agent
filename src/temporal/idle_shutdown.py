"""Idle shutdown support for Temporal workers.

This module provides utilities to automatically shut down workers
that have been idle (no activity execution) for a configurable period.

Usage:
    1. Apply @track_activity decorator to all activity functions
    2. Pass idle_shutdown_minutes to run_worker() or run_all_workers()
    3. Worker will automatically shut down after idle period

Components:
    - IdleTracker: Thread-safe tracker for activity execution timestamps
    - track_activity: Decorator to instrument activities for tracking
    - idle_watchdog: Coroutine that monitors idle state and triggers shutdown
"""

import asyncio
import logging
import threading
import time
from functools import wraps
from typing import Callable, TypeVar

from temporalio.worker import Worker

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


class IdleTracker:
    """Thread-safe tracker for activity execution.
    
    Tracks when activities start and complete to determine if
    the worker has been idle for too long.
    
    Thread-safety is required because activities may run in a
    ThreadPoolExecutor.
    """
    
    def __init__(self):
        self._active_count = 0
        self._last_completion = time.monotonic()
        self._has_received_task = False
        self._lock = threading.Lock()
    
    def task_started(self) -> None:
        """Called when an activity starts execution."""
        with self._lock:
            self._active_count += 1
            self._has_received_task = True
    
    def task_completed(self) -> None:
        """Called when an activity completes (success or failure)."""
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
            self._last_completion = time.monotonic()
    
    def is_idle(self, timeout_seconds: float) -> bool:
        """Check if worker has been idle for the specified duration.
        
        Returns False if:
        - No task has ever been received (prevents premature shutdown on startup)
        - There are active tasks running
        - Time since last completion is less than timeout
        """
        with self._lock:
            if not self._has_received_task:
                return False
            if self._active_count > 0:
                return False
            return (time.monotonic() - self._last_completion) >= timeout_seconds
    
    def time_since_last_activity(self) -> float:
        """Get seconds since last activity completed."""
        with self._lock:
            return time.monotonic() - self._last_completion
    
    @property
    def active_count(self) -> int:
        """Get current number of active tasks."""
        with self._lock:
            return self._active_count


# Global tracker instance - shared across all activities
_global_tracker: IdleTracker | None = None


def get_tracker() -> IdleTracker:
    """Get the global idle tracker instance.
    
    Creates one if it doesn't exist.
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = IdleTracker()
    return _global_tracker


def set_tracker(tracker: IdleTracker) -> None:
    """Set the global idle tracker instance.
    
    Used by run_all_workers to share a tracker across workers.
    """
    global _global_tracker
    _global_tracker = tracker


def track_activity(fn: F) -> F:
    """Decorator to track activity execution for idle shutdown.
    
    Apply this decorator to activity functions (after @activity.defn)
    to enable idle shutdown tracking.
    
    Works with both sync and async functions.
    
    Example:
        @activity.defn(name="extract_drugs")
        @track_activity
        def extract_drugs(input_data: DrugInput) -> dict:
            ...
    """
    tracker = get_tracker()
    
    if asyncio.iscoroutinefunction(fn):
        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            tracker.task_started()
            try:
                return await fn(*args, **kwargs)
            finally:
                tracker.task_completed()
        return async_wrapper  # type: ignore
    else:
        @wraps(fn)
        def sync_wrapper(*args, **kwargs):
            tracker.task_started()
            try:
                return fn(*args, **kwargs)
            finally:
                tracker.task_completed()
        return sync_wrapper  # type: ignore


async def idle_watchdog(
    worker: Worker,
    idle_minutes: float,
    check_interval_seconds: float = 30,
) -> None:
    """Background coroutine that monitors idle state and triggers shutdown.
    
    Periodically checks if the worker has been idle for longer than
    the specified threshold. If so, initiates graceful shutdown.
    
    Args:
        worker: The Temporal Worker instance to shut down
        idle_minutes: Idle threshold in minutes (e.g., 5)
        check_interval_seconds: How often to check idle state (default 30s)
    """
    idle_secs = idle_minutes * 60
    tracker = get_tracker()
    
    logger.info(
        f"Idle watchdog started: will shutdown after {idle_minutes} minutes of inactivity"
    )
    
    while True:
        try:
            await asyncio.sleep(check_interval_seconds)
            
            if tracker.is_idle(idle_secs):
                elapsed = tracker.time_since_last_activity()
                logger.info(
                    f"Worker idle for {elapsed:.1f}s (threshold: {idle_secs}s), "
                    "initiating graceful shutdown..."
                )
                await worker.shutdown()
                return
                
        except asyncio.CancelledError:
            logger.debug("Idle watchdog cancelled")
            break
        except Exception as e:
            logger.exception(f"Error in idle watchdog: {e}")
