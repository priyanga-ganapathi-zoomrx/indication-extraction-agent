"""Temporal configuration for abstract extraction workflows.

This module defines:
- Task queue names for routing work to appropriate workers
- Timeout configurations for different activity types
- Retry policies following Temporal best practices
- Worker settings for concurrency control

Best Practices Applied:
- Fine-grained activities with appropriate timeouts
- Separate task queues for different workload characteristics
- Retry policies tuned for LLM API behavior
- Non-retryable errors for validation failures
"""

from datetime import timedelta
from temporalio.common import RetryPolicy


# =============================================================================
# TASK QUEUES
# =============================================================================
# Separate queues allow:
# - Independent scaling per workload type
# - Different timeout/retry configurations
# - Isolation of slow activities (Sonnet 4.5) from fast ones

class TaskQueues:
    """Task queue names as constants to ensure consistency."""
    
    # Workflow queue - handles orchestration only (lightweight)
    WORKFLOWS = "extraction-workflows"
    
    # Activity queues - grouped by entity and latency profile
    DRUG = "drug-activities"
    DRUG_CLASS = "drug-class-activities"
    INDICATION_EXTRACTION = "indication-extraction"
    INDICATION_VALIDATION = "indication-validation-slow"  # Sonnet 4.5 - high latency


# Convenience dict for programmatic access
TASK_QUEUES = {
    "workflows": TaskQueues.WORKFLOWS,
    "drug": TaskQueues.DRUG,
    "drug_class": TaskQueues.DRUG_CLASS,
    "indication_extraction": TaskQueues.INDICATION_EXTRACTION,
    "indication_validation": TaskQueues.INDICATION_VALIDATION,
}


# =============================================================================
# TIMEOUTS
# =============================================================================
# Timeout best practices:
# - start_to_close_timeout: Max time for activity execution
# - schedule_to_close_timeout: Total time from scheduling to completion
# - heartbeat_timeout: For long-running activities (not needed for LLM calls)
#
# LLM calls typically complete in 5-60s, but can occasionally take longer.
# Set timeouts with buffer for retries and network variability.

class Timeouts:
    """Timeout configurations for different activity types."""
    
    # Fast LLM activities (GPT-4, typically 5-15s)
    FAST_LLM = timedelta(minutes=2)
    
    # Slow LLM activities (Sonnet 4.5, typically 30-60s)
    SLOW_LLM = timedelta(minutes=5)
    
    # Search activities (Tavily API, typically 2-10s)
    SEARCH = timedelta(seconds=45)
    
    # Storage/DB activities (typically <1s)
    STORAGE = timedelta(seconds=30)
    
    # Workflow execution timeout (entire abstract processing)
    WORKFLOW_EXECUTION = timedelta(minutes=30)
    
    # Workflow run timeout (single workflow run, before continue-as-new)
    WORKFLOW_RUN = timedelta(minutes=30)


TIMEOUTS = {
    "fast_llm": Timeouts.FAST_LLM,
    "slow_llm": Timeouts.SLOW_LLM,
    "search": Timeouts.SEARCH,
    "storage": Timeouts.STORAGE,
    "workflow_execution": Timeouts.WORKFLOW_EXECUTION,
    "workflow_run": Timeouts.WORKFLOW_RUN,
}


# =============================================================================
# RETRY POLICIES
# =============================================================================
# Retry best practices:
# - initial_interval: Start with reasonable backoff (not too aggressive)
# - backoff_coefficient: 2.0 is standard exponential backoff
# - maximum_interval: Cap to avoid very long waits
# - maximum_attempts: Limit to avoid infinite retries on persistent failures
# - non_retryable_error_types: Skip retries for validation/parsing errors

class RetryPolicies:
    """Retry policies for different activity types."""
    
    # Fast LLM activities - moderate retries
    FAST_LLM = RetryPolicy(
        initial_interval=timedelta(seconds=5),
        backoff_coefficient=2.0,
        maximum_interval=timedelta(seconds=30),
        maximum_attempts=3,
        non_retryable_error_types=[
            "ValueError",           # Bad input data
            "ValidationError",      # Pydantic validation failures
        ],
    )
    
    # Slow LLM activities - fewer retries (expensive, high latency)
    SLOW_LLM = RetryPolicy(
        initial_interval=timedelta(seconds=15),
        backoff_coefficient=2.0,
        maximum_interval=timedelta(minutes=1),
        maximum_attempts=2,  # Only 1 retry - these are expensive
        non_retryable_error_types=[
            "ValueError",
            "ValidationError",
        ],
    )
    
    # Search activities - quick retries for transient failures
    SEARCH = RetryPolicy(
        initial_interval=timedelta(seconds=2),
        backoff_coefficient=2.0,
        maximum_interval=timedelta(seconds=15),
        maximum_attempts=3,
        non_retryable_error_types=[
            "ValueError",
        ],
    )
    
    # Storage activities - quick retries
    STORAGE = RetryPolicy(
        initial_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
        maximum_interval=timedelta(seconds=10),
        maximum_attempts=3,
        non_retryable_error_types=[
            "ValueError",
            "IntegrityError",  # Database constraint violations
        ],
    )


RETRY_POLICIES = {
    "fast_llm": RetryPolicies.FAST_LLM,
    "slow_llm": RetryPolicies.SLOW_LLM,
    "search": RetryPolicies.SEARCH,
    "storage": RetryPolicies.STORAGE,
}


# =============================================================================
# WORKER SETTINGS
# =============================================================================
# Worker configuration best practices:
# - max_concurrent_workflow_tasks: Workflows are lightweight, can be high
# - max_concurrent_activities: Tune based on LLM rate limits and latency
# - Separate workers per queue to avoid resource contention
#
# Note: Running multiple queues in one worker is an anti-pattern per Temporal docs

class WorkerSettings:
    """Worker configuration for each task queue."""
    
    # Workflow worker - lightweight orchestration
    WORKFLOWS = {
        "max_concurrent_workflow_tasks": 100,
        "max_cached_workflows": 50,
    }
    
    # Drug activities - fast LLM calls
    DRUG = {
        "max_concurrent_activities": 15,
    }
    
    # Drug class activities - heavier workload (5 steps per abstract)
    DRUG_CLASS = {
        "max_concurrent_activities": 10,
    }
    
    # Indication extraction - fast LLM calls
    INDICATION_EXTRACTION = {
        "max_concurrent_activities": 20,
    }
    
    # Indication validation - slow (Sonnet 4.5), limit concurrent
    INDICATION_VALIDATION = {
        "max_concurrent_activities": 5,
    }


WORKER_SETTINGS = {
    "workflows": WorkerSettings.WORKFLOWS,
    "drug": WorkerSettings.DRUG,
    "drug_class": WorkerSettings.DRUG_CLASS,
    "indication_extraction": WorkerSettings.INDICATION_EXTRACTION,
    "indication_validation": WorkerSettings.INDICATION_VALIDATION,
}


# =============================================================================
# TEMPORAL SERVER CONNECTION
# =============================================================================
# Connection settings - override via environment variables in production

import os

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")
TEMPORAL_NAMESPACE = os.getenv("TEMPORAL_NAMESPACE", "default")


# =============================================================================
# ACTIVITY OPTIONS HELPERS
# =============================================================================
# Helper functions to create consistent activity options

def get_fast_llm_activity_options() -> dict:
    """Get activity options for fast LLM activities (GPT-4)."""
    return {
        "start_to_close_timeout": Timeouts.FAST_LLM,
        "retry_policy": RetryPolicies.FAST_LLM,
    }


def get_slow_llm_activity_options() -> dict:
    """Get activity options for slow LLM activities (Sonnet 4.5)."""
    return {
        "start_to_close_timeout": Timeouts.SLOW_LLM,
        "retry_policy": RetryPolicies.SLOW_LLM,
    }


def get_search_activity_options() -> dict:
    """Get activity options for search activities."""
    return {
        "start_to_close_timeout": Timeouts.SEARCH,
        "retry_policy": RetryPolicies.SEARCH,
    }


def get_storage_activity_options() -> dict:
    """Get activity options for storage/DB activities."""
    return {
        "start_to_close_timeout": Timeouts.STORAGE,
        "retry_policy": RetryPolicies.STORAGE,
    }
