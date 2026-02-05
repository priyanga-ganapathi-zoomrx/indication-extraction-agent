"""Client utilities for abstract extraction workflows.

This module provides high-level functions for:
- Starting single and batch extraction workflows
- Waiting for workflow results
- Querying workflow state
- Managing workflow lifecycle

Best Practices Applied:
- Workflow IDs map to business entities (abstract_id)
- Consistent client creation with environment config
- Support for both sync and async result retrieval
- Batch processing with controlled concurrency
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import AsyncIterator, Optional, Union

from temporalio.client import Client, WorkflowHandle, WorkflowExecutionStatus

from src.temporal.config import (
    TaskQueues,
    Timeouts,
    TEMPORAL_HOST,
    TEMPORAL_NAMESPACE,
)
from src.temporal.workflows import (
    AbstractExtractionWorkflow,
    AbstractExtractionInput,
    AbstractExtractionOutput,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CLIENT MANAGEMENT
# =============================================================================

# Global client cache for connection reuse
_client_cache: Optional[Client] = None


async def get_client(
    host: Optional[str] = None,
    namespace: Optional[str] = None,
    use_cache: bool = True,
) -> Client:
    """Get or create a Temporal client.
    
    By default, reuses a cached client for efficiency.
    Set use_cache=False to create a new connection.
    
    Args:
        host: Temporal server address (default from config)
        namespace: Temporal namespace (default from config)
        use_cache: Whether to use cached client (default True)
        
    Returns:
        Connected Temporal Client
        
    Example:
        >>> client = await get_client()
        >>> # Use client to start workflows
    """
    global _client_cache
    
    target_host = host or TEMPORAL_HOST
    target_namespace = namespace or TEMPORAL_NAMESPACE
    
    # Return cached client if available and requested
    if use_cache and _client_cache is not None:
        return _client_cache
    
    logger.info(f"Connecting to Temporal at {target_host}, namespace: {target_namespace}")
    
    client = await Client.connect(
        target_host,
        namespace=target_namespace,
    )
    
    # Cache the client
    if use_cache:
        _client_cache = client
    
    logger.info("Connected to Temporal")
    return client


def generate_workflow_id(abstract_id: str, prefix: str = "extraction") -> str:
    """Generate a consistent workflow ID from abstract ID.
    
    Best Practice: Map workflow IDs to business entities for easy lookup.
    
    Args:
        abstract_id: The abstract identifier
        prefix: Optional prefix for the workflow ID
        
    Returns:
        Workflow ID in format: {prefix}-{abstract_id}
        
    Example:
        >>> generate_workflow_id("12345")
        "extraction-12345"
    """
    return f"{prefix}-{abstract_id}"


# =============================================================================
# SINGLE WORKFLOW OPERATIONS
# =============================================================================

async def start_extraction(
    abstract_id: str,
    abstract_title: str,
    session_title: str = "",
    full_abstract: str = "",
    firms: Optional[list[str]] = None,
    skip_drug_validation: bool = False,
    skip_indication_validation: bool = False,
    skip_drug_class: bool = False,
    storage_base_path: str = "",
    client: Optional[Client] = None,
    workflow_id: Optional[str] = None,
) -> WorkflowHandle:
    """Start an extraction workflow without waiting for completion.
    
    Use this for fire-and-forget style execution where you'll
    retrieve results later.
    
    Args:
        abstract_id: Unique identifier for the abstract
        abstract_title: The abstract title text
        session_title: Optional session/conference title
        full_abstract: Optional full abstract text
        firms: Optional list of pharmaceutical company names
        skip_drug_validation: Skip drug validation step
        skip_indication_validation: Skip indication validation step
        skip_drug_class: Skip drug class extraction
        storage_base_path: Base path for search cache storage
        client: Optional pre-existing Temporal client
        workflow_id: Optional custom workflow ID (default: extraction-{abstract_id})
        
    Returns:
        WorkflowHandle for tracking/querying the workflow
        
    Example:
        >>> handle = await start_extraction(
        ...     abstract_id="12345",
        ...     abstract_title="Phase 3 study of Drug X in NSCLC",
        ...     session_title="Lung Cancer Oral Presentations"
        ... )
        >>> print(f"Started workflow: {handle.id}")
        >>> # Later, get result
        >>> result = await handle.result()
    """
    if client is None:
        client = await get_client()
    
    # Build input
    input_data = AbstractExtractionInput(
        abstract_id=abstract_id,
        abstract_title=abstract_title,
        session_title=session_title,
        full_abstract=full_abstract,
        firms=firms or [],
        skip_drug_validation=skip_drug_validation,
        skip_indication_validation=skip_indication_validation,
        skip_drug_class=skip_drug_class,
        storage_base_path=storage_base_path,
    )
    
    # Generate workflow ID
    wf_id = workflow_id or generate_workflow_id(abstract_id)
    
    logger.info(f"Starting extraction workflow {wf_id} for abstract {abstract_id}")
    
    # Start workflow
    handle = await client.start_workflow(
        AbstractExtractionWorkflow.run,
        input_data,
        id=wf_id,
        task_queue=TaskQueues.WORKFLOWS,
        execution_timeout=Timeouts.WORKFLOW_EXECUTION,
        run_timeout=Timeouts.WORKFLOW_RUN,
    )
    
    logger.info(f"Started workflow {handle.id}, run ID: {handle.result_run_id}")
    
    return handle


async def execute_extraction(
    abstract_id: str,
    abstract_title: str,
    session_title: str = "",
    full_abstract: str = "",
    firms: Optional[list[str]] = None,
    skip_drug_validation: bool = False,
    skip_indication_validation: bool = False,
    skip_drug_class: bool = False,
    storage_base_path: str = "",
    client: Optional[Client] = None,
    workflow_id: Optional[str] = None,
) -> AbstractExtractionOutput:
    """Execute an extraction workflow and wait for the result.
    
    Use this when you need the result immediately.
    Blocks until workflow completes.
    
    Args:
        abstract_id: Unique identifier for the abstract
        abstract_title: The abstract title text
        session_title: Optional session/conference title
        full_abstract: Optional full abstract text
        firms: Optional list of pharmaceutical company names
        skip_drug_validation: Skip drug validation step
        skip_indication_validation: Skip indication validation step
        skip_drug_class: Skip drug class extraction
        storage_base_path: Base path for search cache storage
        client: Optional pre-existing Temporal client
        workflow_id: Optional custom workflow ID
        
    Returns:
        AbstractExtractionOutput with all extraction results
        
    Raises:
        WorkflowFailureError: If workflow fails
        
    Example:
        >>> result = await execute_extraction(
        ...     abstract_id="12345",
        ...     abstract_title="Phase 3 study of Drug X in NSCLC",
        ... )
        >>> print(f"Extracted drugs: {result.drug.extraction.get('primary_drugs')}")
    """
    if client is None:
        client = await get_client()
    
    # Build input
    input_data = AbstractExtractionInput(
        abstract_id=abstract_id,
        abstract_title=abstract_title,
        session_title=session_title,
        full_abstract=full_abstract,
        firms=firms or [],
        skip_drug_validation=skip_drug_validation,
        skip_indication_validation=skip_indication_validation,
        skip_drug_class=skip_drug_class,
        storage_base_path=storage_base_path,
    )
    
    # Generate workflow ID
    wf_id = workflow_id or generate_workflow_id(abstract_id)
    
    logger.info(f"Executing extraction workflow {wf_id} for abstract {abstract_id}")
    
    # Execute workflow and wait for result
    result = await client.execute_workflow(
        AbstractExtractionWorkflow.run,
        input_data,
        id=wf_id,
        task_queue=TaskQueues.WORKFLOWS,
        execution_timeout=Timeouts.WORKFLOW_EXECUTION,
        run_timeout=Timeouts.WORKFLOW_RUN,
    )
    
    logger.info(f"Workflow {wf_id} completed")
    
    return result


# =============================================================================
# BATCH WORKFLOW OPERATIONS
# =============================================================================

@dataclass
class BatchItem:
    """Single item for batch extraction."""
    abstract_id: str
    abstract_title: str
    session_title: str = ""
    full_abstract: str = ""
    firms: list[str] = None
    
    def __post_init__(self):
        if self.firms is None:
            self.firms = []


@dataclass
class BatchResult:
    """Result of a batch extraction item."""
    abstract_id: str
    workflow_id: str
    output: Optional[AbstractExtractionOutput] = None
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.output is not None and self.error is None


async def start_batch_extraction(
    items: list[BatchItem],
    max_concurrent: int = 50,
    skip_drug_validation: bool = False,
    skip_indication_validation: bool = False,
    skip_drug_class: bool = False,
    storage_base_path: str = "",
    client: Optional[Client] = None,
) -> AsyncIterator[BatchResult]:
    """Start batch extraction with controlled concurrency.
    
    Processes items using a semaphore to limit concurrent workflows.
    Yields results as they complete (not necessarily in order).
    
    Args:
        items: List of BatchItem objects to process
        max_concurrent: Maximum concurrent workflow starts (default 50)
        skip_drug_validation: Skip drug validation for all items
        skip_indication_validation: Skip indication validation for all items
        skip_drug_class: Skip drug class extraction for all items
        storage_base_path: Base path for search cache storage
        client: Optional pre-existing Temporal client
        
    Yields:
        BatchResult objects as workflows complete
        
    Example:
        >>> items = [
        ...     BatchItem(abstract_id="1", abstract_title="Study A..."),
        ...     BatchItem(abstract_id="2", abstract_title="Study B..."),
        ... ]
        >>> async for result in start_batch_extraction(items, max_concurrent=10):
        ...     if result.success:
        ...         print(f"Completed: {result.abstract_id}")
        ...     else:
        ...         print(f"Failed: {result.abstract_id}: {result.error}")
    """
    if client is None:
        client = await get_client()
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item: BatchItem) -> BatchResult:
        """Process a single item with semaphore control."""
        async with semaphore:
            workflow_id = generate_workflow_id(item.abstract_id)
            
            try:
                output = await execute_extraction(
                    abstract_id=item.abstract_id,
                    abstract_title=item.abstract_title,
                    session_title=item.session_title,
                    full_abstract=item.full_abstract,
                    firms=item.firms,
                    skip_drug_validation=skip_drug_validation,
                    skip_indication_validation=skip_indication_validation,
                    skip_drug_class=skip_drug_class,
                    storage_base_path=storage_base_path,
                    client=client,
                    workflow_id=workflow_id,
                )
                
                return BatchResult(
                    abstract_id=item.abstract_id,
                    workflow_id=workflow_id,
                    output=output,
                )
                
            except Exception as e:
                logger.error(f"Batch item {item.abstract_id} failed: {e}")
                return BatchResult(
                    abstract_id=item.abstract_id,
                    workflow_id=workflow_id,
                    error=str(e),
                )
    
    # Create tasks for all items
    tasks = [asyncio.create_task(process_item(item)) for item in items]
    
    logger.info(
        f"Started batch extraction for {len(items)} items "
        f"(max concurrent: {max_concurrent})"
    )
    
    # Yield results as they complete
    for completed in asyncio.as_completed(tasks):
        result = await completed
        yield result


# =============================================================================
# WORKFLOW STATUS AND RESULTS
# =============================================================================

async def get_extraction_status(
    abstract_id: str,
    client: Optional[Client] = None,
    workflow_id: Optional[str] = None,
) -> dict:
    """Get the status of an extraction workflow.
    
    Args:
        abstract_id: The abstract identifier
        client: Optional pre-existing Temporal client
        workflow_id: Optional custom workflow ID
        
    Returns:
        Dict with workflow status information:
        - status: WorkflowExecutionStatus enum value
        - workflow_id: The workflow ID
        - run_id: The run ID
        - start_time: When workflow started
        - close_time: When workflow completed (if finished)
        
    Example:
        >>> status = await get_extraction_status("12345")
        >>> print(f"Status: {status['status']}")
    """
    if client is None:
        client = await get_client()
    
    wf_id = workflow_id or generate_workflow_id(abstract_id)
    
    handle = client.get_workflow_handle(wf_id)
    description = await handle.describe()
    
    return {
        "status": description.status,
        "workflow_id": wf_id,
        "run_id": description.run_id,
        "start_time": description.start_time,
        "close_time": description.close_time,
        "execution_time": description.execution_time,
    }


async def get_extraction_result(
    abstract_id: str,
    client: Optional[Client] = None,
    workflow_id: Optional[str] = None,
    timeout: Optional[timedelta] = None,
) -> AbstractExtractionOutput:
    """Get the result of a completed or running extraction workflow.
    
    Waits for workflow completion if still running.
    
    Args:
        abstract_id: The abstract identifier
        client: Optional pre-existing Temporal client
        workflow_id: Optional custom workflow ID
        timeout: Optional timeout for waiting (default: workflow execution timeout)
        
    Returns:
        AbstractExtractionOutput with extraction results
        
    Raises:
        asyncio.TimeoutError: If timeout exceeded
        WorkflowFailureError: If workflow failed
        
    Example:
        >>> # Start workflow earlier
        >>> handle = await start_extraction("12345", "Study title...")
        >>> # Later, retrieve result
        >>> result = await get_extraction_result("12345")
    """
    if client is None:
        client = await get_client()
    
    wf_id = workflow_id or generate_workflow_id(abstract_id)
    
    handle = client.get_workflow_handle(wf_id)
    
    # Wait for result with optional timeout
    if timeout:
        result = await asyncio.wait_for(
            handle.result(),
            timeout=timeout.total_seconds(),
        )
    else:
        result = await handle.result()
    
    return result


async def query_extraction_step(
    abstract_id: str,
    client: Optional[Client] = None,
    workflow_id: Optional[str] = None,
) -> str:
    """Query the current processing step of an extraction workflow.
    
    Uses the workflow's current_step query to get real-time status.
    
    Args:
        abstract_id: The abstract identifier
        client: Optional pre-existing Temporal client
        workflow_id: Optional custom workflow ID
        
    Returns:
        Current step name (e.g., "drug_extraction", "indication_extraction")
        
    Example:
        >>> step = await query_extraction_step("12345")
        >>> print(f"Currently processing: {step}")
    """
    if client is None:
        client = await get_client()
    
    wf_id = workflow_id or generate_workflow_id(abstract_id)
    
    handle = client.get_workflow_handle(wf_id)
    
    # Query the workflow's current step
    current_step = await handle.query(AbstractExtractionWorkflow.current_step)
    
    return current_step


async def cancel_extraction(
    abstract_id: str,
    client: Optional[Client] = None,
    workflow_id: Optional[str] = None,
) -> None:
    """Cancel a running extraction workflow.
    
    Args:
        abstract_id: The abstract identifier
        client: Optional pre-existing Temporal client
        workflow_id: Optional custom workflow ID
        
    Example:
        >>> await cancel_extraction("12345")
        >>> print("Workflow cancelled")
    """
    if client is None:
        client = await get_client()
    
    wf_id = workflow_id or generate_workflow_id(abstract_id)
    
    handle = client.get_workflow_handle(wf_id)
    
    logger.info(f"Cancelling workflow {wf_id}")
    await handle.cancel()
    logger.info(f"Workflow {wf_id} cancelled")
