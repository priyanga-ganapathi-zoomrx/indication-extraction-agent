"""Checkpoint activities for workflow state persistence.

These activities handle loading and saving workflow state to external storage
(GCS or local filesystem). They enable:
- Resuming workflows from checkpoints
- Skipping already-completed steps
- Debugging via JSON inspection

Activities are pure storage operations with no business logic.
"""

from temporalio import activity

from src.agents.core.storage import get_storage_client
from src.temporal.idle_shutdown import track_activity


@activity.defn(name="load_workflow_status")
@track_activity
def load_workflow_status(storage_path: str, abstract_id: str) -> dict | None:
    """Load workflow status from storage.
    
    Args:
        storage_path: Base storage path (gs://bucket/prefix or local path)
        abstract_id: The abstract ID
        
    Returns:
        Status dict if found, None otherwise
    """
    if not storage_path:
        return None
    
    try:
        storage = get_storage_client(storage_path)
        return storage.download_json(f"abstracts/{abstract_id}/status.json")
    except FileNotFoundError:
        return None
    except Exception as e:
        activity.logger.warning(f"Error loading status for {abstract_id}: {e}")
        return None


@activity.defn(name="save_workflow_status")
@track_activity
def save_workflow_status(storage_path: str, abstract_id: str, status: dict) -> None:
    """Save workflow status to storage.
    
    Args:
        storage_path: Base storage path (gs://bucket/prefix or local path)
        abstract_id: The abstract ID
        status: Status dict to save
    """
    if not storage_path:
        activity.logger.info("No storage path provided, skipping status save")
        return
    
    storage = get_storage_client(storage_path)
    storage.upload_json(f"abstracts/{abstract_id}/status.json", status)
    activity.logger.info(f"Saved status for {abstract_id}")


@activity.defn(name="load_step_output")
@track_activity
def load_step_output(storage_path: str, abstract_id: str, step_name: str) -> dict | None:
    """Load step output from storage.
    
    Used to check if a step has already been completed (checkpoint exists).
    
    Args:
        storage_path: Base storage path (gs://bucket/prefix or local path)
        abstract_id: The abstract ID
        step_name: Name of the step (e.g., "drug_extraction", "drug_validation")
        
    Returns:
        Step output dict if found, None otherwise
    """
    if not storage_path:
        return None
    
    try:
        storage = get_storage_client(storage_path)
        return storage.download_json(f"abstracts/{abstract_id}/{step_name}.json")
    except FileNotFoundError:
        return None
    except Exception as e:
        activity.logger.warning(f"Error loading {step_name} for {abstract_id}: {e}")
        return None


@activity.defn(name="save_step_output")
@track_activity
def save_step_output(storage_path: str, abstract_id: str, step_name: str, data: dict) -> None:
    """Save step output to storage.
    
    Creates checkpoint for a completed step.
    
    Args:
        storage_path: Base storage path (gs://bucket/prefix or local path)
        abstract_id: The abstract ID
        step_name: Name of the step (e.g., "drug_extraction", "drug_validation")
        data: Step output dict to save
    """
    if not storage_path:
        activity.logger.info(f"No storage path provided, skipping {step_name} save")
        return
    
    storage = get_storage_client(storage_path)
    storage.upload_json(f"abstracts/{abstract_id}/{step_name}.json", data)
    activity.logger.info(f"Saved {step_name} for {abstract_id}")
