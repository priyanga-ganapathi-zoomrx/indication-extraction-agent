#!/usr/bin/env python3
"""
Drug Class Extraction Processor (Step-Centric)

Processes multiple abstracts through the drug class extraction pipeline
using step-centric batching for optimal parallelism control.

Key features:
- Step-centric: All abstracts go through Step 1, then Step 2, etc.
- Per-step parallelism: Configure different parallelism per step based on token usage
- Per-abstract status tracking: Resume from where each abstract left off
- Retry-friendly: Just rerun to retry failed abstracts
- Batch-level status tracking (extraction_batch_status.json)
- Real-time progress monitoring with tqdm
- Execution time tracking (accumulates across retries)
- Support for both local and GCS storage

Usage:
    # Local storage
    python -m src.scripts.drug_class.extraction_processor --input data/ASCO2025/input/abstract_titles.csv --output_dir data/ASCO2025/drug_class
    
    # GCS storage
    python -m src.scripts.drug_class.extraction_processor --input gs://bucket/ASCO2025/input/abstract_titles.csv --output_dir gs://bucket/ASCO2025/drug_class
"""

import argparse
import csv
import io
import json
import time
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm

from src.agents.drug_class import (
    # Step functions
    identify_regimen,
    fetch_search_results,
    extract_with_tavily,
    extract_with_grounded,
    select_drug_class,
    needs_llm_selection,
    extract_explicit_classes,
    consolidate_drug_classes,
    # Schemas
    DrugClassInput,
    RegimenInput,
    DrugClassExtractionInput,
    SelectionInput,
    ExplicitExtractionInput,
    ConsolidationInput,
    Step1Output,
    Step2Output,
    Step3Output,
    Step4Output,
    PipelineStatus,
)
from src.agents.drug import ExtractionResult
from src.agents.core.storage import LocalStorageClient, GCSStorageClient, get_storage_client


def _get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class StepConfig:
    """Configuration for a pipeline step."""
    name: str
    parallelism: int = 10
    description: str = ""


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    step1: StepConfig = field(default_factory=lambda: StepConfig("step1_regimen", 50, "Regimen ID"))
    step2: StepConfig = field(default_factory=lambda: StepConfig("step2_extraction", 10, "Drug class extraction"))
    step3: StepConfig = field(default_factory=lambda: StepConfig("step3_selection", 30, "Class selection"))
    step4: StepConfig = field(default_factory=lambda: StepConfig("step4_explicit", 25, "Explicit extraction"))
    step5: StepConfig = field(default_factory=lambda: StepConfig("step5_consolidation", 30, "Consolidation"))


def load_abstracts(
    csv_filename: str,
    input_storage: Union[LocalStorageClient, GCSStorageClient],
    drug_storage: Union[LocalStorageClient, GCSStorageClient],
    limit: int = None
) -> tuple[list[DrugClassInput], list[dict], list[str]]:
    """Load abstracts from CSV and drug extraction results from storage.
    
    CSV provides: abstract_id, abstract_title, firm, full_abstract
    Drug data comes from: drug_storage/abstracts/{abstract_id}/extraction.json
    
    Args:
        csv_filename: Filename of the CSV within input_storage
        input_storage: Storage client for reading input CSV
        drug_storage: Storage client for reading drug extraction results
        limit: Optional limit on rows to process
    
    Returns:
        tuple: (inputs, original_rows, fieldnames)
    """
    inputs = []
    original_rows = []
    fieldnames = []
    skipped_no_drugs = []
    
    # Read CSV content via storage client
    csv_content = input_storage.download_text(csv_filename)
    reader = csv.DictReader(io.StringIO(csv_content))
    fieldnames = list(reader.fieldnames or [])
    
    # Find column names (case-insensitive)
    header_map = {h.lower().strip(): h for h in fieldnames}
    id_col = header_map.get('abstract_id') or header_map.get('id')
    title_col = header_map.get('abstract_title') or header_map.get('title')
    abstract_col = header_map.get('full_abstract') or header_map.get('abstract')
    firms_col = header_map.get('firms') or header_map.get('firm') or header_map.get('sponsor')
    
    for row in reader:
        abstract_id = row.get(id_col, "") if id_col else ""
        abstract_title = row.get(title_col, "") if title_col else ""
        full_abstract = row.get(abstract_col, "") if abstract_col else ""
        
        if not abstract_id or not abstract_title:
            continue
        
        # Read drug extraction from storage
        try:
            drug_data = drug_storage.download_json(f"abstracts/{abstract_id}/extraction.json")
            extraction = ExtractionResult(**drug_data)
        except FileNotFoundError:
            skipped_no_drugs.append(abstract_id)
            continue
        
        # Parse firms
        firms = _parse_list(row.get(firms_col, "") if firms_col else "")
        
        inputs.append(DrugClassInput(
            abstract_id=str(abstract_id),
            abstract_title=str(abstract_title),
            full_abstract=str(full_abstract),
            primary_drugs=extraction.primary_drugs,
            firms=firms,
        ))
        original_rows.append(row)
    
    if skipped_no_drugs:
        print(f"⚠ Skipped {len(skipped_no_drugs)} abstracts without drug extraction: {skipped_no_drugs[:5]}{'...' if len(skipped_no_drugs) > 5 else ''}")
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


def _parse_list(value: str) -> list[str]:
    """Parse list from JSON array or ;; separated string.
    
    Handles:
    - JSON arrays: ["item1", "item2"]
    - Double semicolon separated: "item1;;item2"
    - Falls back to comma separated for backward compatibility
    """
    if not value or not value.strip():
        return []
    
    value = value.strip()
    
    # Try JSON array first
    if value.startswith('['):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(d).strip() for d in parsed if d and str(d).strip()]
        except json.JSONDecodeError:
            pass
    
    # Use double semicolon as primary separator (for firms column)
    if ';;' in value:
        return [d.strip() for d in value.split(';;') if d.strip()]
    
    # Fall back to comma separated for backward compatibility
    if ',' in value:
        return [d.strip() for d in value.split(',') if d.strip()]
    
    # Single value
    return [value.strip()] if value.strip() else []


def save_batch_status(
    storage: Union[LocalStorageClient, GCSStorageClient],
    inputs: list[DrugClassInput],
    total_duration: float,
    started_at: str,
) -> dict:
    """Save or update extraction_batch_status.json with accumulated duration.
    
    Args:
        storage: Storage client
        inputs: List of all inputs
        total_duration: Duration of this run in seconds
        started_at: Timestamp when this run started
        
    Returns:
        The batch status dictionary
    """
    # Count statuses
    success_count = 0
    failed_count = 0
    not_processed_count = 0
    failed_ids = []
    
    for inp in inputs:
        status = get_abstract_status(inp.abstract_id, storage)
        if status is None:
            not_processed_count += 1
        elif status.pipeline_status == "success":
            success_count += 1
        else:
            failed_count += 1
            failed_ids.append(inp.abstract_id)
    
    # Load existing batch status if available (for accumulation)
    existing_duration = 0.0
    original_started_at = started_at
    try:
        existing_status = storage.download_json("extraction_batch_status.json")
        existing_duration = existing_status.get("total_duration_seconds", 0.0)
        # Preserve original started_at from first run
        original_started_at = existing_status.get("started_at", started_at)
    except FileNotFoundError:
        pass
    
    accumulated_duration = existing_duration + total_duration
    
    batch_status = {
        "pipeline": "drug_class_extraction",
        "total_abstracts": len(inputs),
        "success": success_count,
        "failed": failed_count,
        "not_processed": not_processed_count,
        "failed_ids": failed_ids,
        "total_duration_seconds": round(accumulated_duration, 2),
        "last_run_duration_seconds": round(total_duration, 2),
        "started_at": original_started_at,
        "modified_at": _get_timestamp(),
    }
    
    storage.upload_json("extraction_batch_status.json", batch_status)
    return batch_status


def save_results_csv(
    inputs: list[DrugClassInput],
    original_rows: list[dict],
    fieldnames: list[str],
    storage: Union[LocalStorageClient, GCSStorageClient],
    output_path: str,
):
    """Save extraction results to CSV with all input columns plus model_response.
    
    Reads step3_output.json (drug selections) and step5_output.json (explicit classes)
    from storage to build the model response.
    Writes CSV to storage (local or GCS).
    """
    output_fieldnames = fieldnames + ["model_response"]
    
    # Write to string buffer first
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=output_fieldnames)
    writer.writeheader()
    
    for i, inp in enumerate(inputs):
        row = dict(original_rows[i]) if i < len(original_rows) else {}
        
        # Build model_response from step outputs
        model_response = {}
        
        # Get step3 output (drug selections)
        try:
            step3_data = storage.download_json(f"abstracts/{inp.abstract_id}/step3_output.json")
            model_response["drug_selections"] = step3_data.get("selections", {})
        except FileNotFoundError:
            model_response["drug_selections"] = {}
        
        # Get step5 output (explicit classes and consolidation)
        try:
            step5_data = storage.download_json(f"abstracts/{inp.abstract_id}/step5_output.json")
            model_response["explicit_drug_classes"] = step5_data.get("explicit_drug_classes", [])
            model_response["final_drug_classes"] = step5_data.get("final_drug_classes", [])
        except FileNotFoundError:
            model_response["explicit_drug_classes"] = []
            model_response["final_drug_classes"] = []
        
        # Get status for metadata
        try:
            status_data = storage.download_json(f"abstracts/{inp.abstract_id}/status.json")
            model_response["pipeline_status"] = status_data.get("pipeline_status", "unknown")
            model_response["total_llm_calls"] = status_data.get("total_llm_calls", 0)
        except FileNotFoundError:
            model_response["pipeline_status"] = "not_processed"
            model_response["total_llm_calls"] = 0
        
        row["model_response"] = json.dumps(model_response, indent=2)
        writer.writerow(row)
    
    # Upload CSV content to storage
    storage.upload_text(output_path, output.getvalue())


def get_abstract_status(abstract_id: str, storage: Union[LocalStorageClient, GCSStorageClient]) -> Optional[PipelineStatus]:
    """Load status for an abstract if it exists."""
    try:
        status_data = storage.download_json(f"abstracts/{abstract_id}/status.json")
        return PipelineStatus(**status_data)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def get_step_from_status(status: Optional[PipelineStatus]) -> str:
    """Determine which step an abstract needs based on its status.
    
    Note: "skipped" status is treated as complete (used when no drugs to process).
    """
    if not status:
        return "step1_regimen"
    
    steps_order = ["step1_regimen", "step2_extraction", "step3_selection", "step4_explicit", "step5_consolidation"]
    
    for step in steps_order:
        step_data = status.steps.get(step, {})
        step_status = step_data.get("status", "pending")
        
        # "skipped" is treated as complete (e.g., when no drugs to process)
        if step_status in ["pending", "running", "failed"]:
            return step
    
    # All steps complete
    return "complete"


def get_abstracts_at_step(
    inputs: list[DrugClassInput],
    step_name: str,
    storage: Union[LocalStorageClient, GCSStorageClient]
) -> list[DrugClassInput]:
    """Get abstracts that need a specific step."""
    pending = []
    
    for inp in inputs:
        status = get_abstract_status(inp.abstract_id, storage)
        current_step = get_step_from_status(status)
        
        if current_step == step_name:
            pending.append(inp)
    
    return pending


def process_step1_single(inp: DrugClassInput, storage: Union[LocalStorageClient, GCSStorageClient]) -> dict:
    """Process Step 1 for a single abstract.
    
    Uses per-drug error handling to preserve partial progress.
    """
    abstract_id = inp.abstract_id
    
    # Initialize or load status
    status = get_abstract_status(abstract_id, storage)
    if not status:
        status = PipelineStatus(abstract_id=abstract_id, abstract_title=inp.abstract_title)
        status.last_updated = _get_timestamp()
    
    # Get primary drugs only (drug class extraction is limited to primary drugs)
    all_drugs = inp.primary_drugs
    if not all_drugs:
        # No drugs - skip steps 1-3 but allow step 4 (explicit extraction) to run
        # Step 4 can extract drug classes mentioned directly in the abstract title
        status.steps["step1_regimen"] = {"status": "skipped", "llm_calls": 0, "reason": "No drugs to process"}
        status.steps["step2_extraction"] = {"status": "skipped", "llm_calls": 0, "reason": "No drugs to process"}
        status.steps["step3_selection"] = {"status": "skipped", "llm_calls": 0, "reason": "No drugs to process"}
        status.last_completed_step = "step3_selection"  # Advance to step 4
        status.last_updated = _get_timestamp()
        
        # Save empty outputs for steps 1-3 so downstream steps can load them
        storage.upload_json(f"abstracts/{abstract_id}/step1_output.json", Step1Output().model_dump())
        storage.upload_json(f"abstracts/{abstract_id}/step2_output.json", Step2Output().model_dump())
        storage.upload_json(f"abstracts/{abstract_id}/step3_output.json", Step3Output().model_dump())
        storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
        
        return {"abstract_id": abstract_id, "success": True, "llm_calls": 0, "skipped": True, "reason": "No drugs - skipping to step 4"}
    
    # Load existing step1 output if available, or create new
    try:
        existing_data = storage.download_json(f"abstracts/{abstract_id}/step1_output.json")
        step1_output = Step1Output(**existing_data)
    except FileNotFoundError:
        step1_output = Step1Output()
    
    step1_llm_calls = 0
    
    for drug in all_drugs:
        if step1_output.is_drug_done(drug):
            continue
        
        try:
            components = identify_regimen(RegimenInput(
                abstract_id=abstract_id,
                abstract_title=inp.abstract_title,
                drug=drug,
            ))
            step1_output.mark_success(drug, components)
            step1_llm_calls += 1
        except Exception as e:
            # Mark individual drug as failed, continue to others
            step1_output.mark_failed(drug, str(e))
        
        # Save progress after each drug (incremental checkpoint)
        storage.upload_json(f"abstracts/{abstract_id}/step1_output.json", step1_output.model_dump())
    
    # Update status based on completion
    if step1_output.is_complete(all_drugs):
        status.steps["step1_regimen"] = {
            "status": "success",
            "llm_calls": step1_llm_calls,
        }
        status.last_completed_step = "step1_regimen"
    else:
        failed_drugs = [d for d in all_drugs if step1_output.drug_status.get(d) == "failed"]
        status.steps["step1_regimen"] = {"status": "failed", "error": f"Failed drugs: {failed_drugs}"}
    
    status.total_llm_calls += step1_llm_calls
    status.last_updated = _get_timestamp()
    storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
    
    success_count = sum(1 for d in all_drugs if step1_output.drug_status.get(d) == "success")
    return {
        "abstract_id": abstract_id,
        "success": step1_output.is_complete(all_drugs),
        "llm_calls": step1_llm_calls,
        "drugs_succeeded": success_count,
        "drugs_failed": len(all_drugs) - success_count,
    }


def process_step2_single(inp: DrugClassInput, storage: Union[LocalStorageClient, GCSStorageClient]) -> dict:
    """Process Step 2 for a single abstract.
    
    Uses per-drug error handling to preserve partial progress.
    """
    abstract_id = inp.abstract_id
    
    # Load status and step1 output
    status = get_abstract_status(abstract_id, storage)
    if not status:
        return {"abstract_id": abstract_id, "success": False, "error": "No status found"}
    
    try:
        step1_data = storage.download_json(f"abstracts/{abstract_id}/step1_output.json")
        step1_output = Step1Output(**step1_data)
    except FileNotFoundError:
        return {"abstract_id": abstract_id, "success": False, "error": "Step 1 output not found"}
    
    all_components = step1_output.get_all_components()
    
    if not all_components:
        return {"abstract_id": abstract_id, "success": False, "error": "No components from Step 1"}
    
    # Load existing step2 output if available, or create new
    try:
        existing_data = storage.download_json(f"abstracts/{abstract_id}/step2_output.json")
        step2_output = Step2Output(**existing_data)
    except FileNotFoundError:
        step2_output = Step2Output()
    
    step2_llm_calls = 0
    
    for drug in all_components:
        if step2_output.is_drug_done(drug):
            continue
        
        try:
            # Fetch search results (with caching)
            drug_results, firm_results = fetch_search_results(drug, inp.firms, storage)
            
            # Try Tavily extraction first
            result = extract_with_tavily(DrugClassExtractionInput(
                abstract_id=abstract_id,
                abstract_title=inp.abstract_title,
                drug=drug,
                full_abstract=inp.full_abstract,
                firms=inp.firms,
                drug_class_results=drug_results,
                firm_search_results=firm_results,
            ))
            step2_llm_calls += 1
            
            # Fallback to grounded search if needed
            if not result.drug_classes or result.drug_classes == ["NA"]:
                result = extract_with_grounded(DrugClassExtractionInput(
                    abstract_id=abstract_id,
                    abstract_title=inp.abstract_title,
                    drug=drug,
                    full_abstract=inp.full_abstract,
                    firms=inp.firms,
                    drug_class_results=drug_results,
                    firm_search_results=firm_results,
                ))
                step2_llm_calls += 1
            
            step2_output.mark_success(drug, result)
            
        except Exception as e:
            # Mark individual drug as failed, continue to others
            step2_output.mark_failed(drug, str(e))
        
        # Save progress after each drug (incremental checkpoint)
        storage.upload_json(f"abstracts/{abstract_id}/step2_output.json", step2_output.model_dump())
    
    # Update status based on completion
    if step2_output.is_complete(all_components):
        status.steps["step2_extraction"] = {
            "status": "success",
            "llm_calls": step2_llm_calls,
        }
        status.last_completed_step = "step2_extraction"
    else:
        failed_drugs = [d for d in all_components if step2_output.drug_status.get(d) == "failed"]
        status.steps["step2_extraction"] = {"status": "failed", "error": f"Failed drugs: {failed_drugs}"}
    
    status.total_llm_calls += step2_llm_calls
    status.last_updated = _get_timestamp()
    storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
    
    success_count = sum(1 for d in all_components if step2_output.drug_status.get(d) == "success")
    return {
        "abstract_id": abstract_id,
        "success": step2_output.is_complete(all_components),
        "llm_calls": step2_llm_calls,
        "drugs_succeeded": success_count,
        "drugs_failed": len(all_components) - success_count,
    }


def process_step3_single(inp: DrugClassInput, storage: Union[LocalStorageClient, GCSStorageClient]) -> dict:
    """Process Step 3 for a single abstract.
    
    Uses per-drug error handling to preserve partial progress.
    """
    abstract_id = inp.abstract_id
    
    status = get_abstract_status(abstract_id, storage)
    if not status:
        return {"abstract_id": abstract_id, "success": False, "error": "No status found"}
    
    try:
        step2_data = storage.download_json(f"abstracts/{abstract_id}/step2_output.json")
        step2_output = Step2Output(**step2_data)
    except FileNotFoundError:
        return {"abstract_id": abstract_id, "success": False, "error": "Step 2 output not found"}
    
    # Load existing step3 output if available, or create new
    try:
        existing_data = storage.download_json(f"abstracts/{abstract_id}/step3_output.json")
        step3_output = Step3Output(**existing_data)
    except FileNotFoundError:
        step3_output = Step3Output()
    
    step3_llm_calls = 0
    all_drugs = list(step2_output.extractions.keys())
    
    for drug_name, extraction_result in step2_output.extractions.items():
        if step3_output.is_drug_done(drug_name):
            continue
        
        try:
            # Check if LLM selection is needed
            if not needs_llm_selection(extraction_result.extraction_details):
                # No LLM needed - just copy the single class
                from src.agents.drug_class.schemas import DrugSelectionResult
                if extraction_result.extraction_details:
                    first_detail = extraction_result.extraction_details[0]
                    selected = [first_detail.normalized_form or first_detail.extracted_text or "NA"]
                else:
                    selected = extraction_result.drug_classes or ["NA"]
                
                result = DrugSelectionResult(
                    drug_name=drug_name,
                    selected_drug_classes=selected,
                    reasoning="Single class - no selection needed",
                )
            else:
                result = select_drug_class(SelectionInput(
                    abstract_id=abstract_id,
                    drug_name=drug_name,
                    extraction_details=extraction_result.extraction_details,
                ))
                step3_llm_calls += 1
            
            step3_output.mark_success(drug_name, result)
            
        except Exception as e:
            # Mark individual drug as failed, continue to others
            step3_output.mark_failed(drug_name, str(e))
        
        # Save progress after each drug (incremental checkpoint)
        storage.upload_json(f"abstracts/{abstract_id}/step3_output.json", step3_output.model_dump())
    
    # Update status based on completion
    if step3_output.is_complete(all_drugs):
        status.steps["step3_selection"] = {
            "status": "success",
            "llm_calls": step3_llm_calls,
        }
        status.last_completed_step = "step3_selection"
    else:
        failed_drugs = [d for d in all_drugs if step3_output.drug_status.get(d) == "failed"]
        status.steps["step3_selection"] = {"status": "failed", "error": f"Failed drugs: {failed_drugs}"}
    
    status.total_llm_calls += step3_llm_calls
    status.last_updated = _get_timestamp()
    storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
    
    success_count = sum(1 for d in all_drugs if step3_output.drug_status.get(d) == "success")
    return {
        "abstract_id": abstract_id,
        "success": step3_output.is_complete(all_drugs),
        "llm_calls": step3_llm_calls,
        "drugs_succeeded": success_count,
        "drugs_failed": len(all_drugs) - success_count,
    }


def process_step4_single(inp: DrugClassInput, storage: Union[LocalStorageClient, GCSStorageClient]) -> dict:
    """Process Step 4 for a single abstract."""
    abstract_id = inp.abstract_id
    
    status = get_abstract_status(abstract_id, storage)
    if not status:
        return {"abstract_id": abstract_id, "success": False, "error": "No status found"}
    
    step4_llm_calls = 0
    
    try:
        step4_output = extract_explicit_classes(ExplicitExtractionInput(
            abstract_id=abstract_id,
            abstract_title=inp.abstract_title,
        ))
        
        # Count LLM call only if title was non-empty
        if inp.abstract_title and inp.abstract_title.strip():
            step4_llm_calls = 1
        
        status.steps["step4_explicit"] = {
            "status": "success",
            "llm_calls": step4_llm_calls,
        }
        status.last_completed_step = "step4_explicit"
        status.total_llm_calls += step4_llm_calls
        status.last_updated = _get_timestamp()
        
        storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
        storage.upload_json(f"abstracts/{abstract_id}/step4_output.json", step4_output.model_dump())
        
        return {"abstract_id": abstract_id, "success": True, "llm_calls": step4_llm_calls}
        
    except Exception as e:
        status.steps["step4_explicit"] = {"status": "failed", "error": str(e)}
        status.last_updated = _get_timestamp()
        storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
        return {"abstract_id": abstract_id, "success": False, "error": str(e)}


def process_step5_single(inp: DrugClassInput, storage: Union[LocalStorageClient, GCSStorageClient]) -> dict:
    """Process Step 5 for a single abstract.
    
    Handles the case where steps 1-3 were skipped (no drugs) - in that case,
    only explicit drug classes from step 4 are used.
    """
    abstract_id = inp.abstract_id
    
    status = get_abstract_status(abstract_id, storage)
    if not status:
        return {"abstract_id": abstract_id, "success": False, "error": "No status found"}
    
    # Load step4 output (required)
    try:
        step4_data = storage.download_json(f"abstracts/{abstract_id}/step4_output.json")
        step4_output = Step4Output(**step4_data)
    except FileNotFoundError:
        return {"abstract_id": abstract_id, "success": False, "error": "Step 4 output not found"}
    
    # Load step3 output (may be empty if no drugs / steps 1-3 were skipped)
    try:
        step3_data = storage.download_json(f"abstracts/{abstract_id}/step3_output.json")
        step3_output = Step3Output(**step3_data)
    except FileNotFoundError:
        # If step3 output doesn't exist, use empty output
        step3_output = Step3Output()
    
    step5_llm_calls = 0
    
    try:
        # Build drug selections for consolidation
        drug_selections = [
            {"drug_name": s.drug_name, "selected_classes": s.selected_drug_classes}
            for s in step3_output.get_results_list()
        ]
        
        step5_output = consolidate_drug_classes(ConsolidationInput(
            abstract_id=abstract_id,
            abstract_title=inp.abstract_title,
            explicit_drug_classes=step4_output.explicit_drug_classes,
            drug_selections=drug_selections,
        ))
        
        # Count LLM call only if consolidation was needed
        has_explicit = step4_output.explicit_drug_classes and step4_output.explicit_drug_classes != ["NA"]
        has_selections = bool(drug_selections)
        if has_explicit and has_selections:
            step5_llm_calls = 1
        
        status.steps["step5_consolidation"] = {
            "status": "success",
            "llm_calls": step5_llm_calls,
        }
        status.last_completed_step = "step5_consolidation"
        status.pipeline_status = "success"
        status.total_llm_calls += step5_llm_calls
        status.last_updated = _get_timestamp()
        
        storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
        storage.upload_json(f"abstracts/{abstract_id}/step5_output.json", step5_output.model_dump())
        
        return {"abstract_id": abstract_id, "success": True, "llm_calls": step5_llm_calls}
        
    except Exception as e:
        status.steps["step5_consolidation"] = {"status": "failed", "error": str(e)}
        status.last_updated = _get_timestamp()
        storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
        return {"abstract_id": abstract_id, "success": False, "error": str(e)}


def run_step_batch(
    step_name: str,
    process_func,
    inputs: list[DrugClassInput],
    storage: Union[LocalStorageClient, GCSStorageClient],
    parallelism: int,
) -> tuple[int, int, float]:
    """Run a step for all pending abstracts.
    
    Returns:
        tuple: (successful_count, failed_count, duration_seconds)
    """
    pending = get_abstracts_at_step(inputs, step_name, storage)
    
    if not pending:
        print(f"  No abstracts pending for {step_name}")
        return 0, 0, 0.0
    
    print(f"  Processing {len(pending)} abstracts (parallelism: {parallelism})")
    
    step_start = time.time()
    successful = 0
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_to_inp = {
            executor.submit(process_func, inp, storage): inp
            for inp in pending
        }
        
        # Use tqdm for progress bar
        with tqdm(
            total=len(pending),
            desc=f"  {step_name}",
            unit="abstract",
            leave=True,
        ) as pbar:
            for future in concurrent.futures.as_completed(future_to_inp):
                inp = future_to_inp[future]
                try:
                    result = future.result()
                    if result.get("success"):
                        successful += 1
                        pbar.set_postfix_str(f"✓ {inp.abstract_id}")
                    else:
                        failed += 1
                        pbar.set_postfix_str(f"✗ {inp.abstract_id}")
                except Exception as e:
                    failed += 1
                    pbar.set_postfix_str(f"✗ {inp.abstract_id}: {str(e)[:30]}")
                pbar.update(1)
    
    step_duration = time.time() - step_start
    return successful, failed, step_duration


def main():
    parser = argparse.ArgumentParser(description="Drug Class Extraction Processor (Step-Centric)")
    parser.add_argument("--input", default="gs://entity-extraction-agent-data-dev/Conference/abstract_titles.csv", 
                        help="Input CSV file (local or gs://bucket/path)")
    parser.add_argument("--drug_output_dir", default="gs://entity-extraction-agent-data-dev/Conference/drug", 
                        help="Directory containing drug extraction results (local or gs://bucket/path)")
    parser.add_argument("--output_dir", default="gs://entity-extraction-agent-data-dev/Conference/drug_class", 
                        help="Output directory for results (local or gs://bucket/path)")
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts to process")
    
    # Per-step parallelism
    parser.add_argument("--step1_parallelism", type=int, default=40, help="Parallelism for Step 1")
    parser.add_argument("--step2_parallelism", type=int, default=40, help="Parallelism for Step 2 (expensive)")
    parser.add_argument("--step3_parallelism", type=int, default=40, help="Parallelism for Step 3")
    parser.add_argument("--step4_parallelism", type=int, default=40, help="Parallelism for Step 4")
    parser.add_argument("--step5_parallelism", type=int, default=40, help="Parallelism for Step 5")
    
    args = parser.parse_args()
    
    # Determine if using GCS or local storage based on output_dir
    is_gcs = args.output_dir.startswith("gs://")
    
    # Initialize storage clients
    drug_storage = get_storage_client(args.drug_output_dir)
    storage = get_storage_client(args.output_dir)
    
    # For input, we need to handle path differently
    if args.input.startswith("gs://"):
        # Parse GCS input path
        from src.agents.core.storage import parse_gcs_path
        bucket, input_prefix = parse_gcs_path(args.input)
        # Split prefix into directory and filename
        if "/" in input_prefix:
            input_dir = input_prefix.rsplit("/", 1)[0]
            input_filename = input_prefix.rsplit("/", 1)[1]
        else:
            input_dir = ""
            input_filename = input_prefix
        input_storage = get_storage_client(f"gs://{bucket}/{input_dir}" if input_dir else f"gs://{bucket}")
    else:
        # Local input - use parent directory as base
        input_path = Path(args.input)
        input_dir = str(input_path.parent)
        input_filename = input_path.name
        input_storage = get_storage_client(input_dir)
    
    # For local storage, ensure output directory exists
    if not is_gcs:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Record batch start time
    batch_started_at = _get_timestamp()
    batch_start_time = time.time()
    
    print("🧬 Drug Class Extraction Processor (Step-Centric)")
    print("=" * 60)
    print(f"Input:          {args.input}")
    print(f"Drug data from: {args.drug_output_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Storage:        {'GCS' if is_gcs else 'Local'}")
    print(f"Limit:          {args.limit or 'all'}")
    print()
    print("Per-step parallelism:")
    print(f"  Step 1 (Regimen):      {args.step1_parallelism}")
    print(f"  Step 2 (Extraction):   {args.step2_parallelism}")
    print(f"  Step 3 (Selection):    {args.step3_parallelism}")
    print(f"  Step 4 (Explicit):     {args.step4_parallelism}")
    print(f"  Step 5 (Consolidation):{args.step5_parallelism}")
    print()
    
    # Load abstracts (reads drug data from storage)
    print("Loading abstracts...")
    inputs, original_rows, fieldnames = load_abstracts(input_filename, input_storage, drug_storage, args.limit)
    print(f"✓ Loaded {len(inputs)} abstracts with drug extraction data")
    print()
    
    # Process step by step
    steps = [
        ("step1_regimen", process_step1_single, args.step1_parallelism),
        ("step2_extraction", process_step2_single, args.step2_parallelism),
        ("step3_selection", process_step3_single, args.step3_parallelism),
        ("step4_explicit", process_step4_single, args.step4_parallelism),
        ("step5_consolidation", process_step5_single, args.step5_parallelism),
    ]
    
    total_success = 0
    total_failed = 0
    step_durations = {}
    
    for step_name, process_func, parallelism in steps:
        print(f"→ {step_name.upper()}")
        success, failed, duration = run_step_batch(step_name, process_func, inputs, storage, parallelism)
        total_success += success
        total_failed += failed
        step_durations[step_name] = duration
        print(f"  Done: {success} success, {failed} failed ({duration:.2f}s)")
        print()
    
    # Calculate total batch duration
    batch_duration = time.time() - batch_start_time
    
    # Save batch status (with accumulation)
    print("Saving batch status...")
    batch_status = save_batch_status(storage, inputs, batch_duration, batch_started_at)
    print(f"✓ Saved extraction_batch_status.json")
    
    # Save CSV output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_output_filename = f"extraction_{timestamp}.csv"
    print(f"Saving CSV output...")
    save_results_csv(inputs, original_rows, fieldnames, storage, csv_output_filename)
    print(f"✓ Saved {csv_output_filename}")
    
    # Summary
    print()
    print("=" * 60)
    print("📊 Summary:")
    
    print(f"   Total abstracts: {len(inputs)}")
    print(f"   Complete:        {batch_status['success']}")
    print(f"   Failed:          {batch_status['failed']}")
    print(f"   Not processed:   {batch_status['not_processed']}")
    print()
    print("Per-step duration:")
    for step_name, duration in step_durations.items():
        print(f"   {step_name}: {duration:.2f}s")
    print()
    print(f"   This run:        {batch_duration:.2f}s")
    print(f"   Total duration:  {batch_status['total_duration_seconds']:.2f}s")
    print(f"   Output:          {args.output_dir}")


if __name__ == "__main__":
    main()
