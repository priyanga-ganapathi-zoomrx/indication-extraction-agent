#!/usr/bin/env python3
"""
Validation Processor for Drug Extraction

Validates previously extracted drugs with:
- Parallel processing for improved throughput
- Per-abstract status tracking (updates status.json)
- Batch-level status summary (updates batch_status.json)
- Retry logic for failed validations
- Real-time progress monitoring with tqdm
- Support for both local and GCS storage

Usage:
    # Local storage
    python -m src.scripts.drug.validation_processor --input data/ASCO2025/input/abstract_titles.csv --output_dir data/ASCO2025/drug
    
    # GCS storage
    python -m src.scripts.drug.validation_processor --input gs://bucket/ASCO2025/input/abstract_titles.csv --output_dir gs://bucket/ASCO2025/drug
"""

import argparse
import csv
import io
import json
import time
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm

from src.agents.drug import validate_drugs, ValidationInput as AgentValidationInput, ValidationResult, DrugValidationError, config
from src.agents.core.storage import LocalStorageClient, GCSStorageClient, get_storage_client


def _get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class ProcessResult:
    """Result of validating a single extraction."""
    abstract_id: str
    abstract_title: str = ""
    response_json: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0
    
    @property
    def success(self) -> bool:
        return bool(self.response_json and not self.error)


@dataclass
class ValidationInput:
    """Input for validation processor."""
    abstract_id: str
    abstract_title: str
    extraction_result: dict


def load_abstracts_for_validation(
    csv_filename: str,
    input_storage: Union[LocalStorageClient, GCSStorageClient],
    output_storage: Union[LocalStorageClient, GCSStorageClient],
    limit: int = None,
) -> tuple[list[ValidationInput], list[dict], list[str]]:
    """Load abstracts from CSV and their extraction results from storage.
    
    Only returns abstracts that have successful extraction results.
    
    Args:
        csv_filename: Filename of the CSV within input_storage
        input_storage: Storage client for reading input CSV
        output_storage: Storage client for reading extraction results
        limit: Optional limit on rows to process
    
    Returns:
        tuple: (ValidationInput objects, original_rows, fieldnames)
    """
    inputs = []
    original_rows = []
    skipped_no_extraction = []
    fieldnames = []
    
    # Read CSV content via storage client
    csv_content = input_storage.download_text(csv_filename)
    reader = csv.DictReader(io.StringIO(csv_content))
    fieldnames = list(reader.fieldnames or [])
    
    # Find column names (case-insensitive)
    header_map = {h.lower().strip(): h for h in fieldnames}
    id_col = header_map.get('abstract_id') or header_map.get('id')
    title_col = header_map.get('abstract_title') or header_map.get('title')
    
    for row in reader:
        abstract_id = row.get(id_col, "") if id_col else ""
        abstract_title = row.get(title_col, "") if title_col else ""
        
        if not abstract_id:
            continue
        
        # Try to load extraction result from output storage
        try:
            extraction_result = output_storage.download_json(f"abstracts/{abstract_id}/extraction.json")
        except FileNotFoundError:
            skipped_no_extraction.append(abstract_id)
            continue
        
        # Skip if extraction had error
        if extraction_result.get("error"):
            skipped_no_extraction.append(abstract_id)
            continue
        
        inputs.append(ValidationInput(
            abstract_id=str(abstract_id),
            abstract_title=str(abstract_title),
            extraction_result=extraction_result,
        ))
        original_rows.append(row)
    
    if skipped_no_extraction:
        print(f"⚠ Skipped {len(skipped_no_extraction)} abstracts without extraction: {skipped_no_extraction[:5]}{'...' if len(skipped_no_extraction) > 5 else ''}")
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


def get_abstract_status(
    abstract_id: str,
    storage: Union[LocalStorageClient, GCSStorageClient],
) -> Optional[dict]:
    """Load status for an abstract if it exists."""
    try:
        return storage.download_json(f"abstracts/{abstract_id}/status.json")
    except FileNotFoundError:
        return None
    except Exception:
        return None


def should_process_validation(
    abstract_id: str,
    storage: Union[LocalStorageClient, GCSStorageClient],
) -> bool:
    """Check if abstract needs validation (extraction successful but validation not done)."""
    status = get_abstract_status(abstract_id, storage)
    if not status:
        return False  # No status means extraction wasn't done
    
    extraction = status.get("extraction", {})
    validation = status.get("validation", {})
    
    # Only validate if extraction was successful and validation not already successful
    return extraction.get("status") == "success" and validation.get("status") != "success"


def process_single(
    input_data: ValidationInput,
    storage: Union[LocalStorageClient, GCSStorageClient],
) -> ProcessResult:
    """Validate single extraction and return result with timing."""
    start_time = time.time()
    
    # Skip if extraction had error
    if input_data.extraction_result.get("error"):
        duration = time.time() - start_time
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            abstract_title=input_data.abstract_title,
            response_json=json.dumps({
                "validation_status": "SKIP",
                "validation_reasoning": f"Extraction had error: {input_data.extraction_result.get('error')}"
            }, indent=2),
            duration_seconds=duration,
        )
    
    try:
        # Convert to agent's ValidationInput
        agent_input = AgentValidationInput(
            abstract_id=input_data.abstract_id,
            abstract_title=input_data.abstract_title,
            extraction_result=input_data.extraction_result,
        )
        
        result: ValidationResult = validate_drugs(agent_input)
        
        # Serialize to JSON
        response_json = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
        
        duration = time.time() - start_time
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            abstract_title=input_data.abstract_title,
            response_json=response_json,
            duration_seconds=duration,
        )
        
    except DrugValidationError as e:
        duration = time.time() - start_time
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            abstract_title=input_data.abstract_title,
            error=str(e),
            duration_seconds=duration,
        )
    except Exception as e:
        duration = time.time() - start_time
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            abstract_title=input_data.abstract_title,
            error=f"Unexpected error: {e}",
            duration_seconds=duration,
        )


def save_validation_result(
    result: ProcessResult,
    storage: Union[LocalStorageClient, GCSStorageClient],
) -> None:
    """Save validation result and update status for a single abstract."""
    abstract_id = result.abstract_id
    timestamp = _get_timestamp()
    
    # Load existing status
    status = get_abstract_status(abstract_id, storage)
    if not status:
        # This shouldn't happen, but handle it
        status = {
            "abstract_id": abstract_id,
            "abstract_title": result.abstract_title,
            "extraction": {"status": "pending", "error": None, "duration_seconds": 0, "completed_at": None},
            "validation": {"status": "pending", "error": None, "duration_seconds": 0, "completed_at": None},
        }
    
    # Update validation status
    status["validation"] = {
        "status": "success" if result.success else "failed",
        "error": result.error,
        "duration_seconds": round(result.duration_seconds, 2),
        "completed_at": timestamp,
    }
    
    # Save status.json
    storage.upload_json(f"abstracts/{abstract_id}/status.json", status)
    
    # Save validation.json if successful
    if result.success:
        try:
            validation_data = json.loads(result.response_json)
            storage.upload_json(f"abstracts/{abstract_id}/validation.json", validation_data)
        except json.JSONDecodeError:
            # If JSON parsing fails, save as raw
            storage.upload_json(f"abstracts/{abstract_id}/validation.json", {"raw_response": result.response_json})


def save_results_csv(
    inputs: list[ValidationInput],
    original_rows: list[dict],
    input_fieldnames: list[str],
    storage: Union[LocalStorageClient, GCSStorageClient],
    output_path: str,
) -> None:
    """Save all validation results to CSV with input columns plus model_response column.
    
    Reads validation.json from storage for each abstract to build the CSV.
    Writes CSV to storage (local or GCS).
    """
    response_column = "validation_response"
    fieldnames = input_fieldnames + [response_column]
    
    # Write to string buffer first
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for inp, original_row in zip(inputs, original_rows):
        row = dict(original_row)
        
        # Try to load validation result from storage
        try:
            validation_data = storage.download_json(f"abstracts/{inp.abstract_id}/validation.json")
            row[response_column] = json.dumps(validation_data, indent=2, ensure_ascii=False)
        except FileNotFoundError:
            # Check if there was an error in status
            status = get_abstract_status(inp.abstract_id, storage)
            if status:
                validation = status.get("validation", {})
                if validation.get("error"):
                    row[response_column] = json.dumps({"error": validation.get("error")}, indent=2)
                else:
                    row[response_column] = json.dumps({"error": "Validation not completed"}, indent=2)
            else:
                row[response_column] = json.dumps({"error": "Validation not completed"}, indent=2)
        
        writer.writerow(row)
    
    # Upload CSV content to storage
    storage.upload_text(output_path, output.getvalue())
    print(f"✓ CSV saved to {output_path}")


def save_batch_status(
    pipeline: str,
    total_abstracts: int,
    success_count: int,
    failed_count: int,
    failed_ids: list[str],
    total_duration: float,
    started_at: str,
    storage: Union[LocalStorageClient, GCSStorageClient],
    phase: str = "validation",
) -> None:
    """Save or update batch_status.json.
    
    Duration is accumulated across retries to track total processing time.
    """
    # Try to load existing batch status
    try:
        batch_status = storage.download_json("batch_status.json")
    except FileNotFoundError:
        batch_status = {
            "pipeline": pipeline,
            "total_abstracts": total_abstracts,
        }
    
    # Get existing duration to accumulate (if any)
    existing_phase = batch_status.get(phase, {})
    existing_duration = existing_phase.get("total_duration_seconds", 0.0)
    accumulated_duration = existing_duration + total_duration
    
    # Preserve original started_at from first run (if exists)
    original_started_at = existing_phase.get("started_at") or started_at
    
    # Update the specified phase
    batch_status[phase] = {
        "success": success_count,
        "failed": failed_count,
        "not_processed": total_abstracts - success_count - failed_count,
        "failed_ids": failed_ids,
        "total_duration_seconds": round(accumulated_duration, 2),
        "started_at": original_started_at,
        "modified_at": _get_timestamp(),
        "last_run_duration_seconds": round(total_duration, 2),
    }
    
    storage.upload_json("batch_status.json", batch_status)


def run_validation_batch(
    inputs: list[ValidationInput],
    storage: Union[LocalStorageClient, GCSStorageClient],
    parallelism: int,
) -> tuple[int, int, list[str]]:
    """Run validation for all abstracts with parallel processing.
    
    Returns:
        tuple: (success_count, failed_count, failed_ids)
    """
    # Filter to only process abstracts that need validation
    pending_inputs = [inp for inp in inputs if should_process_validation(inp.abstract_id, storage)]
    
    if not pending_inputs:
        print("  All abstracts already validated successfully.")
        # Count existing results
        success_count = 0
        for inp in inputs:
            status = get_abstract_status(inp.abstract_id, storage)
            if status:
                validation = status.get("validation", {})
                if validation.get("status") == "success":
                    success_count += 1
        return success_count, 0, []
    
    already_done = len(inputs) - len(pending_inputs)
    if already_done > 0:
        print(f"  Skipping {already_done} already successful validations")
    
    print(f"  Processing {len(pending_inputs)} abstracts (parallelism: {parallelism})")
    
    success_count = 0
    failed_count = 0
    failed_ids = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {
            executor.submit(process_single, inp, storage): inp
            for inp in pending_inputs
        }
        
        with tqdm(total=len(pending_inputs), desc="Validating", unit="abstract") as pbar:
            for future in concurrent.futures.as_completed(futures):
                inp = futures[future]
                
                try:
                    result = future.result()
                    
                    # Save result immediately
                    save_validation_result(result, storage)
                    
                    if result.success:
                        success_count += 1
                    else:
                        failed_count += 1
                        failed_ids.append(inp.abstract_id)
                    
                    pbar.set_postfix({"✓": success_count, "✗": failed_count})
                    pbar.update(1)
                    
                except Exception as e:
                    failed_count += 1
                    failed_ids.append(inp.abstract_id)
                    pbar.set_postfix({"✓": success_count, "✗": failed_count})
                    pbar.update(1)
    
    # Add back the already successful ones
    success_count += already_done
    
    return success_count, failed_count, failed_ids


def main():
    parser = argparse.ArgumentParser(description="Validate Drug Extractions with Status Tracking")
    parser.add_argument("--input", default="gs://entity-extraction-agent-data-dev/Conference/abstract_titles.csv", 
                        help="Input CSV path (local or gs://bucket/path)")
    parser.add_argument("--output_dir", default="gs://entity-extraction-agent-data-dev/Conference/drug", 
                        help="Output directory (local or gs://bucket/path) - same as extraction")
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts to validate")
    parser.add_argument("--parallel_workers", type=int, default=5, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Determine if using GCS or local storage based on output_dir
    is_gcs = args.output_dir.startswith("gs://")
    
    # Initialize storage client for output
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
    
    # Auto-generate output CSV filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_filename = f"validation_{timestamp}.csv"
    
    # For local storage, ensure output directory exists
    if not is_gcs:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("✅ Drug Validation Processor")
    print("=" * 60)
    print(f"Input:      {args.input}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Output CSV: {output_csv_filename}")
    print(f"Storage:    {'GCS' if is_gcs else 'Local'}")
    print(f"Model:      {config.VALIDATION_MODEL}")
    print(f"Limit:      {args.limit or 'all'}")
    print(f"Workers:    {args.parallel_workers}")
    print()
    
    # Load abstracts with their extraction results
    print("Loading abstracts with extraction results...")
    inputs, original_rows, input_fieldnames = load_abstracts_for_validation(
        input_filename, input_storage, storage, args.limit
    )
    print(f"✓ Loaded {len(inputs)} abstracts with extraction results")
    print()
    
    if not inputs:
        print("⚠ No abstracts to validate. Run extraction first.")
        return
    
    # Track batch timing
    batch_start = time.time()
    batch_started_at = _get_timestamp()
    
    # Run validation
    print("Running validations...")
    success_count, failed_count, failed_ids = run_validation_batch(
        inputs, storage, args.parallel_workers
    )
    
    batch_duration = time.time() - batch_start
    
    # Save batch status
    save_batch_status(
        pipeline="drug",
        total_abstracts=len(inputs),
        success_count=success_count,
        failed_count=failed_count,
        failed_ids=failed_ids,
        total_duration=batch_duration,
        started_at=batch_started_at,
        storage=storage,
        phase="validation",
    )
    
    # Save results to CSV
    save_results_csv(inputs, original_rows, input_fieldnames, storage, output_csv_filename)
    
    # Summary
    print()
    print("=" * 60)
    print("📊 Validation Complete:")
    print(f"   Total:    {len(inputs)}")
    print(f"   Success:  {success_count}")
    print(f"   Failed:   {failed_count}")
    if failed_ids:
        print(f"   Failed IDs: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")
    print(f"   Duration: {batch_duration:.1f}s")
    print(f"   Output:   {args.output_dir}")
    print()
    print(f"✓ Status updated in batch_status.json")
    print(f"✓ Per-abstract results saved to abstracts/*/validation.json")
    print(f"✓ CSV saved to {output_csv_filename}")


if __name__ == "__main__":
    main()
