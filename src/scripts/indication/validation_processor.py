#!/usr/bin/env python3
"""
Validation Processor for Indication Extraction

Validates previously extracted indications with:
- Parallel processing for improved throughput
- Per-abstract status tracking (updates status.json)
- Batch-level status summary (updates batch_status.json)
- Retry logic for failed validations
- Real-time progress monitoring with tqdm

Usage:
    python -m src.scripts.indication.validation_processor --input ASCO2025/input/abstract_titles.csv --output_dir ASCO2025/indication
"""

import argparse
import csv
import json
import re
import time
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.agents.indication import IndicationValidationAgent
from src.agents.core.storage import LocalStorageClient


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
    session_title: str
    extraction_result: dict


def load_abstracts_for_validation(
    csv_path: str,
    storage: LocalStorageClient,
    limit: int = None,
) -> tuple[list[ValidationInput], list[dict], list[str]]:
    """Load abstracts from CSV and their extraction results from storage.
    
    Only returns abstracts that have successful extraction results.
    
    Returns:
        tuple: (ValidationInput objects, original_rows, fieldnames)
    """
    inputs = []
    original_rows = []
    skipped_no_extraction = []
    fieldnames = []
    
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        
        # Find column names (case-insensitive)
        header_map = {h.lower().strip(): h for h in fieldnames}
        id_col = header_map.get('abstract_id') or header_map.get('id')
        title_col = header_map.get('abstract_title') or header_map.get('title')
        session_col = header_map.get('session_title') or header_map.get('session')
        
        for row in reader:
            abstract_id = row.get(id_col, "") if id_col else ""
            abstract_title = row.get(title_col, "") if title_col else ""
            session_title = row.get(session_col, "") if session_col else ""
            
            if not abstract_id:
                continue
            
            # Try to load extraction result from storage
            try:
                extraction_result = storage.download_json(f"abstracts/{abstract_id}/extraction.json")
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
                session_title=str(session_title),
                extraction_result=extraction_result,
            ))
            original_rows.append(row)
    
    if skipped_no_extraction:
        print(f"⚠ Skipped {len(skipped_no_extraction)} abstracts without extraction: {skipped_no_extraction[:5]}{'...' if len(skipped_no_extraction) > 5 else ''}")
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


def get_abstract_status(abstract_id: str, storage: LocalStorageClient) -> Optional[dict]:
    """Load status for an abstract if it exists."""
    try:
        return storage.download_json(f"abstracts/{abstract_id}/status.json")
    except FileNotFoundError:
        return None
    except Exception:
        return None


def should_process_validation(abstract_id: str, storage: LocalStorageClient) -> bool:
    """Check if abstract needs validation (extraction successful but validation not done)."""
    status = get_abstract_status(abstract_id, storage)
    if not status:
        return False  # No status means extraction wasn't done
    
    extraction = status.get("extraction", {})
    validation = status.get("validation", {})
    
    # Only validate if extraction was successful and validation not already successful
    return extraction.get("status") == "success" and validation.get("status") != "success"


def extract_json_block(content: str) -> str:
    """Extract JSON from markdown code block or return raw content."""
    match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    return match.group(1) if match else content


def pretty_print_json(json_str: str) -> str:
    """Pretty print JSON string."""
    try:
        parsed = json.loads(json_str)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return json_str


def process_single(
    validator: IndicationValidationAgent,
    input_data: ValidationInput,
    storage: LocalStorageClient,
) -> ProcessResult:
    """Validate single extraction and return result with timing."""
    start_time = time.time()
    
    try:
        raw = validator.invoke(
            session_title=input_data.session_title,
            abstract_title=input_data.abstract_title,
            extraction_result=input_data.extraction_result,
            abstract_id=input_data.abstract_id,
        )
        
        messages = raw.get("messages", [])
        if not messages:
            duration = time.time() - start_time
            return ProcessResult(
                abstract_id=input_data.abstract_id,
                abstract_title=input_data.abstract_title,
                error="No messages",
                duration_seconds=duration,
            )
        
        content = messages[-1].content
        if not content:
            duration = time.time() - start_time
            return ProcessResult(
                abstract_id=input_data.abstract_id,
                abstract_title=input_data.abstract_title,
                error="Empty response",
                duration_seconds=duration,
            )
        
        # Extract and pretty print JSON
        json_str = extract_json_block(content)
        pretty_json = pretty_print_json(json_str)
        
        duration = time.time() - start_time
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            abstract_title=input_data.abstract_title,
            response_json=pretty_json,
            duration_seconds=duration,
        )
        
    except Exception as e:
        duration = time.time() - start_time
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            abstract_title=input_data.abstract_title,
            error=str(e),
            duration_seconds=duration,
        )


def save_validation_result(
    result: ProcessResult,
    storage: LocalStorageClient,
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
    storage: LocalStorageClient,
    output_path: str,
) -> None:
    """Save all validation results to CSV with input columns plus model_response column.
    
    Reads validation.json from storage for each abstract to build the CSV.
    """
    response_column = "validation_response"
    fieldnames = input_fieldnames + [response_column]
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    
    print(f"✓ CSV saved to {output_path}")


def save_batch_status(
    pipeline: str,
    total_abstracts: int,
    success_count: int,
    failed_count: int,
    failed_ids: list[str],
    total_duration: float,
    started_at: str,
    storage: LocalStorageClient,
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
    storage: LocalStorageClient,
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
    
    # Initialize validator once (shared across workers)
    validator = IndicationValidationAgent()
    
    success_count = 0
    failed_count = 0
    failed_ids = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {
            executor.submit(process_single, validator, inp, storage): inp
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
    parser = argparse.ArgumentParser(description="Validate Indication Extractions with Status Tracking")
    parser.add_argument("--input", default="data/ASCO2025/input/abstract_titles.csv", help="Input CSV path (e.g., ASCO2025/input/abstract_titles.csv)")
    parser.add_argument("--output_dir", default="data/ASCO2025/indication", help="Output directory (e.g., ASCO2025/indication) - same as extraction")
    parser.add_argument("--output_csv", default=None, help="Output CSV path (auto-generated if not provided)")
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts to validate")
    parser.add_argument("--parallel_workers", type=int, default=5, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Auto-generate output CSV path if not provided
    if not args.output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"{args.output_dir}/validation_{timestamp}.csv"
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize storage (same directory as extraction)
    storage = LocalStorageClient(base_dir=args.output_dir)
    
    print("🔍 Indication Validation Processor")
    print("=" * 60)
    print(f"Input:      {args.input}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Limit:      {args.limit or 'all'}")
    print(f"Workers:    {args.parallel_workers}")
    print()
    
    # Load abstracts with their extraction results
    print("Loading abstracts with extraction results...")
    inputs, original_rows, input_fieldnames = load_abstracts_for_validation(args.input, storage, args.limit)
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
        pipeline="indication",
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
    save_results_csv(inputs, original_rows, input_fieldnames, storage, args.output_csv)
    
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
    print(f"✓ Status updated in {args.output_dir}/batch_status.json")
    print(f"✓ Per-abstract results saved to {args.output_dir}/abstracts/*/validation.json")
    print(f"✓ CSV saved to {args.output_csv}")


if __name__ == "__main__":
    main()
