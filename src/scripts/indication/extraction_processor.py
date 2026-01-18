#!/usr/bin/env python3
"""
Extraction Processor for Indication Extraction

Processes abstracts through the indication extraction pipeline with:
- Parallel processing for improved throughput
- Per-abstract status tracking (status.json)
- Batch-level status summary (batch_status.json)
- Retry logic for failed extractions
- Real-time progress monitoring with tqdm

Usage:
    python -m src.scripts.indication.extraction_processor --input ASCO2025/input/abstract_titles.csv --output_dir ASCO2025/indication
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

from src.agents.indication import IndicationAgent, IndicationInput
from src.agents.core.storage import LocalStorageClient


def _get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class ProcessResult:
    """Result of processing a single abstract."""
    abstract_id: str
    abstract_title: str = ""
    response_json: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0
    
    @property
    def success(self) -> bool:
        return bool(self.response_json and not self.error)


@dataclass
class StatusEntry:
    """Per-abstract status tracking."""
    abstract_id: str
    abstract_title: str
    extraction_status: str = "pending"  # pending, success, failed
    extraction_error: Optional[str] = None
    extraction_duration: float = 0.0
    extraction_completed_at: Optional[str] = None
    validation_status: str = "pending"
    validation_error: Optional[str] = None
    validation_duration: float = 0.0
    validation_completed_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "abstract_id": self.abstract_id,
            "abstract_title": self.abstract_title,
            "extraction": {
                "status": self.extraction_status,
                "error": self.extraction_error,
                "duration_seconds": round(self.extraction_duration, 2),
                "completed_at": self.extraction_completed_at,
            },
            "validation": {
                "status": self.validation_status,
                "error": self.validation_error,
                "duration_seconds": round(self.validation_duration, 2),
                "completed_at": self.validation_completed_at,
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StatusEntry":
        extraction = data.get("extraction", {})
        validation = data.get("validation", {})
        return cls(
            abstract_id=data.get("abstract_id", ""),
            abstract_title=data.get("abstract_title", ""),
            extraction_status=extraction.get("status", "pending"),
            extraction_error=extraction.get("error"),
            extraction_duration=extraction.get("duration_seconds", 0.0),
            extraction_completed_at=extraction.get("completed_at"),
            validation_status=validation.get("status", "pending"),
            validation_error=validation.get("error"),
            validation_duration=validation.get("duration_seconds", 0.0),
            validation_completed_at=validation.get("completed_at"),
        )


def load_abstracts(csv_path: str, limit: int = None) -> tuple[list[IndicationInput], list[dict], list[str]]:
    """Load abstracts from CSV into IndicationInput objects.
    
    Returns:
        tuple: (inputs, original_rows, fieldnames)
    """
    inputs = []
    original_rows = []
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
            
            if abstract_id or abstract_title:
                inputs.append(IndicationInput(
                    abstract_id=str(abstract_id),
                    abstract_title=str(abstract_title),
                    session_title=str(session_title),
                ))
                original_rows.append(row)
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


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


def get_abstract_status(abstract_id: str, storage: LocalStorageClient) -> Optional[StatusEntry]:
    """Load status for an abstract if it exists."""
    try:
        status_data = storage.download_json(f"abstracts/{abstract_id}/status.json")
        return StatusEntry.from_dict(status_data)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def should_process_extraction(abstract_id: str, storage: LocalStorageClient) -> bool:
    """Check if abstract needs extraction (not already successful)."""
    status = get_abstract_status(abstract_id, storage)
    if not status:
        return True
    return status.extraction_status != "success"


def process_single(
    agent: IndicationAgent,
    input_data: IndicationInput,
    storage: LocalStorageClient,
) -> ProcessResult:
    """Process single abstract and return result with timing."""
    start_time = time.time()
    
    try:
        raw = agent.invoke(input_data.abstract_title, input_data.session_title, input_data.abstract_id)
        
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


def save_extraction_result(
    result: ProcessResult,
    storage: LocalStorageClient,
) -> None:
    """Save extraction result and update status for a single abstract."""
    abstract_id = result.abstract_id
    timestamp = _get_timestamp()
    
    # Load existing status or create new
    status = get_abstract_status(abstract_id, storage)
    if not status:
        status = StatusEntry(
            abstract_id=abstract_id,
            abstract_title=result.abstract_title,
        )
    
    # Update extraction status
    status.extraction_status = "success" if result.success else "failed"
    status.extraction_error = result.error
    status.extraction_duration = result.duration_seconds
    status.extraction_completed_at = timestamp
    
    # Save status.json
    storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
    
    # Save extraction.json if successful
    if result.success:
        try:
            extraction_data = json.loads(result.response_json)
            storage.upload_json(f"abstracts/{abstract_id}/extraction.json", extraction_data)
        except json.JSONDecodeError:
            # If JSON parsing fails, save as raw
            storage.upload_json(f"abstracts/{abstract_id}/extraction.json", {"raw_response": result.response_json})


def save_results_csv(
    inputs: list[IndicationInput],
    original_rows: list[dict],
    input_fieldnames: list[str],
    storage: LocalStorageClient,
    output_path: str,
) -> None:
    """Save all results to CSV with input columns plus model_response column.
    
    Reads extraction.json from storage for each abstract to build the CSV.
    """
    response_column = "model_response"
    fieldnames = input_fieldnames + [response_column]
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for inp, original_row in zip(inputs, original_rows):
            row = dict(original_row)
            
            # Try to load extraction result from storage
            try:
                extraction_data = storage.download_json(f"abstracts/{inp.abstract_id}/extraction.json")
                row[response_column] = json.dumps(extraction_data, indent=2, ensure_ascii=False)
            except FileNotFoundError:
                # Check if there was an error in status
                status = get_abstract_status(inp.abstract_id, storage)
                if status and status.extraction_error:
                    row[response_column] = json.dumps({"error": status.extraction_error}, indent=2)
                else:
                    row[response_column] = json.dumps({"error": "Extraction not completed"}, indent=2)
            
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
    phase: str = "extraction",
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


def run_extraction_batch(
    inputs: list[IndicationInput],
    storage: LocalStorageClient,
    parallelism: int,
) -> tuple[int, int, list[str]]:
    """Run extraction for all abstracts with parallel processing.
    
    Returns:
        tuple: (success_count, failed_count, failed_ids)
    """
    # Filter to only process abstracts that need extraction
    pending_inputs = [inp for inp in inputs if should_process_extraction(inp.abstract_id, storage)]
    
    if not pending_inputs:
        print("  All abstracts already processed successfully.")
        # Count existing results
        success_count = 0
        for inp in inputs:
            status = get_abstract_status(inp.abstract_id, storage)
            if status and status.extraction_status == "success":
                success_count += 1
        return success_count, 0, []
    
    already_done = len(inputs) - len(pending_inputs)
    if already_done > 0:
        print(f"  Skipping {already_done} already successful extractions")
    
    print(f"  Processing {len(pending_inputs)} abstracts (parallelism: {parallelism})")
    
    # Initialize agent once (shared across workers)
    agent = IndicationAgent()
    
    success_count = 0
    failed_count = 0
    failed_ids = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {
            executor.submit(process_single, agent, inp, storage): inp
            for inp in pending_inputs
        }
        
        with tqdm(total=len(pending_inputs), desc="Extracting", unit="abstract") as pbar:
            for future in concurrent.futures.as_completed(futures):
                inp = futures[future]
                
                try:
                    result = future.result()
                    
                    # Save result immediately
                    save_extraction_result(result, storage)
                    
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
    parser = argparse.ArgumentParser(description="Batch Indication Extraction with Status Tracking")
    parser.add_argument("--input", default="data/ASCO2025/input/abstract_titles.csv", help="Input CSV path (e.g., ASCO2025/input/abstract_titles.csv)")
    parser.add_argument("--output_dir", default="data/ASCO2025/indication", help="Output directory (e.g., ASCO2025/indication)")
    parser.add_argument("--output_csv", default=None, help="Output CSV path (auto-generated if not provided)")
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts to process")
    parser.add_argument("--parallel_workers", type=int, default=5, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Auto-generate output CSV path if not provided
    if not args.output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"{args.output_dir}/extraction_{timestamp}.csv"
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize storage
    storage = LocalStorageClient(base_dir=args.output_dir)
    
    print("🏭 Indication Extraction Batch Processor")
    print("=" * 60)
    print(f"Input:      {args.input}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Limit:      {args.limit or 'all'}")
    print(f"Workers:    {args.parallel_workers}")
    print()
    
    # Load abstracts
    print("Loading abstracts...")
    inputs, original_rows, input_fieldnames = load_abstracts(args.input, args.limit)
    print(f"✓ Loaded {len(inputs)} abstracts")
    print()
    
    # Track batch timing
    batch_start = time.time()
    batch_started_at = _get_timestamp()
    
    # Run extraction
    print("Running extractions...")
    success_count, failed_count, failed_ids = run_extraction_batch(
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
        phase="extraction",
    )
    
    # Save results to CSV
    save_results_csv(inputs, original_rows, input_fieldnames, storage, args.output_csv)
    
    # Summary
    print()
    print("=" * 60)
    print("📊 Extraction Complete:")
    print(f"   Total:    {len(inputs)}")
    print(f"   Success:  {success_count}")
    print(f"   Failed:   {failed_count}")
    if failed_ids:
        print(f"   Failed IDs: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")
    print(f"   Duration: {batch_duration:.1f}s")
    print(f"   Output:   {args.output_dir}")
    print()
    print(f"✓ Status saved to {args.output_dir}/batch_status.json")
    print(f"✓ Per-abstract results saved to {args.output_dir}/abstracts/*/")
    print(f"✓ CSV saved to {args.output_csv}")


if __name__ == "__main__":
    main()
