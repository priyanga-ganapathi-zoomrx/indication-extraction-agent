#!/usr/bin/env python3
"""
Drug Class Validation Processor

Batch processor for validating drug class extractions.
Takes extraction output and validates each drug's classification.

Key features:
- Parallel processing for improved throughput
- Per-abstract validation status tracking (validation.json)
- Batch-level status summary (validation_batch_status.json)
- Retry logic for failed validations
- Real-time progress monitoring with tqdm
- Execution time tracking (accumulates across retries)

Usage:
    python -m src.scripts.drug_class.validation_processor --input ASCO2025/input/abstract_titles.csv --output_dir ASCO2025/drug_class
"""

import argparse
import csv
import json
import time
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.agents.drug_class import (
    validate_drug_class,
    ValidationInput,
    ValidationOutput,
    DrugClassInput,
    Step2Output,
    config,
)
from src.agents.core.storage import LocalStorageClient


def _get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use in filenames.
    
    Replaces characters that are problematic in file paths.
    """
    # Replace characters that could be interpreted as path separators or are invalid
    sanitized = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    return sanitized


@dataclass
class ProcessResult:
    """Result of validating a single abstract (all drugs)."""
    abstract_id: str
    abstract_title: str = ""
    drug_results: list = None  # List of per-drug validation results
    overall_status: str = "pending"  # success, failed
    error: Optional[str] = None
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        if self.drug_results is None:
            self.drug_results = []
    
    @property
    def success(self) -> bool:
        return self.overall_status == "success" and not self.error


def load_abstracts(
    csv_path: str,
    storage: LocalStorageClient,
    limit: int = None,
) -> tuple[list[DrugClassInput], list[dict], list[str]]:
    """Load abstracts from CSV that have extraction results.
    
    Only returns abstracts that have step2_output.json (extraction completed).
    
    Args:
        csv_path: Path to input CSV
        storage: Storage client for drug_class output directory
        limit: Optional limit on rows
        
    Returns:
        tuple: (inputs, original_rows, fieldnames)
    """
    inputs = []
    original_rows = []
    fieldnames = []
    skipped_no_extraction = []
    
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        
        header_map = {h.lower().strip(): h for h in fieldnames}
        id_col = header_map.get('abstract_id') or header_map.get('id')
        title_col = header_map.get('abstract_title') or header_map.get('title')
        abstract_col = header_map.get('full_abstract') or header_map.get('abstract')
        
        for row in reader:
            abstract_id = row.get(id_col, "") if id_col else ""
            abstract_title = row.get(title_col, "") if title_col else ""
            full_abstract = row.get(abstract_col, "") if abstract_col else ""
            
            if not abstract_id:
                continue
            
            # Check if extraction exists (step2_output.json)
            try:
                storage.download_json(f"abstracts/{abstract_id}/step2_output.json")
            except FileNotFoundError:
                skipped_no_extraction.append(abstract_id)
                continue
            
            inputs.append(DrugClassInput(
                abstract_id=str(abstract_id),
                abstract_title=str(abstract_title),
                full_abstract=str(full_abstract),
            ))
            original_rows.append(row)
    
    if skipped_no_extraction:
        print(f"⚠ Skipped {len(skipped_no_extraction)} abstracts without extraction: {skipped_no_extraction[:5]}{'...' if len(skipped_no_extraction) > 5 else ''}")
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


def get_validation_status(abstract_id: str, storage: LocalStorageClient) -> Optional[dict]:
    """Load validation status for an abstract if it exists."""
    try:
        return storage.download_json(f"abstracts/{abstract_id}/validation.json")
    except FileNotFoundError:
        return None


def should_process_validation(abstract_id: str, storage: LocalStorageClient) -> bool:
    """Check if validation should be run for this abstract.
    
    Returns True if:
    - No validation.json exists
    - validation.json exists but status is 'failed'
    """
    validation = get_validation_status(abstract_id, storage)
    if validation is None:
        return True
    return validation.get("overall_status") != "success"


def save_validation_result(
    abstract_id: str,
    abstract_title: str,
    drug_results: list[dict],
    duration: float,
    storage: LocalStorageClient,
    error: Optional[str] = None,
):
    """Save consolidated validation.json for an abstract.
    
    Args:
        abstract_id: The abstract ID
        abstract_title: The abstract title
        drug_results: List of per-drug validation results
        duration: Duration in seconds
        storage: Storage client
        error: Optional error message
    """
    # Determine overall status
    if error:
        overall_status = "failed"
    elif all(r.get("success", False) for r in drug_results):
        overall_status = "success"
    else:
        overall_status = "failed"
    
    # Count validation statuses
    pass_count = sum(1 for r in drug_results if r.get("validation_status") == "PASS")
    review_count = sum(1 for r in drug_results if r.get("validation_status") == "REVIEW")
    fail_count = sum(1 for r in drug_results if r.get("validation_status") == "FAIL")
    error_count = sum(1 for r in drug_results if r.get("error"))
    
    validation_data = {
        "abstract_id": abstract_id,
        "abstract_title": abstract_title,
        "overall_status": overall_status,
        "drugs_validated": len(drug_results),
        "summary": {
            "pass": pass_count,
            "review": review_count,
            "fail": fail_count,
            "errors": error_count,
        },
        "drug_results": drug_results,
        "duration_seconds": round(duration, 2),
        "completed_at": _get_timestamp(),
        "error": error,
    }
    
    storage.upload_json(f"abstracts/{abstract_id}/validation.json", validation_data)
    return validation_data


def save_batch_status(
    storage: LocalStorageClient,
    inputs: list[DrugClassInput],
    total_duration: float,
    started_at: str,
) -> dict:
    """Save or update validation_batch_status.json with accumulated duration.
    
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
        validation = get_validation_status(inp.abstract_id, storage)
        if validation is None:
            not_processed_count += 1
        elif validation.get("overall_status") == "success":
            success_count += 1
        else:
            failed_count += 1
            failed_ids.append(inp.abstract_id)
    
    # Load existing batch status if available (for accumulation)
    existing_duration = 0.0
    original_started_at = started_at
    try:
        existing_status = storage.download_json("validation_batch_status.json")
        existing_duration = existing_status.get("total_duration_seconds", 0.0)
        # Preserve original started_at from first run
        original_started_at = existing_status.get("started_at", started_at)
    except FileNotFoundError:
        pass
    
    accumulated_duration = existing_duration + total_duration
    
    batch_status = {
        "pipeline": "drug_class_validation",
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
    
    storage.upload_json("validation_batch_status.json", batch_status)
    return batch_status


def process_single(inp: DrugClassInput, storage: LocalStorageClient) -> ProcessResult:
    """Validate all drug extractions for a single abstract.
    
    Loads step2_output.json from storage and validates each drug extraction.
    Saves per-drug validation_{drug}.json and consolidated validation.json.
    """
    abstract_id = inp.abstract_id
    start_time = time.time()
    drug_results = []
    
    # Load step2 output from storage
    try:
        step2_data = storage.download_json(f"abstracts/{abstract_id}/step2_output.json")
        step2_output = Step2Output(**step2_data)
    except FileNotFoundError:
        duration = time.time() - start_time
        save_validation_result(
            abstract_id, inp.abstract_title, [], duration, storage,
            error="Step 2 output not found"
        )
        return ProcessResult(
            abstract_id=abstract_id,
            abstract_title=inp.abstract_title,
            overall_status="failed",
            error="Step 2 output not found",
            duration_seconds=duration,
        )
    
    # Validate each drug extraction
    for drug_name, extraction_result in step2_output.extractions.items():
        drug_result = {
            "drug_name": drug_name,
            "success": False,
            "validation_status": "",
            "issues_count": 0,
            "llm_calls": 0,
            "error": None,
        }
        
        try:
            # Build extraction result dict for validation
            extraction_dict = {
                "drug_name": extraction_result.drug_name,
                "drug_classes": extraction_result.drug_classes,
                "selected_sources": extraction_result.selected_sources,
                "confidence_score": extraction_result.confidence_score,
                "extraction_details": [d.model_dump() for d in extraction_result.extraction_details],
                "reasoning": extraction_result.reasoning,
            }
            
            # Load search results from cache if available
            search_results = []
            # Normalize drug name the same way step2_search does
            normalized_drug = drug_name.lower().strip().replace(" ", "_").replace("-", "_").replace("/", "_")
            search_cache_path = f"search_cache/{normalized_drug}.json"
            try:
                cache = storage.download_json(search_cache_path)
                # Search cache stores results at drug_class_search.results
                search_results = cache.get("drug_class_search", {}).get("results", [])
            except FileNotFoundError:
                pass  # No cache available
            
            validation_input = ValidationInput(
                abstract_id=abstract_id,
                drug_name=drug_name,
                abstract_title=inp.abstract_title or "",
                full_abstract=inp.full_abstract or "",
                search_results=search_results,
                extraction_result=extraction_dict,
            )
            
            result: ValidationOutput = validate_drug_class(validation_input)
            
            # Save per-drug validation output (sanitize drug_name for filename)
            safe_drug_name = _sanitize_filename(drug_name)
            storage.upload_json(
                f"abstracts/{abstract_id}/validation_{safe_drug_name}.json",
                result.model_dump()
            )
            
            drug_result["success"] = True
            drug_result["validation_status"] = result.validation_status
            drug_result["issues_count"] = len(result.issues_found)
            drug_result["llm_calls"] = result.llm_calls
            
        except Exception as e:
            drug_result["error"] = f"Validation error: {e}"
        
        drug_results.append(drug_result)
    
    duration = time.time() - start_time
    
    # Save consolidated validation.json
    save_validation_result(
        abstract_id, inp.abstract_title, drug_results, duration, storage
    )
    
    # Determine overall status
    all_success = all(r.get("success", False) for r in drug_results)
    
    return ProcessResult(
        abstract_id=abstract_id,
        abstract_title=inp.abstract_title,
        drug_results=drug_results,
        overall_status="success" if all_success else "failed",
        duration_seconds=duration,
    )


def run_validation_batch(
    inputs: list[DrugClassInput],
    storage: LocalStorageClient,
    parallel_workers: int,
) -> tuple[list[ProcessResult], float]:
    """Run validation for all abstracts that need it.
    
    Args:
        inputs: List of all inputs
        storage: Storage client
        parallel_workers: Number of parallel workers
        
    Returns:
        tuple: (results, duration_seconds)
    """
    # Filter to only abstracts that need validation
    pending = [inp for inp in inputs if should_process_validation(inp.abstract_id, storage)]
    
    if not pending:
        print("  No abstracts pending for validation")
        return [], 0.0
    
    print(f"  Processing {len(pending)} abstracts (workers: {parallel_workers})")
    
    start_time = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        future_to_inp = {
            executor.submit(process_single, inp, storage): inp
            for inp in pending
        }
        
        # Use tqdm for progress bar
        with tqdm(
            total=len(pending),
            desc="  Validating",
            unit="abstract",
            leave=True,
        ) as pbar:
            for future in concurrent.futures.as_completed(future_to_inp):
                inp = future_to_inp[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update progress bar
                    if result.success:
                        drug_count = len(result.drug_results)
                        pbar.set_postfix_str(f"✓ {inp.abstract_id} ({drug_count} drugs)")
                    else:
                        pbar.set_postfix_str(f"✗ {inp.abstract_id}")
                        
                except Exception as e:
                    results.append(ProcessResult(
                        abstract_id=inp.abstract_id,
                        abstract_title=inp.abstract_title,
                        overall_status="failed",
                        error=str(e),
                    ))
                    pbar.set_postfix_str(f"✗ {inp.abstract_id}: {str(e)[:30]}")
                
                pbar.update(1)
    
    duration = time.time() - start_time
    return results, duration


def save_results_csv(
    inputs: list[DrugClassInput],
    original_rows: list[dict],
    fieldnames: list[str],
    storage: LocalStorageClient,
    output_path: str,
):
    """Save validation results to CSV with all input columns plus model_response.
    
    Reads validation.json from storage for each abstract to build the model response.
    """
    output_fieldnames = fieldnames + ["model_response"]
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for i, inp in enumerate(inputs):
            row = dict(original_rows[i]) if i < len(original_rows) else {}
            
            # Get validation.json for this abstract
            try:
                validation_data = storage.download_json(f"abstracts/{inp.abstract_id}/validation.json")
                model_response = validation_data
            except FileNotFoundError:
                model_response = {
                    "abstract_id": inp.abstract_id,
                    "overall_status": "not_processed",
                    "error": "Validation not found",
                }
            
            row["model_response"] = json.dumps(model_response, indent=2)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Drug Class Validation Processor")
    parser.add_argument("--input", default="data/ASCO2025/input/abstract_titles.csv", help="Input CSV file (e.g., ASCO2025/input/abstract_titles.csv)")
    parser.add_argument("--output_dir", default="data/ASCO2025/drug_class", help="Output directory (e.g., ASCO2025/drug_class)")
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts to validate")
    parser.add_argument("--parallel_workers", type=int, default=10, help="Number of parallel workers")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Record batch start time
    batch_started_at = _get_timestamp()
    batch_start_time = time.time()
    
    print("✅ Drug Class Validation Processor")
    print("=" * 60)
    print(f"Input:      {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model:      {config.VALIDATION_MODEL}")
    print(f"Limit:      {args.limit or 'all'}")
    print(f"Workers:    {args.parallel_workers}")
    print()
    
    storage = LocalStorageClient(base_dir=args.output_dir)
    
    print("Loading abstracts...")
    inputs, original_rows, fieldnames = load_abstracts(args.input, storage, args.limit)
    print(f"✓ Loaded {len(inputs)} abstracts with extraction data")
    print()
    
    # Run validation batch
    print("Validating...")
    results, run_duration = run_validation_batch(inputs, storage, args.parallel_workers)
    
    # Calculate total batch duration
    batch_duration = time.time() - batch_start_time
    
    # Save batch status (with accumulation)
    print()
    print("Saving batch status...")
    batch_status = save_batch_status(storage, inputs, batch_duration, batch_started_at)
    print(f"✓ Saved validation_batch_status.json")
    
    # Save CSV output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_output_path = str(output_dir / f"validation_{timestamp}.csv")
    print(f"Saving CSV output...")
    save_results_csv(inputs, original_rows, fieldnames, storage, csv_output_path)
    print(f"✓ Saved {csv_output_path}")
    
    # Summary
    print()
    print("=" * 60)
    print("📊 Summary:")
    
    print(f"   Total abstracts: {len(inputs)}")
    print(f"   Success:         {batch_status['success']}")
    print(f"   Failed:          {batch_status['failed']}")
    print(f"   Not processed:   {batch_status['not_processed']}")
    
    # Count validation statuses from results
    if results:
        total_drugs = sum(len(r.drug_results) for r in results)
        pass_count = sum(
            sum(1 for d in r.drug_results if d.get("validation_status") == "PASS")
            for r in results
        )
        review_count = sum(
            sum(1 for d in r.drug_results if d.get("validation_status") == "REVIEW")
            for r in results
        )
        fail_count = sum(
            sum(1 for d in r.drug_results if d.get("validation_status") == "FAIL")
            for r in results
        )
        
        print()
        print("   Per-drug validation:")
        print(f"      Total drugs: {total_drugs}")
        print(f"      PASS:        {pass_count}")
        print(f"      REVIEW:      {review_count}")
        print(f"      FAIL:        {fail_count}")
    
    print()
    print(f"   This run:        {batch_duration:.2f}s")
    print(f"   Total duration:  {batch_status['total_duration_seconds']:.2f}s")
    print(f"   Output:          {args.output_dir}")


if __name__ == "__main__":
    main()

