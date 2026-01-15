#!/usr/bin/env python3
"""
Drug Class Validation Processor

Batch processor for validating drug class extractions.
Takes extraction output and validates each drug's classification.

Usage:
    python -m src.scripts.drug_class.validation_processor --input data/input.csv --output_dir data/output
"""

import argparse
import csv
import json
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.drug_class import (
    validate_drug_class,
    ValidationInput,
    ValidationOutput,
    config,
)
from src.agents.drug_class.pipeline import LocalStorageClient


@dataclass
class ProcessResult:
    """Result of validating a single extraction."""
    abstract_id: str
    drug_name: str
    validation_status: str = ""
    issues_count: int = 0
    llm_calls: int = 0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return bool(self.validation_status and not self.error)


def load_validations(
    csv_path: str,
    extraction_column: str,
    limit: int = None
) -> tuple[list[ValidationInput], list[dict], list[str]]:
    """Load extraction results from CSV for validation.
    
    Args:
        csv_path: Path to extraction output CSV
        extraction_column: Name of column containing extraction response JSON
        limit: Optional limit on rows
        
    Returns:
        tuple: (validation_inputs, original_rows, fieldnames)
    """
    inputs = []
    original_rows = []
    fieldnames = []
    
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        
        header_map = {h.lower().strip(): h for h in fieldnames}
        id_col = header_map.get('abstract_id') or header_map.get('id')
        title_col = header_map.get('abstract_title') or header_map.get('title')
        abstract_col = header_map.get('full_abstract') or header_map.get('abstract')
        
        # Find extraction column
        extraction_col = None
        for col in fieldnames:
            if extraction_column.lower() in col.lower():
                extraction_col = col
                break
        
        if not extraction_col:
            raise ValueError(f"Could not find extraction column matching '{extraction_column}' in CSV")
        
        for row in reader:
            abstract_id = row.get(id_col, "") if id_col else ""
            abstract_title = row.get(title_col, "") if title_col else ""
            full_abstract = row.get(abstract_col, "") if abstract_col else ""
            extraction_response = row.get(extraction_col, "{}")
            
            # Parse extraction response
            try:
                extraction_result = json.loads(extraction_response) if extraction_response else {}
            except json.JSONDecodeError:
                extraction_result = {"error": "Failed to parse extraction response"}
            
            # Extract drug name from extraction result or row
            drug_name = extraction_result.get("drug_name", "")
            if not drug_name:
                drug_col = header_map.get('drug_name') or header_map.get('drug')
                drug_name = row.get(drug_col, "") if drug_col else ""
            
            if not abstract_id:
                continue
            
            inputs.append(ValidationInput(
                abstract_id=str(abstract_id),
                drug_name=str(drug_name),
                abstract_title=str(abstract_title),
                full_abstract=str(full_abstract),
                search_results=[],  # Search results would come from cache
                extraction_result=extraction_result,
            ))
            original_rows.append(row)
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


def process_single(input_data: ValidationInput, storage: LocalStorageClient) -> ProcessResult:
    """Validate single extraction and return result."""
    # Skip if extraction had error
    if input_data.extraction_result.get("error"):
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            drug_name=input_data.drug_name,
            validation_status="SKIP",
            error=f"Extraction had error: {input_data.extraction_result.get('error')}",
        )
    
    try:
        # Load search results from cache if available
        search_cache_path = f"search_cache/{input_data.drug_name.lower().replace(' ', '_')}.json"
        cached_data = storage.read(search_cache_path)
        if cached_data:
            cache = json.loads(cached_data)
            input_data.search_results = cache.get("drug_class_results", [])
        
        result: ValidationOutput = validate_drug_class(input_data)
        
        # Save validation output
        storage.write(
            f"abstracts/{input_data.abstract_id}/validation_{input_data.drug_name}.json",
            result.model_dump_json(indent=2)
        )
        
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            drug_name=input_data.drug_name,
            validation_status=result.validation_status,
            issues_count=len(result.issues_found),
            llm_calls=result.llm_calls,
        )
        
    except Exception as e:
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            drug_name=input_data.drug_name,
            error=f"Validation error: {e}",
        )


def save_results(
    results: list[tuple[int, ProcessResult]],
    original_rows: list[dict],
    fieldnames: list[str],
    output_path: str,
):
    """Save results to CSV with all input columns plus validation response column."""
    results.sort(key=lambda x: x[0])
    
    output_fieldnames = fieldnames + [
        "validation_status",
        "validation_issues_count",
        "validation_llm_calls",
        "validation_error"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for idx, result in results:
            row = dict(original_rows[idx])
            row["validation_status"] = result.validation_status
            row["validation_issues_count"] = result.issues_count
            row["validation_llm_calls"] = result.llm_calls
            row["validation_error"] = result.error or ""
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Drug Class Validation Processor")
    parser.add_argument("--input", required=True, help="Input CSV with extraction results")
    parser.add_argument("--output_dir", default="data/drug_class/output", help="Output directory")
    parser.add_argument("--output_csv", default=None, help="Output CSV file")
    parser.add_argument("--extraction_column", default="extraction_response", 
                        help="Column name containing extraction response JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows to validate")
    parser.add_argument("--parallel_workers", type=int, default=10, help="Number of parallel workers")
    args = parser.parse_args()
    
    if not args.output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"data/drug_class/output/validation_{timestamp}.csv"
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    print("✅ Drug Class Validation Processor")
    print("=" * 60)
    print(f"Input:      {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Model:      {config.VALIDATION_MODEL}")
    print(f"Extraction column: {args.extraction_column}")
    print(f"Limit:      {args.limit or 'all'}")
    print(f"Workers:    {args.parallel_workers}")
    print()
    
    storage = LocalStorageClient(base_path=args.output_dir)
    
    print("Loading extraction results...")
    inputs, original_rows, fieldnames = load_validations(
        args.input, args.extraction_column, args.limit
    )
    print(f"✓ Loaded {len(inputs)} extraction results")
    print()
    
    print("Validating...")
    results: list[tuple[int, ProcessResult]] = []
    save_interval = 5
    last_saved_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        future_to_idx = {
            executor.submit(process_single, inp, storage): i
            for i, inp in enumerate(inputs)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            inp = inputs[idx]
            
            try:
                result = future.result()
                results.append((idx, result))
                
                status_icon = "✓" if result.success else "✗"
                if result.success:
                    output = f"{result.validation_status} ({result.issues_count} issues)"
                else:
                    output = result.error[:30] if result.error else "error"
                
                print(f"[{len(results)}/{len(inputs)}] {inp.abstract_id}: {status_icon} {output}")
                
                # Intermediate save
                if len(results) - last_saved_count >= save_interval:
                    save_results(results, original_rows, fieldnames, args.output_csv)
                    last_saved_count = len(results)
                    print(f"💾 Intermediate save: {len(results)} results")
                    
            except Exception as e:
                print(f"✗ Error validating {inp.abstract_id}: {e}")
                results.append((idx, ProcessResult(
                    abstract_id=inp.abstract_id,
                    drug_name=inp.drug_name,
                    error=str(e)
                )))
    
    # Final save
    print()
    save_results(results, original_rows, fieldnames, args.output_csv)
    print(f"✓ Results saved to {args.output_csv}")
    
    # Summary
    successful = sum(1 for _, r in results if r.success)
    passed = sum(1 for _, r in results if r.validation_status == "PASS")
    review = sum(1 for _, r in results if r.validation_status == "REVIEW")
    failed = sum(1 for _, r in results if r.validation_status == "FAIL")
    
    print()
    print("📊 Summary:")
    print(f"   Total:   {len(results)}")
    print(f"   Success: {successful}")
    print(f"   PASS:    {passed}")
    print(f"   REVIEW:  {review}")
    print(f"   FAIL:    {failed}")
    if results:
        print(f"   Rate:    {successful/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()

