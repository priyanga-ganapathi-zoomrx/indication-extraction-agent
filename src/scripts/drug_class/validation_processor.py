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
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.drug_class import (
    validate_drug_class,
    ValidationInput,
    ValidationOutput,
    DrugClassInput,
    Step2Output,
    config,
)
from src.agents.core.storage import LocalStorageClient


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


def load_abstracts(csv_path: str, limit: int = None) -> tuple[list[DrugClassInput], list[dict], list[str]]:
    """Load abstracts from CSV.
    
    Args:
        csv_path: Path to input CSV
        limit: Optional limit on rows
        
    Returns:
        tuple: (inputs, original_rows, fieldnames)
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
        
        for row in reader:
            abstract_id = row.get(id_col, "") if id_col else ""
            abstract_title = row.get(title_col, "") if title_col else ""
            full_abstract = row.get(abstract_col, "") if abstract_col else ""
            
            if not abstract_id:
                continue
            
            inputs.append(DrugClassInput(
                abstract_id=str(abstract_id),
                abstract_title=str(abstract_title),
                full_abstract=str(full_abstract),
            ))
            original_rows.append(row)
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


def process_single(inp: DrugClassInput, storage: LocalStorageClient) -> list[ProcessResult]:
    """Validate all drug extractions for a single abstract.
    
    Loads step2_output.json from storage and validates each drug extraction.
    """
    abstract_id = inp.abstract_id
    results = []
    
    # Load step2 output from storage
    try:
        step2_data = storage.download_json(f"abstracts/{abstract_id}/step2_output.json")
        step2_output = Step2Output(**step2_data)
    except FileNotFoundError:
        return [ProcessResult(abstract_id=abstract_id, drug_name="", error="Step 2 output not found")]
    
    # Validate each drug extraction
    for drug_name, extraction_result in step2_output.extractions.items():
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
            normalized_drug = drug_name.lower().strip().replace(" ", "_").replace("-", "_")
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
            
            # Save validation output
            storage.upload_json(
                f"abstracts/{abstract_id}/validation_{drug_name}.json",
                result.model_dump()
            )
            
            results.append(ProcessResult(
                abstract_id=abstract_id,
                drug_name=drug_name,
                validation_status=result.validation_status,
                issues_count=len(result.issues_found),
                llm_calls=result.llm_calls,
            ))
            
        except Exception as e:
            results.append(ProcessResult(
                abstract_id=abstract_id,
                drug_name=drug_name,
                error=f"Validation error: {e}",
            ))
    
    return results


def save_results(
    results: list[ProcessResult],
    original_rows: list[dict],
    fieldnames: list[str],
    output_path: str,
    abstract_id_to_idx: dict[str, int],
):
    """Save results to CSV with all input columns plus validation response columns."""
    output_fieldnames = fieldnames + [
        "drug_name",
        "validation_status",
        "validation_issues_count",
        "validation_llm_calls",
        "validation_error"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for result in results:
            idx = abstract_id_to_idx.get(result.abstract_id, 0)
            row = dict(original_rows[idx]) if idx < len(original_rows) else {}
            row["drug_name"] = result.drug_name
            row["validation_status"] = result.validation_status
            row["validation_issues_count"] = result.issues_count
            row["validation_llm_calls"] = result.llm_calls
            row["validation_error"] = result.error or ""
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Drug Class Validation Processor")
    parser.add_argument("--input", default="data/drug_class/input/drugs.csv", help="Input CSV file")
    parser.add_argument("--output_dir", default="data/drug_class/output", help="Output directory")
    parser.add_argument("--output_csv", default=None, help="Output CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts to validate")
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
    print(f"Limit:      {args.limit or 'all'}")
    print(f"Workers:    {args.parallel_workers}")
    print()
    
    storage = LocalStorageClient(base_dir=args.output_dir)
    
    print("Loading abstracts...")
    inputs, original_rows, fieldnames = load_abstracts(args.input, args.limit)
    print(f"✓ Loaded {len(inputs)} abstracts")
    print()
    
    # Build abstract_id to index mapping
    abstract_id_to_idx = {inp.abstract_id: i for i, inp in enumerate(inputs)}
    
    print("Validating...")
    all_results: list[ProcessResult] = []
    completed_abstracts = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        future_to_inp = {
            executor.submit(process_single, inp, storage): inp
            for inp in inputs
        }
        
        for future in concurrent.futures.as_completed(future_to_inp):
            inp = future_to_inp[future]
            completed_abstracts += 1
            
            try:
                drug_results = future.result()
                all_results.extend(drug_results)
                
                for result in drug_results:
                    status_icon = "✓" if result.success else "✗"
                    if result.success:
                        output = f"{result.validation_status} ({result.issues_count} issues)"
                    else:
                        output = result.error[:40] if result.error else "error"
                    
                    print(f"[{completed_abstracts}/{len(inputs)}] {result.abstract_id}/{result.drug_name}: {status_icon} {output}")
                    
            except Exception as e:
                print(f"✗ Error validating {inp.abstract_id}: {e}")
                all_results.append(ProcessResult(
                    abstract_id=inp.abstract_id,
                    drug_name="",
                    error=str(e)
                ))
    
    # Final save
    print()
    save_results(all_results, original_rows, fieldnames, args.output_csv, abstract_id_to_idx)
    print(f"✓ Results saved to {args.output_csv}")
    
    # Summary
    successful = sum(1 for r in all_results if r.success)
    passed = sum(1 for r in all_results if r.validation_status == "PASS")
    review = sum(1 for r in all_results if r.validation_status == "REVIEW")
    failed = sum(1 for r in all_results if r.validation_status == "FAIL")
    
    print()
    print("📊 Summary:")
    print(f"   Total abstracts: {len(inputs)}")
    print(f"   Total validations: {len(all_results)}")
    print(f"   Success: {successful}")
    print(f"   PASS:    {passed}")
    print(f"   REVIEW:  {review}")
    print(f"   FAIL:    {failed}")
    if all_results:
        print(f"   Rate:    {successful/len(all_results)*100:.1f}%")


if __name__ == "__main__":
    main()

