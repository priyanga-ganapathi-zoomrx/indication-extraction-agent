#!/usr/bin/env python3
"""
Step 1 Processor: Regimen Identification

Processes multiple abstracts through Step 1 (Regimen Identification) only.
Useful for testing Step 1 in isolation.

Usage:
    python -m src.scripts.drug_class.step1_processor --input data/input.csv --output_dir data/output
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
    identify_regimen,
    DrugClassInput,
    RegimenInput,
    Step1Output,
    PipelineStatus,
    config,
)
from src.agents.core.storage import LocalStorageClient


@dataclass
class ProcessResult:
    """Result of processing a single abstract."""
    abstract_id: str
    components: list[str] = None
    llm_calls: int = 0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.components is not None and not self.error


def load_abstracts(
    csv_path: str,
    limit: int = None
) -> tuple[list[DrugClassInput], list[dict], list[str]]:
    """Load abstracts from CSV into DrugClassInput objects.
    
    Expects CSV with columns:
    - abstract_id
    - abstract_title
    - firm (comma-separated string, will be parsed to list)
    - extraction_response (JSON string with Primary Drugs, Secondary Drugs, Comparator Drugs)
    """
    inputs = []
    original_rows = []
    fieldnames = []
    
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        
        for row in reader:
            abstract_id = row.get('abstract_id', '').strip()
            abstract_title = row.get('abstract_title', '').strip()
            firm = row.get('firm', '').strip()
            
            if not abstract_id or not abstract_title:
                continue
            
            # Parse extraction_response JSON
            extraction_response_str = row.get('extraction_response', '{}').strip()
            extraction = {}
            if extraction_response_str:
                try:
                    extraction = json.loads(extraction_response_str)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to handle it gracefully
                    pass
            
            # Extract drug lists from extraction_response
            # Handle both "Primary Drugs" and "primary_drugs" key variations
            primary_drugs = extraction.get('Primary Drugs', extraction.get('primary_drugs', []))
            secondary_drugs = extraction.get('Secondary Drugs', extraction.get('secondary_drugs', []))
            comparator_drugs = extraction.get('Comparator Drugs', extraction.get('comparator_drugs', []))
            
            # Ensure they are lists and convert to strings
            if not isinstance(primary_drugs, list):
                primary_drugs = [primary_drugs] if primary_drugs else []
            if not isinstance(secondary_drugs, list):
                secondary_drugs = [secondary_drugs] if secondary_drugs else []
            if not isinstance(comparator_drugs, list):
                comparator_drugs = [comparator_drugs] if comparator_drugs else []
            
            # Normalize drug names (strip whitespace, filter empty)
            primary_drugs = [str(d).strip() for d in primary_drugs if d and str(d).strip()]
            secondary_drugs = [str(d).strip() for d in secondary_drugs if d and str(d).strip()]
            comparator_drugs = [str(d).strip() for d in comparator_drugs if d and str(d).strip()]
            
            # Parse firms from comma-separated string to list
            firms = [f.strip() for f in firm.split(',') if f.strip()] if firm else []
            
            inputs.append(DrugClassInput(
                abstract_id=abstract_id,
                abstract_title=abstract_title,
                primary_drugs=primary_drugs,
                secondary_drugs=secondary_drugs,
                comparator_drugs=comparator_drugs,
                firms=firms,
            ))
            original_rows.append(row)
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


def process_single(inp: DrugClassInput, storage: LocalStorageClient) -> ProcessResult:
    """Process Step 1 for a single abstract."""
    abstract_id = inp.abstract_id
    all_drugs = inp.primary_drugs + inp.secondary_drugs + inp.comparator_drugs
    
    if not all_drugs:
        return ProcessResult(abstract_id=abstract_id, error="No drugs to process")
    
    step1_output = Step1Output()
    llm_calls = 0
    
    try:
        for drug in all_drugs:
            components = identify_regimen(RegimenInput(
                abstract_id=abstract_id,
                abstract_title=inp.abstract_title,
                drug=drug,
            ))
            step1_output.mark_success(drug, components)
            llm_calls += 1
        
        # Save output
        storage.upload_json(
            f"abstracts/{abstract_id}/step1_output.json",
            step1_output.model_dump()
        )
        
        # Update status
        status = PipelineStatus(abstract_id=abstract_id, abstract_title=inp.abstract_title)
        status.steps["step1_regimen"] = {"status": "success", "llm_calls": llm_calls}
        status.last_completed_step = "step1_regimen"
        status.last_updated = datetime.utcnow().isoformat() + "Z"
        storage.upload_json(
            f"abstracts/{abstract_id}/status.json",
            status.to_dict()
        )
        
        return ProcessResult(
            abstract_id=abstract_id,
            components=step1_output.get_all_components(),
            llm_calls=llm_calls,
        )
        
    except Exception as e:
        return ProcessResult(abstract_id=abstract_id, error=str(e))


def save_results(
    results: list[tuple[int, ProcessResult]],
    original_rows: list[dict],
    fieldnames: list[str],
    output_path: str,
):
    """Save results to CSV."""
    results.sort(key=lambda x: x[0])
    
    output_fieldnames = fieldnames + ["step1_components", "step1_error"]
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for idx, result in results:
            row = dict(original_rows[idx])
            row["step1_components"] = json.dumps(result.components, ensure_ascii=False) if result.components else ""
            row["step1_error"] = result.error or ""
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Step 1 Processor: Regimen Identification")
    parser.add_argument("--input", default="data/drug_class/input/drugs.csv", help="Input CSV file")
    parser.add_argument("--output_dir", default="data/drug_class/output", help="Output directory")
    parser.add_argument("--output_csv", default=None, help="Output CSV file (auto-generated if not specified)")
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts")
    parser.add_argument("--parallel_workers", type=int, default=50, help="Parallel workers")
    args = parser.parse_args()
    
    # Auto-generate output CSV path
    if not args.output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"data/drug_class/output/step1_{timestamp}.csv"
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    print("🧬 Step 1 Processor: Regimen Identification")
    print("=" * 60)
    print(f"Input:      {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Model:      {config.REGIMEN_MODEL}")
    print(f"Limit:      {args.limit or 'all'}")
    print(f"Workers:    {args.parallel_workers}")
    print()
    
    storage = LocalStorageClient(base_dir=args.output_dir)
    
    print("Loading abstracts...")
    inputs, original_rows, fieldnames = load_abstracts(args.input, args.limit)
    print(f"✓ Loaded {len(inputs)} abstracts")
    print()
    
    print("Processing...")
    results: list[tuple[int, ProcessResult]] = []
    
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
                
                status = "✓" if result.success else "✗"
                info = f"{len(result.components)} components" if result.success else result.error
                print(f"[{len(results)}/{len(inputs)}] {inp.abstract_id}: {status} {info}")
                
            except Exception as e:
                results.append((idx, ProcessResult(abstract_id=inp.abstract_id, error=str(e))))
                print(f"[{len(results)}/{len(inputs)}] {inp.abstract_id}: ✗ {e}")
    
    # Save results
    print()
    save_results(results, original_rows, fieldnames, args.output_csv)
    print(f"✓ Results saved to {args.output_csv}")
    
    # Summary
    successful = sum(1 for _, r in results if r.success)
    print()
    print("📊 Summary:")
    print(f"   Total:   {len(results)}")
    print(f"   Success: {successful}")
    print(f"   Failed:  {len(results) - successful}")
    if results:
        print(f"   Rate:    {successful/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()

