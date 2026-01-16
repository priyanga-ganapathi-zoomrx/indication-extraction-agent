#!/usr/bin/env python3
"""
Step 3 Processor: Drug Class Selection

Processes multiple abstracts through Step 3 (Drug Class Selection) only.
Requires Step 2 to be completed first.

Usage:
    python -m src.scripts.drug_class.step3_processor --input data/input.csv --output_dir data/output
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
    select_drug_class,
    needs_llm_selection,
    DrugClassInput,
    SelectionInput,
    DrugSelectionResult,
    Step2Output,
    Step3Output,
    PipelineStatus,
    config,
)
from src.agents.core.storage import LocalStorageClient


@dataclass
class ProcessResult:
    """Result of processing a single abstract."""
    abstract_id: str
    selections: dict = None
    llm_calls: int = 0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.selections is not None and not self.error


def load_abstracts(csv_path: str, limit: int = None) -> tuple[list[DrugClassInput], list[dict], list[str]]:
    """Load abstracts from CSV."""
    inputs = []
    original_rows = []
    fieldnames = []
    
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        
        header_map = {h.lower().strip(): h for h in fieldnames}
        id_col = header_map.get('abstract_id') or header_map.get('id')
        title_col = header_map.get('abstract_title') or header_map.get('title')
        
        for row in reader:
            abstract_id = row.get(id_col, "") if id_col else ""
            abstract_title = row.get(title_col, "") if title_col else ""
            
            if not abstract_id:
                continue
            
            inputs.append(DrugClassInput(
                abstract_id=str(abstract_id),
                abstract_title=str(abstract_title),
            ))
            original_rows.append(row)
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


def process_single(inp: DrugClassInput, storage: LocalStorageClient) -> ProcessResult:
    """Process Step 3 for a single abstract."""
    abstract_id = inp.abstract_id
    
    # Load step2 output
    step2_data = storage.read(f"abstracts/{abstract_id}/step2_output.json")
    if not step2_data:
        return ProcessResult(abstract_id=abstract_id, error="Step 2 output not found")
    
    step2_output = Step2Output(**json.loads(step2_data))
    
    step3_output = Step3Output()
    llm_calls = 0
    
    try:
        for drug_name, extraction_result in step2_output.extractions.items():
            # Check if LLM selection is needed
            if not needs_llm_selection(extraction_result.extraction_details):
                # No LLM needed
                if extraction_result.extraction_details:
                    selected = [extraction_result.extraction_details[0].get("normalized_form", "NA")]
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
                llm_calls += 1
            
            step3_output.mark_success(drug_name, result)
        
        # Save output
        storage.write(
            f"abstracts/{abstract_id}/step3_output.json",
            step3_output.model_dump_json(indent=2)
        )
        
        # Update status
        status_data = storage.read(f"abstracts/{abstract_id}/status.json")
        status = PipelineStatus(**json.loads(status_data)) if status_data else PipelineStatus(abstract_id=abstract_id, abstract_title=inp.abstract_title)
        status.steps["step3_selection"] = {"status": "success", "llm_calls": llm_calls}
        status.last_completed_step = "step3_selection"
        status.total_llm_calls += llm_calls
        storage.write(f"abstracts/{abstract_id}/status.json", json.dumps(status.to_dict(), indent=2, ensure_ascii=False))
        
        return ProcessResult(
            abstract_id=abstract_id,
            selections={d: r.selected_drug_classes for d, r in step3_output.selections.items()},
            llm_calls=llm_calls,
        )
        
    except Exception as e:
        return ProcessResult(abstract_id=abstract_id, error=str(e))


def save_results(results: list[tuple[int, ProcessResult]], original_rows: list[dict], fieldnames: list[str], output_path: str):
    """Save results to CSV."""
    results.sort(key=lambda x: x[0])
    
    output_fieldnames = fieldnames + ["step3_selections", "step3_llm_calls", "step3_error"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for idx, result in results:
            row = dict(original_rows[idx])
            row["step3_selections"] = json.dumps(result.selections, ensure_ascii=False) if result.selections else ""
            row["step3_llm_calls"] = result.llm_calls
            row["step3_error"] = result.error or ""
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Step 3 Processor: Drug Class Selection")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output_dir", default="data/drug_class/output", help="Output directory")
    parser.add_argument("--output_csv", default=None, help="Output CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts")
    parser.add_argument("--parallel_workers", type=int, default=30, help="Parallel workers")
    args = parser.parse_args()
    
    if not args.output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"data/drug_class/output/step3_{timestamp}.csv"
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    print("🧬 Step 3 Processor: Drug Class Selection")
    print("=" * 60)
    print(f"Input:      {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Model:      {config.SELECTION_MODEL}")
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
        future_to_idx = {executor.submit(process_single, inp, storage): i for i, inp in enumerate(inputs)}
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            inp = inputs[idx]
            
            try:
                result = future.result()
                results.append((idx, result))
                status = "✓" if result.success else "✗"
                info = f"{len(result.selections)} drugs" if result.success else result.error
                print(f"[{len(results)}/{len(inputs)}] {inp.abstract_id}: {status} {info}")
            except Exception as e:
                results.append((idx, ProcessResult(abstract_id=inp.abstract_id, error=str(e))))
    
    print()
    save_results(results, original_rows, fieldnames, args.output_csv)
    print(f"✓ Results saved to {args.output_csv}")
    
    successful = sum(1 for _, r in results if r.success)
    print()
    print("📊 Summary:")
    print(f"   Total:   {len(results)}")
    print(f"   Success: {successful}")
    print(f"   Failed:  {len(results) - successful}")


if __name__ == "__main__":
    main()

