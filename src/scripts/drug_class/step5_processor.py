#!/usr/bin/env python3
"""
Step 5 Processor: Consolidation

Processes multiple abstracts through Step 5 (Consolidation) only.
Requires Steps 3 and 4 to be completed first.

Usage:
    python -m src.scripts.drug_class.step5_processor --input data/input.csv --output_dir data/output
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
    consolidate_drug_classes,
    DrugClassInput,
    ConsolidationInput,
    Step3Output,
    Step4Output,
    Step5Output,
    PipelineStatus,
    config,
)
from src.agents.drug_class.pipeline import LocalStorageClient


@dataclass
class ProcessResult:
    """Result of processing a single abstract."""
    abstract_id: str
    refined_classes: list[str] = None
    removed_classes: list[str] = None
    llm_calls: int = 0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.refined_classes is not None and not self.error


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
    """Process Step 5 for a single abstract."""
    abstract_id = inp.abstract_id
    
    # Load step3 and step4 outputs
    step3_data = storage.read(f"abstracts/{abstract_id}/step3_output.json")
    step4_data = storage.read(f"abstracts/{abstract_id}/step4_output.json")
    
    if not step3_data:
        return ProcessResult(abstract_id=abstract_id, error="Step 3 output not found")
    if not step4_data:
        return ProcessResult(abstract_id=abstract_id, error="Step 4 output not found")
    
    step3_output = Step3Output(**json.loads(step3_data))
    step4_output = Step4Output(**json.loads(step4_data))
    
    llm_calls = 0
    
    try:
        # Build drug selections
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
            llm_calls = 1
        
        # Save output
        storage.write(
            f"abstracts/{abstract_id}/step5_output.json",
            step5_output.model_dump_json(indent=2)
        )
        
        # Update status
        status_data = storage.read(f"abstracts/{abstract_id}/status.json")
        status = PipelineStatus(**json.loads(status_data)) if status_data else PipelineStatus(abstract_id=abstract_id, abstract_title=inp.abstract_title)
        status.steps["step5_consolidation"] = {"status": "success", "llm_calls": llm_calls}
        status.last_completed_step = "step5_consolidation"
        status.pipeline_status = "success"
        status.total_llm_calls += llm_calls
        storage.write(f"abstracts/{abstract_id}/status.json", json.dumps(status.to_dict(), indent=2))
        
        return ProcessResult(
            abstract_id=abstract_id,
            refined_classes=step5_output.refined_explicit_classes,
            removed_classes=step5_output.removed_classes,
            llm_calls=llm_calls,
        )
        
    except Exception as e:
        return ProcessResult(abstract_id=abstract_id, error=str(e))


def save_results(results: list[tuple[int, ProcessResult]], original_rows: list[dict], fieldnames: list[str], output_path: str):
    """Save results to CSV."""
    results.sort(key=lambda x: x[0])
    
    output_fieldnames = fieldnames + ["step5_refined_classes", "step5_removed_classes", "step5_llm_calls", "step5_error"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for idx, result in results:
            row = dict(original_rows[idx])
            row["step5_refined_classes"] = json.dumps(result.refined_classes) if result.refined_classes else ""
            row["step5_removed_classes"] = json.dumps(result.removed_classes) if result.removed_classes else ""
            row["step5_llm_calls"] = result.llm_calls
            row["step5_error"] = result.error or ""
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Step 5 Processor: Consolidation")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output_dir", default="data/drug_class/output", help="Output directory")
    parser.add_argument("--output_csv", default=None, help="Output CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts")
    parser.add_argument("--parallel_workers", type=int, default=30, help="Parallel workers")
    args = parser.parse_args()
    
    if not args.output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"data/drug_class/output/step5_{timestamp}.csv"
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    print("🧬 Step 5 Processor: Consolidation")
    print("=" * 60)
    print(f"Input:      {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Model:      {config.CONSOLIDATION_MODEL}")
    print(f"Limit:      {args.limit or 'all'}")
    print(f"Workers:    {args.parallel_workers}")
    print()
    
    storage = LocalStorageClient(base_path=args.output_dir)
    
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
                info = f"refined={result.refined_classes}, removed={len(result.removed_classes or [])}" if result.success else result.error
                print(f"[{len(results)}/{len(inputs)}] {inp.abstract_id}: {status} {info[:60]}...")
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

