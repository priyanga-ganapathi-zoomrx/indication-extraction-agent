#!/usr/bin/env python3
"""
Step 4 Processor: Explicit Drug Class Extraction

Processes multiple abstracts through Step 4 (Explicit Drug Class Extraction) only.
Can be run independently (doesn't require previous steps).

Usage:
    python -m src.scripts.drug_class.step4_processor --input data/input.csv --output_dir data/output
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
    extract_explicit_classes,
    DrugClassInput,
    ExplicitExtractionInput,
    PipelineStatus,
    config,
)
from src.agents.core.storage import LocalStorageClient


@dataclass
class ProcessResult:
    """Result of processing a single abstract."""
    abstract_id: str
    explicit_classes: list[str] = None
    llm_calls: int = 0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.explicit_classes is not None and not self.error


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


def process_single(inp: DrugClassInput, storage: LocalStorageClient) -> ProcessResult:
    """Process Step 4 for a single abstract."""
    abstract_id = inp.abstract_id
    llm_calls = 0
    
    try:
        step4_output = extract_explicit_classes(ExplicitExtractionInput(
            abstract_id=abstract_id,
            abstract_title=inp.abstract_title,
        ))
        
        # Count LLM call only if title was non-empty
        if inp.abstract_title and inp.abstract_title.strip():
            llm_calls = 1
        
        # Save output
        storage.upload_json(
            f"abstracts/{abstract_id}/step4_output.json",
            step4_output.model_dump()
        )
        
        # Update status
        try:
            status_data = storage.download_json(f"abstracts/{abstract_id}/status.json")
            status = PipelineStatus(**status_data)
        except FileNotFoundError:
            status = PipelineStatus(abstract_id=abstract_id, abstract_title=inp.abstract_title)
        status.steps["step4_explicit"] = {"status": "success", "llm_calls": llm_calls}
        status.last_completed_step = "step4_explicit"
        status.total_llm_calls += llm_calls
        status.last_updated = datetime.utcnow().isoformat() + "Z"
        storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
        
        return ProcessResult(
            abstract_id=abstract_id,
            explicit_classes=step4_output.explicit_drug_classes,
            llm_calls=llm_calls,
        )
        
    except Exception as e:
        return ProcessResult(abstract_id=abstract_id, error=str(e))


def save_results(results: list[tuple[int, ProcessResult]], original_rows: list[dict], fieldnames: list[str], output_path: str):
    """Save results to CSV."""
    results.sort(key=lambda x: x[0])
    
    output_fieldnames = fieldnames + ["step4_explicit_classes", "step4_llm_calls", "step4_error"]
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for idx, result in results:
            row = dict(original_rows[idx])
            row["step4_explicit_classes"] = json.dumps(result.explicit_classes, ensure_ascii=False) if result.explicit_classes else ""
            row["step4_llm_calls"] = result.llm_calls
            row["step4_error"] = result.error or ""
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Step 4 Processor: Explicit Drug Class Extraction")
    parser.add_argument("--input", default="data/drug_class/input/drugs.csv", help="Input CSV file")
    parser.add_argument("--output_dir", default="data/drug_class/output", help="Output directory")
    parser.add_argument("--output_csv", default=None, help="Output CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts")
    parser.add_argument("--parallel_workers", type=int, default=25, help="Parallel workers")
    args = parser.parse_args()
    
    if not args.output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"data/drug_class/output/step4_{timestamp}.csv"
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    print("🧬 Step 4 Processor: Explicit Drug Class Extraction")
    print("=" * 60)
    print(f"Input:      {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Model:      {config.EXPLICIT_MODEL}")
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
                info = str(result.explicit_classes) if result.success else result.error
                print(f"[{len(results)}/{len(inputs)}] {inp.abstract_id}: {status} {info[:50]}...")
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

