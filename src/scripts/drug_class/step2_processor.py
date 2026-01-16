#!/usr/bin/env python3
"""
Step 2 Processor: Drug Class Extraction

Processes multiple abstracts through Step 2 (Drug Class Extraction) only.
Requires Step 1 to be completed first.

Usage:
    python -m src.scripts.drug_class.step2_processor --input data/input.csv --output_dir data/output
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
    fetch_search_results,
    extract_with_tavily,
    extract_with_grounded,
    DrugClassInput,
    DrugClassExtractionInput,
    Step1Output,
    Step2Output,
    PipelineStatus,
    config,
)
from src.agents.core.storage import LocalStorageClient


@dataclass
class ProcessResult:
    """Result of processing a single abstract."""
    abstract_id: str
    extractions: dict = None
    llm_calls: int = 0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.extractions is not None and not self.error


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
        firms_col = header_map.get('firms') or header_map.get('firm') or header_map.get('sponsor')
        
        for row in reader:
            abstract_id = row.get(id_col, "") if id_col else ""
            abstract_title = row.get(title_col, "") if title_col else ""
            
            if not abstract_id or not abstract_title:
                continue
            
            firms = _parse_list(row.get(firms_col, "") if firms_col else "")
            
            inputs.append(DrugClassInput(
                abstract_id=str(abstract_id),
                abstract_title=str(abstract_title),
                firms=firms,
            ))
            original_rows.append(row)
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


def _parse_list(value: str) -> list[str]:
    """Parse list from JSON array or comma-separated string."""
    if not value or not value.strip():
        return []
    value = value.strip()
    if value.startswith('['):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(d).strip() for d in parsed if d and str(d).strip()]
        except json.JSONDecodeError:
            pass
    return [d.strip() for d in value.replace(';', ',').split(',') if d.strip()]


def process_single(inp: DrugClassInput, storage: LocalStorageClient) -> ProcessResult:
    """Process Step 2 for a single abstract."""
    abstract_id = inp.abstract_id
    
    # Load step1 output
    try:
        step1_data = storage.download_json(f"abstracts/{abstract_id}/step1_output.json")
    except FileNotFoundError:
        return ProcessResult(abstract_id=abstract_id, error="Step 1 output not found")
    
    step1_output = Step1Output(**step1_data)
    all_components = step1_output.get_all_components()
    
    if not all_components:
        return ProcessResult(abstract_id=abstract_id, error="No components from Step 1")
    
    step2_output = Step2Output()
    llm_calls = 0
    
    try:
        for drug in all_components:
            # Fetch search results
            drug_results, firm_results = fetch_search_results(drug, inp.firms, storage)
            
            # Try Tavily extraction
            result = extract_with_tavily(DrugClassExtractionInput(
                abstract_id=abstract_id,
                abstract_title=inp.abstract_title,
                drug=drug,
                firms=inp.firms,
                drug_class_results=drug_results,
                firm_search_results=firm_results,
            ))
            llm_calls += 1
            
            # Fallback to grounded search if needed
            if not result.drug_classes or result.drug_classes == ["NA"]:
                result = extract_with_grounded(DrugClassExtractionInput(
                    abstract_id=abstract_id,
                    abstract_title=inp.abstract_title,
                    drug=drug,
                    firms=inp.firms,
                    drug_class_results=drug_results,
                    firm_search_results=firm_results,
                ))
                llm_calls += 1
            
            step2_output.mark_success(drug, result)
        
        # Save output
        storage.upload_json(
            f"abstracts/{abstract_id}/step2_output.json",
            step2_output.model_dump()
        )
        
        # Update status
        try:
            status_data = storage.download_json(f"abstracts/{abstract_id}/status.json")
            status = PipelineStatus(**status_data)
        except FileNotFoundError:
            status = PipelineStatus(abstract_id=abstract_id, abstract_title=inp.abstract_title)
        status.steps["step2_extraction"] = {"status": "success", "llm_calls": llm_calls}
        status.last_completed_step = "step2_extraction"
        status.total_llm_calls += llm_calls
        status.last_updated = datetime.utcnow().isoformat() + "Z"
        storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
        
        return ProcessResult(
            abstract_id=abstract_id,
            extractions={d: r.drug_classes for d, r in step2_output.extractions.items()},
            llm_calls=llm_calls,
        )
        
    except Exception as e:
        return ProcessResult(abstract_id=abstract_id, error=str(e))


def save_results(results: list[tuple[int, ProcessResult]], original_rows: list[dict], fieldnames: list[str], output_path: str):
    """Save results to CSV."""
    results.sort(key=lambda x: x[0])
    
    output_fieldnames = fieldnames + ["step2_extractions", "step2_llm_calls", "step2_error"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        
        for idx, result in results:
            row = dict(original_rows[idx])
            row["step2_extractions"] = json.dumps(result.extractions, ensure_ascii=False) if result.extractions else ""
            row["step2_llm_calls"] = result.llm_calls
            row["step2_error"] = result.error or ""
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Step 2 Processor: Drug Class Extraction")
    parser.add_argument("--input", default="data/drug_class/input/drugs.csv", help="Input CSV file")
    parser.add_argument("--output_dir", default="data/drug_class/output", help="Output directory")
    parser.add_argument("--output_csv", default=None, help="Output CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts")
    parser.add_argument("--parallel_workers", type=int, default=10, help="Parallel workers")
    args = parser.parse_args()
    
    if not args.output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = f"data/drug_class/output/step2_{timestamp}.csv"
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    print("🧬 Step 2 Processor: Drug Class Extraction")
    print("=" * 60)
    print(f"Input:      {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Model:      {config.EXTRACTION_MODEL}")
    print(f"Grounded:   {config.GROUNDED_MODEL}")
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
                info = f"{len(result.extractions)} drugs" if result.success else result.error
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

