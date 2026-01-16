#!/usr/bin/env python3
"""
Drug Extraction Processor

Batch processor for drug extraction using extract_drugs() function.
Uses DrugConfig for model settings.
"""

import argparse
import csv
import json
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.drug import extract_drugs, DrugInput, ExtractionResult, DrugExtractionError, config


@dataclass
class ProcessResult:
    """Result of processing a single abstract."""
    abstract_id: str
    response_json: str = ""
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return bool(self.response_json and not self.error)


def load_abstracts(
    csv_path: str,
    limit: int = None
) -> tuple[list[DrugInput], list[dict], list[str]]:
    """Load abstracts from CSV into DrugInput objects.
    
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
        
        for row in reader:
            abstract_id = row.get(id_col, "") if id_col else ""
            abstract_title = row.get(title_col, "") if title_col else ""
            
            if abstract_id or abstract_title:
                inputs.append(DrugInput(
                    abstract_id=str(abstract_id),
                    abstract_title=str(abstract_title),
                ))
                original_rows.append(row)
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


def process_single(input_data: DrugInput) -> ProcessResult:
    """Process single abstract and return result."""
    try:
        result: ExtractionResult = extract_drugs(input_data)
        
        # Serialize to JSON with aliases (Primary Drugs, etc.)
        response_json = json.dumps(result.model_dump(by_alias=True), indent=2, ensure_ascii=False)
        
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            response_json=response_json,
        )
        
    except DrugExtractionError as e:
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            error=str(e),
        )
    except Exception as e:
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            error=f"Unexpected error: {e}",
        )


def save_results(
    results: list[tuple[int, ProcessResult]],
    original_rows: list[dict],
    input_fieldnames: list[str],
    output_path: str,
    model_name: str
):
    """Save results to CSV with all input columns plus response column."""
    # Sort by original order
    results.sort(key=lambda x: x[0])
    
    response_column = f"{model_name}_response"
    fieldnames = input_fieldnames + [response_column]
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for (idx, result) in results:
            row = dict(original_rows[idx])
            if result.error:
                row[response_column] = json.dumps({"error": result.error}, indent=2)
            else:
                row[response_column] = result.response_json
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Drug Extraction Processor")
    parser.add_argument("--input", default="data/drug/input/abstract_titles.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts to process")
    parser.add_argument("--model_name", default="extraction", help="Model name for column prefix")
    parser.add_argument("--parallel_workers", type=int, default=3, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Auto-generate output path
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/drug/output/extraction_{args.model_name}_{timestamp}.csv"
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("🔬 Drug Extraction Processor")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Model:  {config.EXTRACTION_MODEL}")
    print(f"Limit:  {args.limit or 'all'}")
    print(f"Workers: {args.parallel_workers}")
    print()
    
    # Load abstracts
    print("Loading abstracts...")
    inputs, original_rows, input_fieldnames = load_abstracts(args.input, args.limit)
    print(f"✓ Loaded {len(inputs)} abstracts")
    print()
    
    # Process with parallel workers and intermediate saves
    print("Processing...")
    results: list[tuple[int, ProcessResult]] = []
    save_interval = 5
    last_saved_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        future_to_index = {
            executor.submit(process_single, inp): i
            for i, inp in enumerate(inputs)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            inp = inputs[index]
            
            try:
                result = future.result()
                results.append((index, result))
                
                status = "✓" if result.success else "✗"
                output = result.response_json[:40] if result.success else result.error
                print(f"[{len(results)}/{len(inputs)}] {inp.abstract_id}: {status} {output}...")
                
                # Intermediate save every 5 results
                if len(results) - last_saved_count >= save_interval:
                    save_results(results, original_rows, input_fieldnames, args.output, args.model_name)
                    last_saved_count = len(results)
                    print(f"💾 Intermediate save: {len(results)} results")
                    
            except Exception as e:
                print(f"✗ Error processing {inp.abstract_id}: {e}")
                results.append((index, ProcessResult(abstract_id=inp.abstract_id, error=str(e))))
    
    # Final save
    print()
    save_results(results, original_rows, input_fieldnames, args.output, args.model_name)
    print(f"✓ Results saved to {args.output}")
    
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
