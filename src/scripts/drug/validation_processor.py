#!/usr/bin/env python3
"""
Drug Validation Processor

Batch processor for drug validation using validate_drugs() function.
Takes extraction output as input.
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

from src.agents.drug import validate_drugs, ValidationInput, ValidationResult, DrugValidationError, config


@dataclass
class ProcessResult:
    """Result of validating a single extraction."""
    abstract_id: str
    response_json: str = ""
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return bool(self.response_json and not self.error)


def load_extractions(
    csv_path: str,
    response_column: str,
    limit: int = None
) -> tuple[list[ValidationInput], list[dict], list[str]]:
    """Load extraction results from CSV.
    
    Args:
        csv_path: Path to extraction output CSV
        response_column: Name of column containing extraction response JSON
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
        
        # Find column names (case-insensitive)
        header_map = {h.lower().strip(): h for h in fieldnames}
        id_col = header_map.get('abstract_id') or header_map.get('id')
        title_col = header_map.get('abstract_title') or header_map.get('title')
        
        # Find extraction response column
        extraction_col = None
        for col in fieldnames:
            if response_column.lower() in col.lower():
                extraction_col = col
                break
        
        if not extraction_col:
            raise ValueError(f"Could not find extraction response column matching '{response_column}' in CSV")
        
        for row in reader:
            abstract_id = row.get(id_col, "") if id_col else ""
            abstract_title = row.get(title_col, "") if title_col else ""
            extraction_response = row.get(extraction_col, "{}")
            
            # Parse extraction response JSON
            try:
                extraction_result = json.loads(extraction_response) if extraction_response else {}
            except json.JSONDecodeError:
                extraction_result = {"error": "Failed to parse extraction response"}
            
            inputs.append(ValidationInput(
                abstract_id=str(abstract_id),
                abstract_title=str(abstract_title),
                extraction_result=extraction_result,
            ))
            original_rows.append(row)
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


def process_single(input_data: ValidationInput) -> ProcessResult:
    """Validate single extraction and return result."""
    # Skip if extraction had error
    if input_data.extraction_result.get("error"):
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            response_json=json.dumps({
                "validation_status": "SKIP",
                "validation_reasoning": f"Extraction had error: {input_data.extraction_result.get('error')}"
            }, indent=2),
        )
    
    try:
        result: ValidationResult = validate_drugs(input_data)
        
        # Serialize to JSON
        response_json = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
        
        return ProcessResult(
            abstract_id=input_data.abstract_id,
            response_json=response_json,
        )
        
    except DrugValidationError as e:
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
    """Save results to CSV with all input columns plus validation response column."""
    # Sort by original order
    results.sort(key=lambda x: x[0])
    
    response_column = f"{model_name}_validation_response"
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
    parser = argparse.ArgumentParser(description="Drug Validation Processor")
    parser.add_argument("--input", default="data/drug/input/extraction_result.csv", help="Input CSV from extraction processor")
    parser.add_argument("--output", default=None)
    parser.add_argument("--extraction_column", default="extraction_response", 
                        help="Column name containing extraction response JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows to validate")
    parser.add_argument("--model_name", default="validation", help="Model name for column prefix")
    parser.add_argument("--parallel_workers", type=int, default=3, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Auto-generate output path
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/drug/output/validation_{args.model_name}_{timestamp}.csv"
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("✅ Drug Validation Processor")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Model:  {config.VALIDATION_MODEL}")
    print(f"Extraction column: {args.extraction_column}")
    print(f"Limit:  {args.limit or 'all'}")
    print(f"Workers: {args.parallel_workers}")
    print()
    
    # Load extractions
    print("Loading extraction results...")
    inputs, original_rows, input_fieldnames = load_extractions(
        args.input, args.extraction_column, args.limit
    )
    print(f"✓ Loaded {len(inputs)} extraction results")
    print()
    
    # Process with parallel workers and intermediate saves
    print("Validating...")
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
                
                # Extract validation status for logging
                status_icon = "✓" if result.success else "✗"
                if result.success:
                    try:
                        parsed = json.loads(result.response_json)
                        validation_status = parsed.get("validation_status", "?")
                        output = validation_status
                    except:
                        output = "parsed"
                else:
                    output = result.error[:30] if result.error else "error"
                
                print(f"[{len(results)}/{len(inputs)}] {inp.abstract_id}: {status_icon} {output}")
                
                # Intermediate save every 5 results
                if len(results) - last_saved_count >= save_interval:
                    save_results(results, original_rows, input_fieldnames, args.output, args.model_name)
                    last_saved_count = len(results)
                    print(f"💾 Intermediate save: {len(results)} results")
                    
            except Exception as e:
                print(f"✗ Error validating {inp.abstract_id}: {e}")
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
