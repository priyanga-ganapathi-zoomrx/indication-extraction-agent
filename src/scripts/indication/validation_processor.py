#!/usr/bin/env python3
"""
Validation Processor for Indication Extraction

Validates previously extracted indications using IndicationValidationAgent.
For local experimentation and QC review.
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.indication import IndicationValidationAgent


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
    limit: int = None
) -> tuple[list[dict], list[dict], list[str]]:
    """Load extraction results from CSV.
    
    Expects CSV with 'model_response' column containing extraction JSON.
    
    Args:
        csv_path: Path to CSV with extraction results
        limit: Optional limit on rows to load
        
    Returns:
        tuple: (extractions, original_rows, fieldnames)
    """
    extractions = []
    original_rows = []
    fieldnames = []
    
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        
        for row in reader:
            # Parse extraction response from model_response column
            response_json = row.get("model_response", "{}")
            try:
                extraction_result = json.loads(response_json) if response_json else {}
            except json.JSONDecodeError:
                extraction_result = {}
            
            extractions.append({
                "abstract_id": row.get("abstract_id", ""),
                "session_title": row.get("session_title", ""),
                "abstract_title": row.get("abstract_title", ""),
                "extraction_result": extraction_result,
            })
            original_rows.append(row)
    
    if limit:
        return extractions[:limit], original_rows[:limit], fieldnames
    return extractions, original_rows, fieldnames


def extract_json_block(content: str) -> str:
    """Extract JSON from markdown code block or return raw content."""
    match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    return match.group(1) if match else content


def pretty_print_json(json_str: str) -> str:
    """Pretty print JSON string."""
    try:
        parsed = json.loads(json_str)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return json_str


def process_single(
    validator: IndicationValidationAgent,
    extraction: dict,
) -> ProcessResult:
    """Validate single extraction and return raw JSON response."""
    abstract_id = extraction["abstract_id"]
    
    try:
        # Skip if extraction result is empty/error
        extraction_result = extraction.get("extraction_result", {})
        if extraction_result.get("error"):
            return ProcessResult(
                abstract_id=abstract_id,
                response_json=json.dumps({
                    "validation_status": "SKIP",
                    "reasoning": f"Extraction had error: {extraction_result.get('error')}"
                }, indent=2),
            )
        
        raw = validator.invoke(
            session_title=extraction["session_title"],
            abstract_title=extraction["abstract_title"],
            extraction_result=extraction_result,
            abstract_id=abstract_id,
        )
        
        messages = raw.get("messages", [])
        if not messages:
            return ProcessResult(abstract_id=abstract_id, error="No messages")
        
        content = messages[-1].content
        if not content:
            return ProcessResult(abstract_id=abstract_id, error="Empty response")
        
        # Extract and pretty print JSON
        json_str = extract_json_block(content)
        pretty_json = pretty_print_json(json_str)
        
        return ProcessResult(
            abstract_id=abstract_id,
            response_json=pretty_json,
        )
        
    except Exception as e:
        return ProcessResult(abstract_id=abstract_id, error=str(e))


def save_results(
    results: list[ProcessResult],
    original_rows: list[dict],
    input_fieldnames: list[str],
    output_path: str,
    model_name: str
):
    """Save results with all input columns plus single validation response column."""
    response_column = f"{model_name}_validation_response"
    fieldnames = input_fieldnames + [response_column]
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for original_row, r in zip(original_rows, results):
            row = dict(original_row)
            if r.error:
                row[response_column] = json.dumps({"error": r.error}, indent=2)
            else:
                row[response_column] = r.response_json
            writer.writerow(row)
    
    print(f"✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate Indication Extractions")
    parser.add_argument("--input", default="data/indication/input/extraction_result.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows to validate")
    parser.add_argument("--model_name", default="model", help="Model name for output column prefix")
    args = parser.parse_args()
    
    # Auto-generate output path
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/indication/output/validated_{args.model_name}_{timestamp}.csv"
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Indication Validation Processor")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Limit:  {args.limit or 'all'}")
    print()
    
    # Load extractions
    print("Loading extraction results...")
    extractions, original_rows, input_fieldnames = load_extractions(args.input, args.limit)
    print(f"✓ Loaded {len(extractions)} extractions")
    print()
    
    # Initialize validator
    print("Initializing validation agent...")
    validator = IndicationValidationAgent()
    print("✓ Agent ready")
    print()
    
    # Process
    print("Validating...")
    results = []
    for i, extraction in enumerate(extractions, 1):
        abstract_id = extraction["abstract_id"]
        abstract_title = extraction["abstract_title"][:50]
        print(f"[{i}/{len(extractions)}] {abstract_id}: {abstract_title}...")
        
        result = process_single(validator, extraction)
        results.append(result)
        
        # Extract status from response for logging
        status = "✓" if result.success else "✗"
        if result.success:
            try:
                parsed = json.loads(result.response_json)
                validation_status = parsed.get("validation_status", "?")
                output = f"{validation_status}"
            except:
                output = "parsed"
        else:
            output = result.error[:40] if result.error else "error"
        print(f"         {status} {output}")
    
    # Save
    print()
    save_results(results, original_rows, input_fieldnames, args.output, args.model_name)
    
    # Summary
    successful = sum(1 for r in results if r.success)
    print()
    print("📊 Summary:")
    print(f"   Total:   {len(results)}")
    print(f"   Success: {successful}")
    print(f"   Failed:  {len(results) - successful}")
    if results:
        print(f"   Rate:    {successful/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()
