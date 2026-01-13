#!/usr/bin/env python3
"""
Batch Processor for Indication Extraction

Simplified processor using new IndicationAgent.
For local experimentation and prompt testing.
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.indication import IndicationAgent, IndicationInput


@dataclass
class ProcessResult:
    """Result of processing a single abstract."""
    abstract_id: str
    response_json: str = ""
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return bool(self.response_json and not self.error)


def load_abstracts(csv_path: str, limit: int = None) -> tuple[list[IndicationInput], list[dict], list[str]]:
    """Load abstracts from CSV into IndicationInput objects.
    
    Returns:
        tuple: (inputs, original_rows, fieldnames)
    """
    inputs = []
    original_rows = []
    fieldnames = []
    
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        
        for row in reader:
            inputs.append(IndicationInput(
                abstract_id=row.get("abstract_id", row.get("abstract_id", "")),
                abstract_title=row.get("abstract_title", row.get("abstract Title", "")),
                session_title=row.get("session_title", row.get("Session title", "")),
            ))
            original_rows.append(row)
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


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


def process_single(agent: IndicationAgent, input: IndicationInput) -> ProcessResult:
    """Process single abstract and return the raw LLM JSON response."""
    try:
        raw = agent.invoke(input.abstract_title, input.session_title, input.abstract_id)
        
        messages = raw.get("messages", [])
        if not messages:
            return ProcessResult(abstract_id=input.abstract_id, error="No messages")
        
        content = messages[-1].content
        if not content:
            return ProcessResult(abstract_id=input.abstract_id, error="Empty response")
        
        # Extract and pretty print JSON
        json_str = extract_json_block(content)
        pretty_json = pretty_print_json(json_str)
        
        return ProcessResult(
            abstract_id=input.abstract_id,
            response_json=pretty_json,
        )
        
    except Exception as e:
        return ProcessResult(abstract_id=input.abstract_id, error=str(e))


def save_results(
    results: list[ProcessResult], 
    original_rows: list[dict],
    input_fieldnames: list[str],
    output_path: str, 
    model_name: str
):
    """Save results to CSV with all input columns plus single model response column."""
    # Single output column with full JSON response
    response_column = f"{model_name}_response"
    fieldnames = input_fieldnames + [response_column]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for original_row, r in zip(original_rows, results):
            row = dict(original_row)
            # Add response JSON or error
            if r.error:
                row[response_column] = json.dumps({"error": r.error}, indent=2)
            else:
                row[response_column] = r.response_json
            writer.writerow(row)
    
    print(f"✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch Indication Extraction")
    parser.add_argument("--input", default="data/indication/input/abstract_titles.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts to process")
    parser.add_argument("--model_name", default="model", help="Model name for column prefix")
    args = parser.parse_args()
    
    # Auto-generate output path
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/indication/output/batch_{args.model_name}_{timestamp}.csv"
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print("🏭 Indication Extraction Batch Processor")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Limit:  {args.limit or 'all'}")
    print()
    
    # Load abstracts
    print("Loading abstracts...")
    inputs, original_rows, input_fieldnames = load_abstracts(args.input, args.limit)
    print(f"✓ Loaded {len(inputs)} abstracts")
    print()
    
    # Initialize agent
    print("Initializing agent...")
    agent = IndicationAgent()
    print("✓ Agent ready")
    print()
    
    # Process
    print("Processing...")
    results = []
    for i, inp in enumerate(inputs, 1):
        print(f"[{i}/{len(inputs)}] {inp.abstract_id}: {inp.abstract_title[:50]}...")
        result = process_single(agent, inp)
        results.append(result)
        
        status = "✓" if result.success else "✗"
        output = result.response_json[:40] if result.success else result.error
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
    print(f"   Rate:    {successful/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()
