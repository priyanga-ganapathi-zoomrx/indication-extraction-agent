#!/usr/bin/env python3
"""
Batch Processor for Regimen Identification

This script processes multiple drugs to identify if they are regimens
and extract their component drugs.

Features:
- Reads from input CSV (abstract_id, abstract_title, drug_name, etc.)
- Handles multiple drugs per row (comma/semicolon separated)
- For each drug, identifies if it's a regimen and extracts components
- Outputs all original columns plus:
  - flattened_components: Combined array of all components from all drugs
  - grouped_components: Object with drug name as key and components array as value
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regimen_identification_agent import RegimenIdentificationAgent


def load_rows_from_csv(csv_path: str, max_entries: int = None, randomize: bool = False) -> List[Dict]:
    """Load rows from CSV file.

    Each row may contain multiple drugs (comma/semicolon separated).
    Returns row-level entries with drug list and original row data.

    Args:
        csv_path: Path to the CSV file
        max_entries: Maximum number of CSV rows to return
        randomize: Whether to randomize the selection

    Returns:
        List of dictionaries with row data
    """
    entries = []

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return entries

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)

            # Store original fieldnames
            original_fieldnames = reader.fieldnames or []

            # Normalize headers mapping
            header_map = {h.lower().strip(): h for h in original_fieldnames}

            # Find the correct header names for our required fields
            drug_name_col = header_map.get('drug_name') or header_map.get('drug name') or header_map.get('drug')
            abstract_id_col = header_map.get('abstract_id') or header_map.get('id')
            abstract_title_col = header_map.get('abstract_title') or header_map.get('title')

            if not drug_name_col:
                print(f"Warning: Could not find drug_name column in {csv_path}")
                print(f"Available columns: {list(header_map.keys())}")
                return entries

            for row_id, row in enumerate(reader, start=1):
                # Parse drug_name as list (comma-separated or semicolon-separated)
                raw_drug_name = row.get(drug_name_col, '').strip()
                if raw_drug_name:
                    individual_drugs = [d.strip() for d in raw_drug_name.replace(';', ',').split(',') if d.strip()]
                else:
                    individual_drugs = []

                if not individual_drugs:
                    continue

                # Get abstract fields
                abstract_id = row.get(abstract_id_col, '').strip() if abstract_id_col else ''
                abstract_title = row.get(abstract_title_col, '').strip() if abstract_title_col else ''

                # Store original row data (all columns)
                original_row = {col: row.get(col, '') for col in original_fieldnames}

                entries.append({
                    'row_id': row_id,
                    'original_row': original_row,
                    'original_fieldnames': original_fieldnames,
                    'individual_drugs': individual_drugs,
                    'abstract_id': abstract_id,
                    'abstract_title': abstract_title,
                })

        print(f"✓ Loaded {len(entries)} CSV rows")

        if randomize and max_entries and len(entries) > max_entries:
            import random
            entries = random.sample(entries, max_entries)
        elif max_entries and len(entries) > max_entries:
            entries = entries[:max_entries]

        return entries

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def process_single_drug(
    drug: str,
    abstract_title: str,
    abstract_id: str,
    agent: RegimenIdentificationAgent,
) -> Dict[str, Any]:
    """Process a single drug and return its components.

    Args:
        drug: Drug name
        abstract_title: Abstract title
        abstract_id: Abstract ID
        agent: Initialized RegimenIdentificationAgent

    Returns:
        Dictionary with components for this drug
    """
    # Invoke the agent
    result = agent.invoke(
        drug=drug,
        abstract_title=abstract_title,
        abstract_id=abstract_id,
    )

    # Parse the response
    parsed = agent.parse_response(result)
    parsed['llm_calls'] = result.get('llm_calls', 0)

    return parsed


def process_single_row(
    entry: Dict,
    agent: RegimenIdentificationAgent,
    index: int,
) -> Dict:
    """Process a single row (may contain multiple drugs) and return grouped results.

    Args:
        entry: Row dictionary with individual_drugs list and original row data
        agent: Initialized RegimenIdentificationAgent
        index: Index of the row for logging

    Returns:
        Dictionary with grouped processing result
    """
    individual_drugs = entry['individual_drugs']
    abstract_title = entry['abstract_title']
    abstract_id = entry['abstract_id']
    original_row = entry['original_row']
    original_fieldnames = entry['original_fieldnames']

    print(f"[{index}] Processing row with drugs: {individual_drugs}")

    # Process each drug individually
    grouped_components = {}  # drug -> [components]
    all_components = []  # Flattened list of all components
    success_flags = []
    total_llm_calls = 0

    for drug in individual_drugs:
        print(f"  - Processing drug: {drug}")

        try:
            result = process_single_drug(
                drug=drug,
                abstract_title=abstract_title,
                abstract_id=abstract_id,
                agent=agent,
            )

            # Store grouped components
            components = result.get("components", [drug])
            if not components:
                components = [drug]  # Fallback to original drug if empty
            
            grouped_components[drug] = components
            success_flags.append(result.get("success", False))
            total_llm_calls += result.get("llm_calls", 0)

            # Add to flattened list (avoid duplicates)
            for comp in components:
                if comp and comp not in all_components:
                    all_components.append(comp)

        except Exception as e:
            print(f"  Error processing drug {drug}: {e}")
            grouped_components[drug] = [drug]  # Fallback to original drug
            success_flags.append(False)

    # If no components found, use original drugs
    if not all_components:
        all_components = individual_drugs.copy()

    # Determine overall success
    overall_success = any(success_flags) if success_flags else False

    # Build output row (preserve original columns + add new ones)
    output_row = original_row.copy()
    output_row.update({
        "flattened_components": json.dumps(all_components),
        "grouped_components": json.dumps(grouped_components, indent=2),
        "success": overall_success,
        "llm_calls": total_llm_calls,
    })

    return output_row


def process_rows_batch(
    entries: List[Dict],
    agent: RegimenIdentificationAgent,
    output_file: str = None,
    max_workers: int = 3,
) -> pd.DataFrame:
    """Process a batch of rows and return results DataFrame.

    Args:
        entries: List of row dictionaries
        agent: Initialized RegimenIdentificationAgent
        output_file: Optional output file path to save intermediate results
        max_workers: Number of parallel workers (default: 3)

    Returns:
        DataFrame with processing results
    """
    print(f"Processing {len(entries)} rows (using {max_workers} parallel threads)")

    results = []
    write_lock = threading.Lock()

    # Get original fieldnames from first entry
    original_fieldnames = entries[0]['original_fieldnames'] if entries else []
    
    # Define output fieldnames (original + new columns)
    output_fieldnames = list(original_fieldnames) + [
        'flattened_components', 'grouped_components', 'success', 'llm_calls'
    ]

    # Open CSV file for incremental writing
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=output_fieldnames, extrasaction='ignore')
        writer.writeheader()

        def write_result(result: Dict, idx: int):
            """Write a single result to CSV (thread-safe)."""
            with write_lock:
                writer.writerow(result)
                csvfile.flush()
                print(f"  ✓ Row {idx} saved to CSV")

        if max_workers == 1:
            # Sequential processing
            for i, entry in enumerate(entries, 1):
                result = process_single_row(entry, agent, i)
                results.append((i, result))
                write_result(result, i)
        else:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        process_single_row,
                        entry,
                        agent,
                        i,
                    ): i
                    for i, entry in enumerate(entries, 1)
                }

                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results.append((idx, result))
                        write_result(result, idx)
                    except Exception as e:
                        print(f"Error in thread execution for row {idx}: {e}")

    # Sort results by original index to maintain order
    results.sort(key=lambda x: x[0])
    results = [result for _, result in results]

    # Create final DataFrame
    results_df = pd.DataFrame(results)

    print(f"Final results saved to {output_file}")

    return results_df


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(description='Batch Process Regimen Identification')
    parser.add_argument('--input_file', default='data/drug_class_input_150.csv',
                        help='Input CSV file with drugs (default: data/drug_class_input_150.csv)')
    parser.add_argument('--output_file', default=None,
                        help='Output CSV file (default: auto-generated)')
    parser.add_argument('--model', default='gemini/gemini-3-pro-preview',
                        help='LLM model to use (default: gemini/gemini-3-pro-preview)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='LLM temperature (default: 0.0)')
    parser.add_argument('--max_tokens', type=int, default=50000,
                        help='LLM max tokens (default: 50000)')
    parser.add_argument('--max_entries', type=int, default=None,
                        help='Maximum CSV rows to process (default: all)')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='Parallel workers (default: 1)')
    parser.add_argument('--randomize', action='store_true',
                        help='Randomize row selection')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model.replace("/", "-")
        args.output_file = f"data/regimen_identification_{model_safe}_{timestamp}.csv"

    print("💊 Regimen Identification Batch Processor")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Max entries: {args.max_entries or 'all'}")
    print(f"Max workers: {args.max_workers}")
    print(f"Randomize: {args.randomize}")
    print()

    # Load input CSV
    print("Loading input CSV...")
    entries = load_rows_from_csv(args.input_file, args.max_entries, args.randomize)
    if not entries:
        print("No entries found in input CSV.")
        return

    # Initialize agent
    print(f"\nInitializing Regimen Identification Agent ({args.model})...")
    agent = RegimenIdentificationAgent(
        agent_name="BatchRegimenIdentificationProcessor",
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print("✓ Agent initialized")

    # Process entries
    print(f"\nProcessing {len(entries)} rows...")
    print("-" * 60)

    results_df = process_rows_batch(
        entries=entries,
        agent=agent,
        output_file=args.output_file,
        max_workers=args.max_workers,
    )

    # Summary
    print(f"\n{'=' * 60}")
    total_processed = len(results_df)
    successful = results_df['success'].sum() if 'success' in results_df.columns else 0
    success_rate = (successful / total_processed * 100) if total_processed > 0 else 0

    print("📊 Summary:")
    print(f"  Total rows processed: {total_processed}")
    print(f"  Successful: {int(successful)}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

