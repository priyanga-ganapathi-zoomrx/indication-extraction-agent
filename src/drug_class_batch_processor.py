#!/usr/bin/env python3
"""
Batch Processor for Drug Class Extraction Agent

This script processes multiple drugs using the drug class extraction agent
and saves results to CSV format for analysis.

Features:
- Reads abstract_id, abstract_title, drug_name, firm, full_abstract from input CSV
- Handles multiple drugs per row (comma/semicolon separated)
- Groups results by drug with flattened drug_classes column
- Preserves all original input columns in output
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drug_class_agent import DrugClassAgent


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

            # Normalize headers mapping
            if reader.fieldnames:
                header_map = {h.lower().strip(): h for h in reader.fieldnames}
            else:
                header_map = {}

            # Find the correct header names for our required fields
            drug_name_col = header_map.get('drug_name') or header_map.get('drug name') or header_map.get('drug')
            firm_col = header_map.get('firm') or header_map.get('company') or header_map.get('sponsor')
            abstract_id_col = header_map.get('abstract_id') or header_map.get('id')
            abstract_title_col = header_map.get('abstract_title') or header_map.get('title')
            full_abstract_col = header_map.get('full_abstract') or header_map.get('abstract')
            ground_truth_col = header_map.get('drug class - ground truth (manually extracted)') or header_map.get('ground_truth')

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

                # Parse firm as list (comma-separated or semicolon-separated)
                raw_firm = row.get(firm_col, '').strip() if firm_col else ''
                if raw_firm:
                    firms = [f.strip() for f in raw_firm.replace(';', ',').split(',') if f.strip()]
                else:
                    firms = []

                # Get abstract fields
                abstract_id = row.get(abstract_id_col, '').strip() if abstract_id_col else ''
                abstract_title = row.get(abstract_title_col, '').strip() if abstract_title_col else ''
                full_abstract = row.get(full_abstract_col, '').strip() if full_abstract_col else ''
                ground_truth = row.get(ground_truth_col, '').strip() if ground_truth_col else ''

                # Store original row data for output
                original_row = {
                    'abstract_id': abstract_id,
                    'abstract_title': abstract_title,
                    'drug_name': raw_drug_name,  # Original (may have multiple)
                    'Drug Class - Ground truth (Manually extracted)': ground_truth,
                    'firm': raw_firm,
                    'full_abstract': full_abstract,
                }

                entries.append({
                    'row_id': row_id,
                    'original_row': original_row,
                    'individual_drugs': individual_drugs,
                    'firms': firms,
                    'abstract_id': abstract_id,
                    'abstract_title': abstract_title,
                    'full_abstract': full_abstract,
                })

        print(f"  Loaded {len(entries)} CSV rows")

        if randomize and max_entries and len(entries) > max_entries:
            import random
            entries = random.sample(entries, max_entries)
        elif max_entries and len(entries) > max_entries:
            entries = entries[:max_entries]

        return entries

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def extract_drug_class_from_response(result: Dict) -> Dict[str, Any]:
    """Extract drug class data from agent response.

    Args:
        result: Agent invocation result

    Returns:
        Dictionary with extracted fields
    """
    try:
        drug_class_result = result.get('drug_class_result', {})

        if not drug_class_result:
            return {
                'drug_classes': ['NA'],
                'content_urls': ['NA'],
                'steps_taken': [],
                'success': False,
            }

        # Check for error
        if 'error' in drug_class_result:
            return {
                'drug_classes': drug_class_result.get('drug_classes', ['NA']),
                'content_urls': drug_class_result.get('content_urls', ['NA']),
                'steps_taken': drug_class_result.get('steps_taken', []),
                'success': False,
                'error': drug_class_result.get('error', ''),
            }

        drug_classes = drug_class_result.get('drug_classes', ['NA'])
        content_urls = drug_class_result.get('content_urls', ['NA'])
        steps_taken = drug_class_result.get('steps_taken', [])

        # Check if extraction was successful (not just NA)
        success = drug_classes != ['NA'] and len(drug_classes) > 0

        return {
            'drug_classes': drug_classes,
            'content_urls': content_urls,
            'steps_taken': steps_taken,
            'success': success,
        }

    except Exception as e:
        print(f"Error extracting drug class: {e}")
        return {
            'drug_classes': ['NA'],
            'content_urls': ['NA'],
            'steps_taken': [],
            'success': False,
            'error': str(e),
        }


def process_single_row(entry: Dict, agent: DrugClassAgent, index: int) -> Dict:
    """Process a single row (may contain multiple drugs) and return grouped results.

    Args:
        entry: Row dictionary with individual_drugs list and original row data
        agent: Initialized DrugClassAgent
        index: Index of the row for logging

    Returns:
        Dictionary with grouped processing result
    """
    individual_drugs = entry['individual_drugs']
    firms = entry['firms']
    abstract_title = entry['abstract_title']
    full_abstract = entry['full_abstract']
    original_row = entry['original_row']

    print(f"[{index}] Processing row with drugs: {individual_drugs}")

    # Process each drug individually - separate groupings for each field
    drug_classes_grouped = {}
    content_urls_grouped = {}
    steps_taken_grouped = {}
    all_drug_classes = []  # For flattened output
    success_flags = []
    total_llm_calls = 0
    total_search_results = 0

    for drug in individual_drugs:
        print(f"  - Processing drug: {drug}")

        try:
            # Invoke the agent with firm list and abstract info
            result = agent.invoke(
                drug=drug,
                firm=firms,
                abstract_title=abstract_title,
                full_abstract=full_abstract,
                abstract_id=entry['abstract_id']
            )

            # Extract drug class data
            extracted_data = extract_drug_class_from_response(result)

            # Count search results
            drug_class_search_count = len(result.get('drug_class_search_results', []))
            firm_search_count = len(result.get('firm_search_results', []))
            total_search_results += drug_class_search_count + firm_search_count
            total_llm_calls += result.get('llm_calls', 0)

            # Store grouped results separately
            drug_classes_grouped[drug] = extracted_data.get("drug_classes", ["NA"])
            content_urls_grouped[drug] = extracted_data.get("content_urls", ["NA"])
            steps_taken_grouped[drug] = extracted_data.get("steps_taken", [])
            success_flags.append(extracted_data.get("success", False))

            # Collect drug classes for flattened output (exclude "NA")
            drug_classes = extracted_data.get("drug_classes", [])
            for dc in drug_classes:
                if dc and dc != "NA" and dc not in all_drug_classes:
                    all_drug_classes.append(dc)

        except Exception as e:
            print(f"  Error processing drug {drug}: {e}")
            drug_classes_grouped[drug] = ["NA"]
            content_urls_grouped[drug] = ["NA"]
            steps_taken_grouped[drug] = []
            success_flags.append(False)

    # If no valid drug classes found, use ["NA"]
    if not all_drug_classes:
        all_drug_classes = ["NA"]

    # Determine overall success
    overall_success = any(success_flags)

    # Build output row (preserve original columns + add new ones)
    output_row = original_row.copy()
    output_row.update({
        "drug_classes_grouped": json.dumps(drug_classes_grouped, indent=2),  # Only drug classes grouped by drug (pretty-printed)
        "content_urls_grouped": json.dumps(content_urls_grouped, indent=2),  # Content URLs grouped by drug (pretty-printed)
        "steps_taken_grouped": json.dumps(steps_taken_grouped, indent=2),  # Steps taken grouped by drug (pretty-printed)
        "drug_classes": json.dumps(all_drug_classes),  # Flattened
        "success": overall_success,
        "llm_calls": total_llm_calls,
        "search_results_count": total_search_results,
    })

    return output_row


def process_rows_batch(entries: List[Dict], agent: DrugClassAgent,
                       output_file: str = None, max_workers: int = 3) -> pd.DataFrame:
    """Process a batch of rows and return results DataFrame.

    Args:
        entries: List of row dictionaries
        agent: Initialized DrugClassAgent
        output_file: Optional output file path to save intermediate results
        max_workers: Number of parallel workers (default: 3)

    Returns:
        DataFrame with processing results
    """
    print(f"Processing {len(entries)} rows (using {max_workers} parallel threads)")

    results = []

    # Process rows in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_row, entry, agent, i): i
            for i, entry in enumerate(entries, 1)
        }

        # Collect results as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_row = future.result()
                results.append((index, result_row))  # Store with index to maintain order
                completed_count += 1

                # Save intermediate results every 10 rows
                if output_file and completed_count % 10 == 0:
                    # Sort results by index to maintain original order for intermediate saves
                    sorted_results = [result for _, result in sorted(results)]
                    temp_df = pd.DataFrame(sorted_results)
                    temp_df.to_csv(output_file, index=False)
                    print(f"Saved intermediate results ({completed_count}/{len(entries)}) to {output_file}")

            except Exception as e:
                print(f"Error in thread execution: {e}")

    # Sort results by original index to maintain order
    results.sort(key=lambda x: x[0])
    results = [result for _, result in results]

    # Create final DataFrame
    results_df = pd.DataFrame(results)

    # Save final results
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Final results saved to {output_file}")

    return results_df


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(description='Batch Process Drug Class Extraction')
    parser.add_argument('--input_file', default='data/drug_class_input_500.csv',
                        help='Input CSV file with drugs')
    parser.add_argument('--output_file', default=None,
                        help='Output CSV file (default: auto-generated)')
    parser.add_argument('--max_entries', type=int, default=None,
                        help='Max CSV rows to process (default: all)')
    parser.add_argument('--randomize', action='store_true',
                        help='Randomize row selection')
    parser.add_argument('--extraction_model', default='gpt-4.1',
                        help='Model to use for drug class extraction (default: gpt-4.1)')
    parser.add_argument('--max_workers', type=int, default=3,
                        help='Maximum parallel workers (default: 3)')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"drug_class_results_{timestamp}.csv"

    print("🧬 Batch Drug Class Extraction Processor")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Extraction model: {args.extraction_model}")
    print(f"Max entries: {args.max_entries or 'all'}")
    print(f"Randomize: {args.randomize}")
    print(f"Max workers: {args.max_workers}")
    print()

    # Load rows from CSV
    print("Loading rows from CSV...")
    entries = load_rows_from_csv(
        args.input_file,
        max_entries=args.max_entries,
        randomize=args.randomize
    )

    if not entries:
        print("No rows loaded. Exiting.")
        return

    print(f"Total rows to process: {len(entries)}")
    print()

    # Initialize agent
    print("Initializing Drug Class Agent...")
    agent = DrugClassAgent(
        agent_name="BatchDrugClassProcessor",
        extraction_model=args.extraction_model,
    )
    print("✓ Agent initialized successfully!")
    print()

    # Process rows
    results_df = process_rows_batch(
        entries, agent, args.output_file, max_workers=args.max_workers
    )

    # Summary
    total_processed = len(results_df)
    successful = results_df['success'].sum() if 'success' in results_df.columns else 0
    success_rate = (successful / total_processed * 100) if total_processed > 0 else 0

    print()
    print("📊 Processing Summary:")
    print(f"Total rows processed: {total_processed}")
    print(f"Successful extractions: {int(successful)}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
