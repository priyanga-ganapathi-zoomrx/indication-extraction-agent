#!/usr/bin/env python3
"""
Drug Class Grounded Search Processor

This script processes drugs using the DrugClassGroundedSearchAgent to identify
drug classes with source attribution using OpenAI's web_search_preview tool.

Input: CSV file with drugs (uses flattened_components column by default)
Output: CSV file with drug classes, sources, evidence, and web search annotations

Features:
- Reads drugs from CSV (supports flattened_components or drug_name columns)
- Filters to only process drugs with NA or empty drug_classes_grouped
- Uses OpenAI grounded search to find drug classes from authoritative sources
- Captures web search annotations with source citations
- Supports parallel processing with ThreadPoolExecutor
- Saves intermediate results incrementally
- Tracks execution time
"""

import argparse
import concurrent.futures
import csv
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.drug_class import DrugClassGroundedSearchAgent


def should_process_drug(drug_classes_grouped_value: str, drug_name: str) -> bool:
    """Check if a drug should be processed based on its existing drug_classes_grouped value.

    A drug should be processed if its drug_classes_grouped is:
    - Empty string or None
    - Contains only "NA" or ["NA"]
    - An empty array []
    - JSON object where all values are ["NA"] or empty

    Args:
        drug_classes_grouped_value: The drug_classes_grouped column value
        drug_name: The specific drug name to check within the grouped value

    Returns:
        True if the drug should be processed (has NA or empty drug class)
    """
    if not drug_classes_grouped_value or drug_classes_grouped_value.strip() == '':
        return True

    try:
        parsed = json.loads(drug_classes_grouped_value)
        
        # If it's a dict (grouped by drug name)
        if isinstance(parsed, dict):
            # Check if specific drug exists and has NA or empty
            if drug_name in parsed:
                drug_classes = parsed[drug_name]
                if not drug_classes:
                    return True
                if isinstance(drug_classes, list):
                    # Check if all values are "NA" or empty
                    return all(dc == "NA" or dc == "" or dc is None for dc in drug_classes)
                return drug_classes == "NA"
            # Drug not in dict, should process
            return True
        
        # If it's a list directly
        if isinstance(parsed, list):
            if not parsed:
                return True
            return all(dc == "NA" or dc == "" or dc is None for dc in parsed)
        
        # If it's a string
        if isinstance(parsed, str):
            return parsed == "NA" or parsed == ""
            
    except json.JSONDecodeError:
        # If not valid JSON, check as string
        stripped = drug_classes_grouped_value.strip()
        return stripped == "NA" or stripped == "" or stripped == "[]" or stripped == '["NA"]'

    return False


def load_rows_from_csv(
    csv_path: str,
    max_entries: int = None,
    skip_rows: int = 0,
    use_drug_name: bool = False,
    filter_na_only: bool = True,
) -> List[Dict]:
    """Load rows from CSV file.

    By default, uses flattened_components column (JSON array) for drug names.
    If use_drug_name=True or flattened_components not found, falls back to drug_name column.

    Args:
        csv_path: Path to the CSV file
        max_entries: Maximum number of CSV rows to return
        skip_rows: Number of rows to skip from the beginning
        use_drug_name: If True, use drug_name column instead of flattened_components
        filter_na_only: If True, only include drugs with NA or empty drug_classes_grouped

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
            flattened_components_col = header_map.get('flattened_components')
            drug_name_col = header_map.get('drug_name') or header_map.get('drug name') or header_map.get('drug')
            abstract_id_col = header_map.get('abstract_id') or header_map.get('id')
            abstract_title_col = header_map.get('abstract_title') or header_map.get('title')
            full_abstract_col = header_map.get('full_abstract') or header_map.get('abstract')
            firm_col = header_map.get('firm') or header_map.get('company') or header_map.get('sponsor')
            drug_classes_grouped_col = header_map.get('drug_classes_grouped')

            # Determine which column to use for drug names
            if not use_drug_name and flattened_components_col:
                source_col = flattened_components_col
                is_json_column = True
                print("  Using 'flattened_components' column for drug names (JSON array)")
            elif not use_drug_name and not flattened_components_col:
                print("  Warning: 'flattened_components' column not found, falling back to 'drug_name'")
                source_col = drug_name_col
                is_json_column = False
            else:
                source_col = drug_name_col
                is_json_column = False
                print("  Using 'drug_name' column for drug names")

            if not source_col:
                print(f"Warning: Could not find drug column in {csv_path}")
                print(f"Available columns: {list(header_map.keys())}")
                return entries

            if filter_na_only and drug_classes_grouped_col:
                print("  Filtering: Only processing drugs with NA or empty drug_classes_grouped")
            elif filter_na_only and not drug_classes_grouped_col:
                print("  Warning: 'drug_classes_grouped' column not found, processing all drugs")
                filter_na_only = False

            rows_list = list(reader)
            
            # Skip rows if specified
            if skip_rows > 0:
                rows_list = rows_list[skip_rows:]

            # Limit rows if specified
            if max_entries is not None:
                rows_list = rows_list[:max_entries]

            total_drugs_before_filter = 0
            total_drugs_after_filter = 0

            for row_id, row in enumerate(rows_list, start=skip_rows + 1):
                # Parse drugs from the source column
                drug_value = row.get(source_col, '').strip()
                individual_drugs = []

                if drug_value:
                    if is_json_column:
                        # Parse JSON array from flattened_components column
                        try:
                            parsed = json.loads(drug_value)
                            if isinstance(parsed, list):
                                individual_drugs = [d.strip() for d in parsed if d and str(d).strip()]
                            else:
                                individual_drugs = [str(parsed).strip()] if parsed else []
                        except json.JSONDecodeError:
                            # Fallback: treat as comma-separated string
                            individual_drugs = [d.strip() for d in drug_value.replace(';', ',').split(',') if d.strip()]
                    else:
                        # Split by comma or semicolon and strip whitespace
                        individual_drugs = [d.strip() for d in drug_value.replace(';', ',').split(',') if d.strip()]

                if not individual_drugs:
                    continue

                total_drugs_before_filter += len(individual_drugs)

                # Get drug_classes_grouped for filtering
                drug_classes_grouped = row.get(drug_classes_grouped_col, '') if drug_classes_grouped_col else ''

                # Filter drugs based on their existing drug class
                if filter_na_only:
                    drugs_to_process = [
                        drug for drug in individual_drugs 
                        if should_process_drug(drug_classes_grouped, drug)
                    ]
                else:
                    drugs_to_process = individual_drugs

                if not drugs_to_process:
                    continue

                total_drugs_after_filter += len(drugs_to_process)

                # Get abstract fields
                abstract_id = row.get(abstract_id_col, '').strip() if abstract_id_col else ''
                abstract_title = row.get(abstract_title_col, '').strip() if abstract_title_col else ''
                full_abstract = row.get(full_abstract_col, '').strip() if full_abstract_col else ''
                firm = row.get(firm_col, '').strip() if firm_col else ''

                # Get original drug_name for output
                raw_drug_name = row.get(drug_name_col, '').strip() if drug_name_col else ''

                # Store original row data for output (all columns from input)
                original_row = {col: row.get(col, '') for col in original_fieldnames}
                # Ensure key columns are present with correct names
                original_row.update({
                    'abstract_id': abstract_id,
                    'abstract_title': abstract_title,
                    'drug_name': raw_drug_name,
                    'firm': firm,
                    'full_abstract': full_abstract,
                })

                entries.append({
                    'row_id': row_id,
                    'original_row': original_row,
                    'original_fieldnames': original_fieldnames,
                    'individual_drugs': drugs_to_process,
                    'abstract_id': abstract_id,
                    'abstract_title': abstract_title,
                    'full_abstract': full_abstract,
                })

        print(f"✓ Loaded {len(entries)} CSV rows")
        if filter_na_only:
            print(f"  Total drugs before filter: {total_drugs_before_filter}")
            print(f"  Total drugs after filter (NA/empty only): {total_drugs_after_filter}")
        return entries

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_single_drug(
    drug: str,
    abstract_title: str,
    abstract_id: str,
    agent: DrugClassGroundedSearchAgent,
) -> Dict[str, Any]:
    """Process a single drug and return extraction results.

    Args:
        drug: Drug name
        abstract_title: Abstract title for context
        abstract_id: Abstract ID
        agent: Initialized DrugClassGroundedSearchAgent

    Returns:
        Dictionary with extraction results for this drug
    """
    # Invoke the agent
    result = agent.invoke(
        drug_name=drug,
        abstract_title=abstract_title,
        abstract_id=abstract_id,
    )

    return result


def process_single_row(
    entry: Dict,
    agent: DrugClassGroundedSearchAgent,
    index: int,
) -> Dict:
    """Process a single row (may contain multiple drugs) and return grouped results.

    Args:
        entry: Row dictionary with individual_drugs list and original row data
        agent: Initialized DrugClassGroundedSearchAgent
        index: Index of the row for logging

    Returns:
        Dictionary with grouped processing result
    """
    individual_drugs = entry['individual_drugs']
    abstract_title = entry['abstract_title']
    abstract_id = entry['abstract_id']
    original_row = entry['original_row']

    print(f"[{index}] Processing row with drugs: {individual_drugs}")

    # Process each drug individually
    drug_classes_grouped = {}
    drug_class_details_grouped = {}
    reasoning_grouped = {}
    no_class_found_grouped = {}
    all_drug_classes = []
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

            # Store grouped results
            drug_classes_grouped[drug] = result.get("drug_classes", ["NA"])
            drug_class_details_grouped[drug] = result.get("drug_class_details", [])
            reasoning_grouped[drug] = result.get("reasoning", "")
            no_class_found_grouped[drug] = result.get("no_class_found", True)
            success_flags.append(result.get("success", False))
            total_llm_calls += result.get("llm_calls", 0)

            # Collect drug classes for flattened output (exclude "NA")
            drug_classes = result.get("drug_classes", [])
            for dc in drug_classes:
                if dc and dc != "NA" and dc not in all_drug_classes:
                    all_drug_classes.append(dc)

            # Log result
            if result.get("no_class_found", True):
                print(f"      No class found")
            else:
                print(f"      Found classes: {drug_classes}")

        except Exception as e:
            print(f"  Error processing drug {drug}: {e}")
            import traceback
            traceback.print_exc()
            drug_classes_grouped[drug] = ["NA"]
            drug_class_details_grouped[drug] = []
            reasoning_grouped[drug] = f"Error: {str(e)}"
            no_class_found_grouped[drug] = True
            success_flags.append(False)

    # If no valid drug classes found, use ["NA"]
    if not all_drug_classes:
        all_drug_classes = ["NA"]

    # Determine overall success
    overall_success = any(success_flags)

    # Build output row (preserve original columns + add new ones)
    output_row = original_row.copy()

    # Format reasoning with actual newlines for better CSV readability
    reasoning_formatted = {}
    for drug, reasoning in reasoning_grouped.items():
        if isinstance(reasoning, str):
            formatted = reasoning.replace('\\n', '\n')
            formatted = re.sub(r'(?<!\n)\s+(\d+\.)\s+', r'\n\1 ', formatted)
            reasoning_formatted[drug] = formatted
        else:
            reasoning_formatted[drug] = reasoning

    output_row.update({
        "grounded_drug_classes_grouped": json.dumps(drug_classes_grouped, indent=2),
        "grounded_drug_class_details_grouped": json.dumps(drug_class_details_grouped, indent=2),
        "grounded_reasoning_grouped": json.dumps(reasoning_formatted, indent=2).replace('\\n', '\n'),
        "grounded_no_class_found_grouped": json.dumps(no_class_found_grouped, indent=2),
        "grounded_drug_classes": json.dumps(all_drug_classes),
        "grounded_success": overall_success,
        "grounded_llm_calls": total_llm_calls,
    })

    return output_row


def process_rows_batch(
    entries: List[Dict],
    agent: DrugClassGroundedSearchAgent,
    output_file: str = None,
    max_workers: int = 3,
) -> pd.DataFrame:
    """Process a batch of rows and return results DataFrame.

    Args:
        entries: List of row dictionaries
        agent: Initialized DrugClassGroundedSearchAgent
        output_file: Optional output file path to save intermediate results
        max_workers: Number of parallel workers (default: 3)

    Returns:
        DataFrame with processing results
    """
    print(f"Processing {len(entries)} rows (using {max_workers} parallel threads)")

    results = []
    write_lock = threading.Lock()

    # Get original fieldnames from first entry and add new processing columns
    original_fieldnames = entries[0].get('original_fieldnames', []) if entries else []

    # New columns added by processing
    new_columns = [
        'grounded_drug_classes_grouped',
        'grounded_drug_class_details_grouped',
        'grounded_reasoning_grouped',
        'grounded_no_class_found_grouped',
        'grounded_drug_classes',
        'grounded_success',
        'grounded_llm_calls',
    ]

    # Combine: all original columns + new columns (avoiding duplicates)
    output_fieldnames = list(original_fieldnames) + [col for col in new_columns if col not in original_fieldnames]

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
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Drug Class Grounded Search Processor')
    parser.add_argument('--input_file', default='data/drug_class_validation_input_na.csv',
                        help='Input CSV file with drugs')
    parser.add_argument('--output_file', default=None,
                        help='Output CSV file (default: auto-generated)')
    parser.add_argument('--llm_model', default="gemini/gemini-3-flash-preview",
                        help='LLM model to use (should support web_search_preview)')
    parser.add_argument('--temperature', type=float, default=0,
                        help='LLM temperature (default: 0)')
    parser.add_argument('--max_tokens', type=int, default=8192,
                        help='LLM max tokens (default: 8192)')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Maximum number of rows to process (default: all)')
    parser.add_argument('--skip_rows', type=int, default=0,
                        help='Number of rows to skip from the beginning')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--use_drug_name', action='store_true',
                        help='Use drug_name column instead of flattened_components')
    parser.add_argument('--enable_caching', action='store_true',
                        help='Enable prompt caching (for Anthropic models)')
    parser.add_argument('--disable_web_search', action='store_true',
                        help='Disable web_search_preview tool')
    parser.add_argument('--process_all', action='store_true',
                        help='Process all drugs, not just those with NA/empty drug_classes_grouped')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        model_safe = args.llm_model.replace("/", "-")
        args.output_file = f"data/{base_name}_grounded_{model_safe}_{timestamp}.csv"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print("🔍 Drug Class Grounded Search Processor")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"LLM model: {args.llm_model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Max rows: {args.max_rows or 'all'}")
    print(f"Skip rows: {args.skip_rows}")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Use flattened_components: {not args.use_drug_name}")
    print(f"Enable caching: {args.enable_caching}")
    print(f"Web search enabled: {not args.disable_web_search}")
    print(f"Filter NA only: {not args.process_all}")
    print()

    # Load input CSV
    print("Loading input CSV...")
    entries = load_rows_from_csv(
        args.input_file,
        max_entries=args.max_rows,
        skip_rows=args.skip_rows,
        use_drug_name=args.use_drug_name,
        filter_na_only=not args.process_all,
    )

    if not entries:
        print("No entries found in input CSV (after filtering).")
        return

    # Count total drugs to process
    total_drugs = sum(len(entry['individual_drugs']) for entry in entries)
    print(f"Total drugs to process: {total_drugs}")
    print()

    # Initialize agent
    print(f"Initializing Drug Class Grounded Search Agent ({args.llm_model})...")
    agent = DrugClassGroundedSearchAgent(
        agent_name="DrugClassGroundedSearchProcessor",
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_caching=args.enable_caching,
        enable_web_search=not args.disable_web_search,
    )
    print("✓ Agent initialized")
    print()

    # Process entries
    print(f"Processing {len(entries)} rows...")
    print("-" * 80)

    results_df = process_rows_batch(
        entries=entries,
        agent=agent,
        output_file=args.output_file,
        max_workers=args.num_workers,
    )

    # Summary
    print()
    print("=" * 80)
    total_processed = len(results_df)

    if 'grounded_success' in results_df.columns:
        successful = results_df['grounded_success'].sum()
        success_rate = (successful / total_processed * 100) if total_processed > 0 else 0

        print("📊 Summary:")
        print(f"  Total rows processed: {total_processed}")
        print(f"  Successful: {int(successful)} ({success_rate:.1f}%)")

    print(f"  Results saved to: {args.output_file}")

    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print()
    print(f"⏱️  Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")


if __name__ == "__main__":
    main()
