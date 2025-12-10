#!/usr/bin/env python3
"""
Batch Processor for Drug Class ReAct Extraction Agent

This script processes multiple drugs using the drug class ReAct extraction agent
and saves results to CSV format for analysis.

Features:
- Reads abstract_id, abstract_title, drug_name, firm, full_abstract from input CSV
- Loads pre-fetched search results from cache JSON file
- Handles multiple drugs per row (comma/semicolon separated)
- Groups results by drug with flattened drug_classes column
- Preserves all original input columns in output
- Uses ReAct pattern with tool calling for rule retrieval
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drug_class_react_agent import DrugClassReActAgent


def load_cache(cache_file: str) -> Dict:
    """Load cached search results.

    Args:
        cache_file: Path to cache JSON file

    Returns:
        Dictionary with cached data
    """
    if not os.path.exists(cache_file):
        print(f"Warning: Cache file not found at {cache_file}")
        return {"drugs": {}}

    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            drug_count = len(data.get('drugs', {}))
            print(f"✓ Loaded cache with {drug_count} unique drugs")
            return data
    except Exception as e:
        print(f"Error loading cache: {e}")
        return {"drugs": {}}


def get_firms_key(firms: List[str]) -> str:
    """Create a consistent key for a list of firms.

    Args:
        firms: List of firm names

    Returns:
        JSON-serialized key
    """
    sorted_firms = sorted([f.strip() for f in firms if f.strip()])
    return json.dumps(sorted_firms)


def get_search_results_from_cache(
    cache_data: Dict,
    drug_name: str,
    firms: List[str]
) -> Tuple[List[Dict], List[Dict]]:
    """Look up search results from cache for a drug+firm combination.

    Args:
        cache_data: Cache dictionary
        drug_name: Drug name to look up
        firms: List of firm names

    Returns:
        Tuple of (drug_class_results, firm_results)
    """
    drugs_cache = cache_data.get("drugs", {})

    # Look up drug in cache
    drug_data = drugs_cache.get(drug_name, {})

    if not drug_data:
        return [], []

    # Get drug class search results (shared for all firms)
    drug_class_results = drug_data.get("drug_class_search", {}).get("results", [])

    # Get firm-specific results
    firms_key = get_firms_key(firms)
    firm_results = drug_data.get("firm_searches", {}).get(firms_key, {}).get("results", [])

    return drug_class_results, firm_results


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
    firms: List[str],
    abstract_title: str,
    full_abstract: str,
    abstract_id: str,
    cache_data: Dict,
    agent: DrugClassReActAgent,
) -> Dict[str, Any]:
    """Process a single drug and return extraction results.

    Args:
        drug: Drug name
        firms: List of firm names
        abstract_title: Abstract title
        full_abstract: Full abstract text
        abstract_id: Abstract ID
        cache_data: Cache dictionary with search results
        agent: Initialized DrugClassReActAgent

    Returns:
        Dictionary with extraction results for this drug
    """
    # Get search results from cache
    drug_class_results, firm_results = get_search_results_from_cache(
        cache_data, drug, firms
    )

    # Invoke the agent
    result = agent.invoke(
        drug=drug,
        abstract_title=abstract_title,
        full_abstract=full_abstract,
        drug_class_results=drug_class_results,
        firm_results=firm_results,
        abstract_id=abstract_id,
    )

    # Parse the response
    parsed = agent.parse_response(result)
    parsed['llm_calls'] = result.get('llm_calls', 0)

    return parsed


def process_single_row(
    entry: Dict,
    cache_data: Dict,
    agent: DrugClassReActAgent,
    index: int,
) -> Dict:
    """Process a single row (may contain multiple drugs) and return grouped results.

    Args:
        entry: Row dictionary with individual_drugs list and original row data
        cache_data: Cache dictionary with search results
        agent: Initialized DrugClassReActAgent
        index: Index of the row for logging

    Returns:
        Dictionary with grouped processing result
    """
    individual_drugs = entry['individual_drugs']
    firms = entry['firms']
    abstract_title = entry['abstract_title']
    full_abstract = entry['full_abstract']
    abstract_id = entry['abstract_id']
    original_row = entry['original_row']

    print(f"[{index}] Processing row with drugs: {individual_drugs}")

    # Process each drug individually - separate groupings for each field
    drug_classes_grouped = {}
    selected_sources_grouped = {}
    confidence_scores_grouped = {}
    reasoning_grouped = {}
    rules_retrieved_grouped = {}
    components_identified_grouped = {}
    quality_metrics_grouped = {}
    all_drug_classes = []  # For flattened output
    success_flags = []
    total_llm_calls = 0

    for drug in individual_drugs:
        print(f"  - Processing drug: {drug}")

        try:
            result = process_single_drug(
                drug=drug,
                firms=firms,
                abstract_title=abstract_title,
                full_abstract=full_abstract,
                abstract_id=abstract_id,
                cache_data=cache_data,
                agent=agent,
            )

            # Store grouped results
            drug_classes_grouped[drug] = result.get("drug_classes", ["NA"])
            selected_sources_grouped[drug] = result.get("selected_sources", [])
            confidence_scores_grouped[drug] = result.get("confidence_score")
            reasoning_grouped[drug] = result.get("reasoning", "")
            rules_retrieved_grouped[drug] = result.get("rules_retrieved", [])
            components_identified_grouped[drug] = result.get("components_identified", [])
            quality_metrics_grouped[drug] = {
                "completeness": result.get("quality_metrics_completeness"),
                "rule_adherence": result.get("quality_metrics_rule_adherence"),
                "clinical_accuracy": result.get("quality_metrics_clinical_accuracy"),
                "formatting_compliance": result.get("quality_metrics_formatting_compliance"),
            }
            success_flags.append(result.get("success", False))
            total_llm_calls += result.get("llm_calls", 0)

            # Collect drug classes for flattened output (exclude "NA")
            drug_classes = result.get("drug_classes", [])
            for dc in drug_classes:
                if dc and dc != "NA" and dc not in all_drug_classes:
                    all_drug_classes.append(dc)

        except Exception as e:
            print(f"  Error processing drug {drug}: {e}")
            drug_classes_grouped[drug] = ["NA"]
            selected_sources_grouped[drug] = []
            confidence_scores_grouped[drug] = None
            reasoning_grouped[drug] = f"Error: {str(e)}"
            rules_retrieved_grouped[drug] = []
            components_identified_grouped[drug] = []
            quality_metrics_grouped[drug] = {}
            success_flags.append(False)

    # If no valid drug classes found, use ["NA"]
    if not all_drug_classes:
        all_drug_classes = ["NA"]

    # Determine overall success
    overall_success = any(success_flags)

    # Build output row (preserve original columns + add new ones)
    output_row = original_row.copy()
    output_row.update({
        "drug_classes_grouped": json.dumps(drug_classes_grouped, indent=2),
        "selected_sources_grouped": json.dumps(selected_sources_grouped, indent=2),
        "confidence_scores_grouped": json.dumps(confidence_scores_grouped, indent=2),
        "reasoning_grouped": json.dumps(reasoning_grouped, indent=2),
        "rules_retrieved_grouped": json.dumps(rules_retrieved_grouped, indent=2),
        "components_identified_grouped": json.dumps(components_identified_grouped, indent=2),
        "quality_metrics_grouped": json.dumps(quality_metrics_grouped, indent=2),
        "drug_classes": json.dumps(all_drug_classes),  # Flattened
        "success": overall_success,
        "llm_calls": total_llm_calls,
    })

    return output_row


def process_rows_batch(
    entries: List[Dict],
    cache_data: Dict,
    agent: DrugClassReActAgent,
    output_file: str = None,
    max_workers: int = 3,
) -> pd.DataFrame:
    """Process a batch of rows and return results DataFrame.

    Args:
        entries: List of row dictionaries
        cache_data: Cache dictionary with search results
        agent: Initialized DrugClassReActAgent
        output_file: Optional output file path to save intermediate results
        max_workers: Number of parallel workers (default: 3)

    Returns:
        DataFrame with processing results
    """
    print(f"Processing {len(entries)} rows (using {max_workers} parallel threads)")

    results = []
    write_lock = threading.Lock()

    # Define output fieldnames
    output_fieldnames = [
        'abstract_id', 'abstract_title', 'drug_name',
        'Drug Class - Ground truth (Manually extracted)', 'firm', 'full_abstract',
        'drug_classes_grouped', 'selected_sources_grouped', 'confidence_scores_grouped',
        'reasoning_grouped', 'rules_retrieved_grouped', 'components_identified_grouped',
        'quality_metrics_grouped', 'drug_classes', 'success', 'llm_calls'
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
                result = process_single_row(entry, cache_data, agent, i)
                results.append((i, result))
                write_result(result, i)
        else:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        process_single_row,
                        entry,
                        cache_data,
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
    parser = argparse.ArgumentParser(description='Batch Process Drug Class Extraction using ReAct Agent')
    parser.add_argument('--input_file', default='data/drug_class_asco_100.csv',
                        help='Input CSV file with drugs (default: data/drug_class_asco_100.csv)')
    parser.add_argument('--cache_file', default='data/drug_search_cache.json',
                        help='Input JSON cache file with search results (default: data/drug_search_cache.json)')
    parser.add_argument('--output_file', default=None,
                        help='Output CSV file (default: auto-generated)')
    parser.add_argument('--model', default='gemini/gemini-2.5-pro',
                        help='LLM model to use (default: gemini/gemini-2.5-pro)')
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
        args.output_file = f"data/drug_class_react_extraction_{model_safe}_{timestamp}.csv"

    print("🧬 Drug Class ReAct Batch Processor")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Cache file: {args.cache_file}")
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

    # Load cache
    print("Loading cache...")
    cache_data = load_cache(args.cache_file)

    # Initialize agent
    print(f"\nInitializing Drug Class ReAct Agent ({args.model})...")
    agent = DrugClassReActAgent(
        agent_name="BatchDrugClassReActProcessor",
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
        cache_data=cache_data,
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

