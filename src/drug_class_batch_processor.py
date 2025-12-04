#!/usr/bin/env python3
"""
Batch Processor for Drug Class Extraction Agent

This script processes multiple drugs using the drug class extraction agent
and saves results to CSV format for analysis.
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


def load_drugs_from_csv(csv_path: str, max_drugs: int = None, randomize: bool = False) -> List[Dict]:
    """Load drugs from CSV file.

    Args:
        csv_path: Path to the CSV file
        max_drugs: Maximum number of drugs to load
        randomize: Whether to randomize the selection

    Returns:
        List of dictionaries with drug data
    """
    drugs = []

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return drugs

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

            if not drug_name_col:
                print(f"Warning: Could not find drug_name column in {csv_path}")
                print(f"Available columns: {list(header_map.keys())}")
                return drugs

            for row in reader:
                drug_data = {
                    'drug_name': row.get(drug_name_col, '').strip(),
                    'firm': row.get(firm_col, '').strip() if firm_col else '',
                }

                # Only add if we have a drug name
                if drug_data['drug_name']:
                    drugs.append(drug_data)

        if randomize and max_drugs and len(drugs) > max_drugs:
            import random
            drugs = random.sample(drugs, max_drugs)
        elif max_drugs and len(drugs) > max_drugs:
            drugs = drugs[:max_drugs]

        return drugs

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
                'drug_classes': [],
                'content_urls': [],
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
            'drug_classes': [],
            'content_urls': [],
            'steps_taken': [],
            'success': False,
            'error': str(e),
        }


def process_single_drug(drug_data: Dict, agent: DrugClassAgent, index: int) -> Dict:
    """Process a single drug and return the result.

    Args:
        drug_data: Drug dictionary with drug_name and firm
        agent: Initialized DrugClassAgent
        index: Index of the drug for logging

    Returns:
        Dictionary with processing result
    """
    drug_name = drug_data['drug_name']
    firm = drug_data['firm']

    print(f"Processing drug {index}: {drug_name}")

    try:
        # Invoke the agent
        result = agent.invoke(drug=drug_name, firm=firm)

        # Extract drug class data
        extracted_data = extract_drug_class_from_response(result)

        # Count search results
        drug_class_search_count = len(result.get('drug_class_search_results', []))
        firm_search_count = len(result.get('firm_search_results', []))
        total_search_results = drug_class_search_count + firm_search_count

        # Build result row
        result_row = {
            'drug_name': drug_name,
            'firm': firm,
            'drug_classes': json.dumps(extracted_data['drug_classes']),
            'content_urls': json.dumps(extracted_data['content_urls']),
            'steps_taken': json.dumps(extracted_data['steps_taken']),
            'success': extracted_data['success'],
            'llm_calls': result.get('llm_calls', 0),
            'search_results_count': total_search_results,
        }

        return result_row

    except Exception as e:
        print(f"Error processing drug {drug_name}: {e}")
        # Add error result
        result_row = {
            'drug_name': drug_name,
            'firm': firm,
            'drug_classes': json.dumps([]),
            'content_urls': json.dumps([]),
            'steps_taken': json.dumps([]),
            'success': False,
            'llm_calls': 0,
            'search_results_count': 0,
        }
        return result_row


def process_drugs_batch(drugs: List[Dict], agent: DrugClassAgent,
                        output_file: str = None, max_workers: int = 3) -> pd.DataFrame:
    """Process a batch of drugs and return results DataFrame.

    Args:
        drugs: List of drug dictionaries
        agent: Initialized DrugClassAgent
        output_file: Optional output file path to save intermediate results
        max_workers: Number of parallel workers (default: 3)

    Returns:
        DataFrame with processing results
    """
    print(f"Processing {len(drugs)} drugs (using {max_workers} parallel threads)")

    results = []

    # Process drugs in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_drug, drug, agent, i): i
            for i, drug in enumerate(drugs, 1)
        }

        # Collect results as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_row = future.result()
                results.append((index, result_row))  # Store with index to maintain order
                completed_count += 1

                # Save intermediate results every 10 drugs
                if output_file and completed_count % 10 == 0:
                    # Sort results by index to maintain original order for intermediate saves
                    sorted_results = [result for _, result in sorted(results)]
                    temp_df = pd.DataFrame(sorted_results)
                    temp_df.to_csv(output_file, index=False)
                    print(f"Saved intermediate results ({completed_count}/{len(drugs)}) to {output_file}")

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
    parser.add_argument('--input_file', default='data/drug_class_input.csv',
                        help='Input CSV file with drugs')
    parser.add_argument('--output_file', default=None,
                        help='Output CSV file (default: auto-generated)')
    parser.add_argument('--num_drugs', type=int, default=None,
                        help='Number of drugs to process (default: all)')
    parser.add_argument('--randomize', action='store_true',
                        help='Randomize drug selection')
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
    print(f"Number of drugs: {args.num_drugs or 'all'}")
    print(f"Randomize: {args.randomize}")
    print(f"Max workers: {args.max_workers}")
    print()

    # Load drugs
    print("Loading drugs...")
    drugs = load_drugs_from_csv(
        args.input_file,
        max_drugs=args.num_drugs,
        randomize=args.randomize
    )

    if not drugs:
        print("No drugs loaded. Exiting.")
        return

    print(f"Loaded {len(drugs)} drugs")
    print()

    # Initialize agent
    print("Initializing Drug Class Agent...")
    agent = DrugClassAgent(
        agent_name="BatchDrugClassProcessor",
        extraction_model=args.extraction_model,
    )
    print("✓ Agent initialized successfully!")
    print()

    # Process drugs
    results_df = process_drugs_batch(
        drugs, agent, args.output_file, max_workers=args.max_workers
    )

    # Summary
    total_processed = len(results_df)
    successful = results_df['success'].sum()
    success_rate = (successful / total_processed * 100) if total_processed > 0 else 0

    print()
    print("📊 Processing Summary:")
    print(f"Total drugs processed: {total_processed}")
    print(f"Successful extractions: {int(successful)}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

