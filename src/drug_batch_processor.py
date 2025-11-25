#!/usr/bin/env python3
"""
Batch Processor for Drug Extraction Agent

This script processes multiple abstracts using the drug extraction agent
and saves results to CSV format for analysis.
"""

import csv
import json
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import argparse
import concurrent.futures
import sys
import os

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drug_agent import DrugExtractionAgent


def load_abstracts_from_csv(csv_path: str, max_abstracts: int = None, randomize: bool = False) -> List[Dict]:
    """Load abstracts from CSV file.

    Args:
        csv_path: Path to the CSV file
        max_abstracts: Maximum number of abstracts to load
        randomize: Whether to randomize the selection

    Returns:
        List of dictionaries with abstract data
    """
    abstracts = []

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return abstracts

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as file:
            # Read all rows to a list of dictionaries
            reader = csv.DictReader(file)
            
            # Normalize headers mapping
            # Map normalized (lowercase, stripped) headers to actual headers
            if reader.fieldnames:
                header_map = {h.lower().strip(): h for h in reader.fieldnames}
            else:
                header_map = {}
            
            # Find the correct header names for our required fields
            # We look for common variations
            id_col = header_map.get('abstract_id') or header_map.get('abstract id') or header_map.get('id')
            title_col = header_map.get('abstract_title') or header_map.get('abstract title') or header_map.get('title')
            session_col = header_map.get('session_title') or header_map.get('session title')
            ground_truth_col = header_map.get('ground_truth') or header_map.get('ground truth')

            if not id_col or not title_col:
                print(f"Warning: Could not find required columns (ID or Title) in {csv_path}")
                print(f"Available columns: {list(header_map.keys())}")
                # Continue anyway, might fail later but let's try
            
            for row in reader:
                abstract_data = {
                    'abstract_id': row.get(id_col, '') if id_col else '',
                    'session_title': row.get(session_col, '') if session_col else '',
                    'abstract_title': row.get(title_col, '') if title_col else '',
                    'ground_truth': row.get(ground_truth_col, '') if ground_truth_col else '',
                }
                
                # Only add if we have at least an ID or Title
                if abstract_data['abstract_id'] or abstract_data['abstract_title']:
                    abstracts.append(abstract_data)

        if randomize and max_abstracts and len(abstracts) > max_abstracts:
            import random
            abstracts = random.sample(abstracts, max_abstracts)
        elif max_abstracts and len(abstracts) > max_abstracts:
            abstracts = abstracts[:max_abstracts]

        return abstracts

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def extract_drugs_from_response(result: Dict) -> Dict[str, Any]:
    """Extract drugs from agent response.

    Args:
        result: Agent invocation result

    Returns:
        Dictionary with extracted fields: primary_drugs, secondary_drugs, comparator_drugs, success
    """
    try:
        # Get the final message from the agent
        messages = result.get('messages', [])
        if not messages:
            return {
                'primary_drugs': [],
                'secondary_drugs': [],
                'comparator_drugs': [],
                'success': False,
            }

        final_message = messages[-1]
        content = getattr(final_message, 'content', '')

        if not content or content.startswith("I encountered an error"):
            return {
                'primary_drugs': [],
                'secondary_drugs': [],
                'comparator_drugs': [],
                'success': False,
            }

        # Try to parse JSON response
        import re
        json_str = content

        # 1. Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 2. Try to find the first '{' and last '}'
            # This handles cases where there's text before/after the JSON but no code blocks
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx+1]

        try:
            parsed = json.loads(json_str)
            
            # Helper to safely get list from various key formats
            def get_list(data, keys):
                for key in keys:
                    if key in data:
                        val = data[key]
                        if isinstance(val, list):
                            return val
                return []

            primary_drugs = get_list(parsed, ['Primary Drugs', 'primary_drugs', 'Primary', 'primary'])
            secondary_drugs = get_list(parsed, ['Secondary Drugs', 'secondary_drugs', 'Secondary', 'secondary'])
            comparator_drugs = get_list(parsed, ['Comparator Drugs', 'comparator_drugs', 'Comparator', 'comparator'])
            reasoning = get_list(parsed, ['Reasoning', 'reasoning'])
            
            return {
                'primary_drugs': primary_drugs,
                'secondary_drugs': secondary_drugs,
                'comparator_drugs': comparator_drugs,
                'reasoning': reasoning,
                'success': True,
            }
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Content: {content[:200]}...")
            return {
                'primary_drugs': [],
                'secondary_drugs': [],
                'comparator_drugs': [],
                'reasoning': [],
                'success': False,
            }

    except Exception as e:
        print(f"Error extracting drugs: {e}")
        return {
            'primary_drugs': [],
            'secondary_drugs': [],
            'comparator_drugs': [],
            'reasoning': [],
            'success': False,
        }


def process_single_abstract(abstract: Dict, agent: DrugExtractionAgent,
                           model_name: str, index: int) -> Dict:
    """Process a single abstract and return the result.

    Args:
        abstract: Abstract dictionary
        agent: Initialized DrugExtractionAgent
        model_name: Name of the model being used
        index: Index of the abstract for logging

    Returns:
        Dictionary with processing result
    """
    print(f"Processing abstract {index}: ID {abstract['abstract_id']}")

    try:
        # Invoke the agent
        result = agent.invoke(
            abstract_title=abstract['abstract_title'],
            session_title=abstract['session_title'],
            abstract_id=abstract['abstract_id']
        )

        # Extract drugs
        extracted_data = extract_drugs_from_response(result)

        # Build result row
        result_row = {
            'abstract_id': abstract['abstract_id'],
            'session_title': abstract['session_title'],
            'abstract_title': abstract['abstract_title'],
            'ground_truth': abstract['ground_truth'],
            f'{model_name}_primary_drugs': json.dumps(extracted_data['primary_drugs']),
            f'{model_name}_secondary_drugs': json.dumps(extracted_data['secondary_drugs']),
            f'{model_name}_comparator_drugs': json.dumps(extracted_data['comparator_drugs']),
            f'{model_name}_reasoning': json.dumps(extracted_data['reasoning']),
            f'{model_name}_success': extracted_data['success'],
            f'{model_name}_llm_calls': result.get('llm_calls', 0)
        }

        return result_row

    except Exception as e:
        print(f"Error processing abstract {abstract['abstract_id']}: {e}")
        # Add error result
        result_row = {
            'abstract_id': abstract['abstract_id'],
            'session_title': abstract['session_title'],
            'abstract_title': abstract['abstract_title'],
            'ground_truth': abstract['ground_truth'],
            f'{model_name}_primary_drugs': json.dumps([]),
            f'{model_name}_secondary_drugs': json.dumps([]),
            f'{model_name}_comparator_drugs': json.dumps([]),
            f'{model_name}_reasoning': json.dumps([]),
            f'{model_name}_success': False,
            f'{model_name}_llm_calls': 0
        }
        return result_row


def process_abstracts_batch(abstracts: List[Dict], agent: DrugExtractionAgent,
                          model_name: str, output_file: str = None) -> pd.DataFrame:
    """Process a batch of abstracts and return results DataFrame.

    Args:
        abstracts: List of abstract dictionaries
        agent: Initialized DrugExtractionAgent
        model_name: Name of the model being used
        output_file: Optional output file path to save intermediate results

    Returns:
        DataFrame with processing results
    """
    print(f"Processing {len(abstracts)} abstracts with model: {model_name} (using 3 parallel threads)")

    results = []

    # Process abstracts in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_abstract, abstract, agent, model_name, i): i
            for i, abstract in enumerate(abstracts, 1)
        }

        # Collect results as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_row = future.result()
                results.append((index, result_row))  # Store with index to maintain order
                completed_count += 1

                # Save intermediate results every 10 abstracts
                if output_file and completed_count % 10 == 0:
                    # Sort results by index to maintain original order for intermediate saves
                    sorted_results = [result for _, result in sorted(results)]
                    temp_df = pd.DataFrame(sorted_results)
                    temp_df.to_csv(output_file, index=False)
                    print(f"Saved intermediate results ({completed_count}/{len(abstracts)}) to {output_file}")

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
    parser = argparse.ArgumentParser(description='Batch Process Drug Extraction')
    parser.add_argument('--input_file', default='data/drug_extraction_input.csv',
                       help='Input CSV file with abstracts')
    parser.add_argument('--output_file', default=None,
                       help='Output CSV file (default: auto-generated)')
    parser.add_argument('--num_abstracts', type=int, default=None,
                       help='Number of abstracts to process (default: all)')
    parser.add_argument('--randomize', action='store_true',
                       help='Randomize abstract selection')
    parser.add_argument('--model_name', default='default_model',
                       help='Name of the model for column naming')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"drug_batch_results_{args.model_name}_{timestamp}.csv"

    print("💊 Batch Drug Extraction Processor")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model name: {args.model_name}")
    print(f"Number of abstracts: {args.num_abstracts or 'all'}")
    print(f"Randomize: {args.randomize}")
    print()

    # Load abstracts
    print("Loading abstracts...")
    abstracts = load_abstracts_from_csv(
        args.input_file,
        max_abstracts=args.num_abstracts,
        randomize=args.randomize
    )

    if not abstracts:
        print("No abstracts loaded. Exiting.")
        return

    print(f"Loaded {len(abstracts)} abstracts")
    print()

    # Initialize agent
    print("Initializing Drug Extraction Agent...")
    agent = DrugExtractionAgent(agent_name=f"BatchProcessor_{args.model_name}")
    print("✓ Agent initialized successfully!")
    print()

    # Process abstracts
    results_df = process_abstracts_batch(abstracts, agent, args.model_name, args.output_file)

    # Summary
    total_processed = len(results_df)
    successful = results_df[f'{args.model_name}_success'].sum()
    success_rate = (successful / total_processed * 100) if total_processed > 0 else 0

    print()
    print("📊 Processing Summary:")
    print(f"Total abstracts processed: {total_processed}")
    print(f"Successful extractions: {int(successful)}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

