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
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                abstracts.append({
                    'abstract_id': row.get('abstract_id', row.get('\ufeffabstract_id', '')),
                    'session_title': row.get('Session title', ''),
                    'abstract_title': row.get('abstract Title', ''),
                    'ground_truth': row.get('Ground Truth', ''),
                })

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

        # Look for JSON in the response (with or without code blocks)
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly in content
            json_match = re.search(r'\{[^}]*"Primary Drugs"[^}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content

        try:
            parsed = json.loads(json_str)
            
            # Handle different possible key formats
            primary_drugs = (
                parsed.get('Primary Drugs', []) or 
                parsed.get('primary_drugs', []) or 
                parsed.get('Primary', []) or
                []
            )
            secondary_drugs = (
                parsed.get('Secondary Drugs', []) or 
                parsed.get('secondary_drugs', []) or 
                parsed.get('Secondary', []) or
                []
            )
            comparator_drugs = (
                parsed.get('Comparator Drugs', []) or 
                parsed.get('comparator_drugs', []) or 
                parsed.get('Comparator', []) or
                []
            )
            
            return {
                'primary_drugs': primary_drugs if isinstance(primary_drugs, list) else [],
                'secondary_drugs': secondary_drugs if isinstance(secondary_drugs, list) else [],
                'comparator_drugs': comparator_drugs if isinstance(comparator_drugs, list) else [],
                'success': True,
            }
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Content: {content[:200]}...")
            return {
                'primary_drugs': [],
                'secondary_drugs': [],
                'comparator_drugs': [],
                'success': False,
            }

    except Exception as e:
        print(f"Error extracting drugs: {e}")
        return {
            'primary_drugs': [],
            'secondary_drugs': [],
            'comparator_drugs': [],
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
    parser.add_argument('--input_file', default='data/abstract_titles.csv',
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

