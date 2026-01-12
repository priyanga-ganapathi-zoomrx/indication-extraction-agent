#!/usr/bin/env python3
"""
Batch Processor for Indication Extraction Agent

This script processes multiple abstracts using the indication extraction agent
and saves results to CSV format compatible with the analysis script.
"""

import csv
import json
import os
import re
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import argparse
import concurrent.futures

from src.agents.indication import IndicationExtractionAgent


def load_abstracts_from_csv(csv_path: str, max_abstracts: int = None, randomize: bool = False) -> List[Dict]:
    """Load abstracts from CSV file.

    Args:
        csv_path: Path to the CSV file
        max_abstracts: Maximum number of abstracts to load
        randomize: Whether to randomize the selection

    Returns:
        List of dictionaries with abstract data (all original fields preserved)
    """
    abstracts = []

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return abstracts

    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Keep all original fields from the CSV
                abstract_data = dict(row)
                
                # Normalize key field names for agent invocation (handle BOM and column name variations)
                # Support both snake_case and original column names
                abstract_data['_abstract_id'] = row.get('abstract_id', row.get('\ufeffabstract_id', ''))
                abstract_data['_session_title'] = row.get('session_title', row.get('Session title', ''))
                abstract_data['_abstract_title'] = row.get('abstract_title', row.get('abstract Title', ''))
                
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


def extract_indication_from_response(result: Dict) -> Dict[str, Any]:
    """Extract indication and additional fields from agent response.

    Args:
        result: Agent invocation result

    Returns:
        Dictionary with extracted fields: indication, success, selected_source, reasoning, 
        confidence_score, rules_retrieved, components_identified, and quality_metrics
    """
    try:
        # Get the final message from the agent
        messages = result.get('messages', [])
        if not messages:
            return {
                'indication': '',
                'success': False,
                'selected_source': '',
                'reasoning': '',
                'confidence_score': None,
                'rules_retrieved': [],
                'components_identified': [],
                'quality_metrics_completeness': None,
                'quality_metrics_rule_adherence': None,
                'quality_metrics_clinical_accuracy': None,
                'quality_metrics_formatting_compliance': None
            }

        final_message = messages[-1]
        content = getattr(final_message, 'content', '')

        if not content or content.startswith("I encountered an error"):
            return {
                'indication': '',
                'success': False,
                'selected_source': '',
                'reasoning': '',
                'confidence_score': None,
                'rules_retrieved': [],
                'components_identified': [],
                'quality_metrics_completeness': None,
                'quality_metrics_rule_adherence': None,
                'quality_metrics_clinical_accuracy': None,
                'quality_metrics_formatting_compliance': None
            }

        # Try to parse JSON response if present
        # Look for JSON in the response
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if 'generated_indication' in parsed:
                    # Extract quality metrics
                    quality_metrics = parsed.get('quality_metrics', {})
                    
                    return {
                        'indication': str(parsed.get('generated_indication', '')).strip(),
                        'success': True,
                        'selected_source': parsed.get('selected_source', ''),
                        'reasoning': parsed.get('reasoning', ''),
                        'confidence_score': parsed.get('confidence_score', None),
                        'rules_retrieved': parsed.get('rules_retrieved', []),
                        'components_identified': parsed.get('components_identified', []),
                        'quality_metrics_completeness': quality_metrics.get('completeness', None),
                        'quality_metrics_rule_adherence': quality_metrics.get('rule_adherence', None),
                        'quality_metrics_clinical_accuracy': quality_metrics.get('clinical_accuracy', None),
                        'quality_metrics_formatting_compliance': quality_metrics.get('formatting_compliance', None)
                    }
            except json.JSONDecodeError:
                pass

        # Fallback: extract from plain text
        # Look for indication patterns
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Indication:') or line.startswith('Medical Indication:'):
                indication = line.split(':', 1)[1].strip()
                return {
                    'indication': indication,
                    'success': True,
                    'selected_source': '',
                    'reasoning': '',
                    'confidence_score': None,
                    'rules_retrieved': [],
                    'components_identified': [],
                    'quality_metrics_completeness': None,
                    'quality_metrics_rule_adherence': None,
                    'quality_metrics_clinical_accuracy': None,
                    'quality_metrics_formatting_compliance': None
                }
            elif len(line) > 10 and not line.startswith('Based on') and not line.startswith('The'):
                # Assume this is the indication
                return {
                    'indication': line,
                    'success': True,
                    'selected_source': '',
                    'reasoning': '',
                    'confidence_score': None,
                    'rules_retrieved': [],
                    'components_identified': [],
                    'quality_metrics_completeness': None,
                    'quality_metrics_rule_adherence': None,
                    'quality_metrics_clinical_accuracy': None,
                    'quality_metrics_formatting_compliance': None
                }

        # Last resort: return the entire content if it's reasonable length
        if len(content) > 10 and len(content) < 500:
            return {
                'indication': content.strip(),
                'success': True,
                'selected_source': '',
                'reasoning': '',
                'confidence_score': None,
                'rules_retrieved': [],
                'components_identified': [],
                'quality_metrics_completeness': None,
                'quality_metrics_rule_adherence': None,
                'quality_metrics_clinical_accuracy': None,
                'quality_metrics_formatting_compliance': None
            }

        return {
            'indication': '',
            'success': False,
            'selected_source': '',
            'reasoning': '',
            'confidence_score': None,
            'rules_retrieved': [],
            'components_identified': [],
            'quality_metrics_completeness': None,
            'quality_metrics_rule_adherence': None,
            'quality_metrics_clinical_accuracy': None,
            'quality_metrics_formatting_compliance': None
        }

    except Exception as e:
        print(f"Error extracting indication: {e}")
        return {
            'indication': '',
            'success': False,
            'selected_source': '',
            'reasoning': '',
            'confidence_score': None,
            'rules_retrieved': [],
            'components_identified': [],
            'quality_metrics_completeness': None,
            'quality_metrics_rule_adherence': None,
            'quality_metrics_clinical_accuracy': None,
            'quality_metrics_formatting_compliance': None
        }


def process_single_abstract(abstract: Dict, agent: IndicationExtractionAgent,
                           model_name: str, index: int) -> Dict:
    """Process a single abstract and return the result.

    Args:
        abstract: Abstract dictionary (contains all original CSV fields plus normalized keys)
        agent: Initialized IndicationExtractionAgent
        model_name: Name of the model being used
        index: Index of the abstract for logging

    Returns:
        Dictionary with processing result (all original fields + model output fields)
    """
    print(f"Processing abstract {index}: ID {abstract['_abstract_id']}")

    # Start with all original fields from the input CSV (exclude internal normalized keys)
    result_row = {k: v for k, v in abstract.items() if not k.startswith('_')}

    try:
        # Invoke the agent using normalized keys
        result = agent.invoke(
            abstract_title=abstract['_abstract_title'],
            session_title=abstract['_session_title'],
            abstract_id=abstract['_abstract_id']
        )

        # Extract indication and additional fields
        extracted_data = extract_indication_from_response(result)

        # Add model output fields
        result_row.update({
            f'{model_name}_indication_response': extracted_data['indication'],
            f'{model_name}_success': extracted_data['success'],
            f'{model_name}_selected_source': extracted_data['selected_source'],
            f'{model_name}_confidence_score': extracted_data['confidence_score'],
            f'{model_name}_reasoning': extracted_data['reasoning'],
            f'{model_name}_rules_retrieved': json.dumps(extracted_data['rules_retrieved'], indent=2),
            f'{model_name}_components_identified': json.dumps(extracted_data['components_identified'], indent=2),
            f'{model_name}_quality_metrics_completeness': extracted_data['quality_metrics_completeness'],
            f'{model_name}_quality_metrics_rule_adherence': extracted_data['quality_metrics_rule_adherence'],
            f'{model_name}_quality_metrics_clinical_accuracy': extracted_data['quality_metrics_clinical_accuracy'],
            f'{model_name}_quality_metrics_formatting_compliance': extracted_data['quality_metrics_formatting_compliance'],
            f'{model_name}_llm_calls': result.get('llm_calls', 0)
        })

        return result_row

    except Exception as e:
        print(f"Error processing abstract {abstract['_abstract_id']}: {e}")
        # Add error result fields
        result_row.update({
            f'{model_name}_indication_response': "",
            f'{model_name}_success': False,
            f'{model_name}_selected_source': "",
            f'{model_name}_confidence_score': None,
            f'{model_name}_reasoning': "",
            f'{model_name}_rules_retrieved': "[]",
            f'{model_name}_components_identified': "[]",
            f'{model_name}_quality_metrics_completeness': None,
            f'{model_name}_quality_metrics_rule_adherence': None,
            f'{model_name}_quality_metrics_clinical_accuracy': None,
            f'{model_name}_quality_metrics_formatting_compliance': None,
            f'{model_name}_llm_calls': 0
        })
        return result_row


def process_abstracts_batch(abstracts: List[Dict], agent: IndicationExtractionAgent,
                          model_name: str, output_file: str = None,
                          num_workers: int = 3) -> pd.DataFrame:
    """Process a batch of abstracts and return results DataFrame.

    Args:
        abstracts: List of abstract dictionaries
        agent: Initialized IndicationExtractionAgent
        model_name: Name of the model being used
        output_file: Optional output file path to save intermediate results
        num_workers: Number of parallel workers (default: 3)

    Returns:
        DataFrame with processing results
    """
    print(f"Processing {len(abstracts)} abstracts with model: {model_name} (using {num_workers} parallel threads)")

    results = []

    # Process abstracts in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
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
    parser = argparse.ArgumentParser(description='Batch Process Indication Extraction')
    parser.add_argument('--input_file', default='data/abstract_titles.csv',
                       help='Input CSV file with abstracts')
    parser.add_argument('--output_file', default=None,
                       help='Output CSV file (default: auto-generated)')
    parser.add_argument('--num_abstracts', type=int, default=None,
                       help='Number of abstracts to process (default: all)') 
    parser.add_argument('--randomize', action='store_true',
                       help='Randomize abstract selection')
    parser.add_argument('--model_name', default='gemini-2-5-pro',
                       help='Name of the model for column naming')
    parser.add_argument('--num_workers', type=int, default=3,
                       help='Number of parallel workers (default: 3)')
    parser.add_argument('--enable_caching', action='store_true',
                       help='Enable prompt caching for Gemini models (reduces costs)')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"batch_results_{args.model_name}_{timestamp}.csv"

    print("🏭 Batch Indication Extraction Processor")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model name: {args.model_name}")
    print(f"Number of abstracts: {args.num_abstracts or 'all'}")
    print(f"Randomize: {args.randomize}")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Prompt caching: {'enabled' if args.enable_caching else 'disabled'}")
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
    print("Initializing Indication Extraction Agent...")
    agent = IndicationExtractionAgent(
        agent_name=f"BatchProcessor_{args.model_name}",
        enable_caching=args.enable_caching
    )
    print("✓ Agent initialized successfully!")
    print()

    # Process abstracts
    results_df = process_abstracts_batch(
        abstracts, agent, args.model_name, args.output_file, args.num_workers
    )

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

