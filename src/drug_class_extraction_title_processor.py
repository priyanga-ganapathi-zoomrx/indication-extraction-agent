#!/usr/bin/env python3
"""
Drug Class Extraction from Title Processor

This script batch processes abstract titles from a CSV file to extract
explicit drug classes using the DrugClassExtractionTitleAgent.

Input: CSV file with abstract_id and abstract_title columns
Output: CSV file with extraction results added

Features:
- Reads abstract titles from CSV
- Extracts drug classes for each abstract title (one LLM call per record)
- Supports parallel processing with ThreadPoolExecutor
- Saves intermediate results incrementally
"""

import argparse
import concurrent.futures
import json
import os
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd

from src.drug_class_extraction_title_agent import DrugClassExtractionTitleAgent


def load_abstracts_from_csv(
    csv_path: str,
    max_rows: int = None,
    skip_rows: int = 0,
) -> List[Dict]:
    """Load abstract titles from CSV file.

    Args:
        csv_path: Path to the CSV file with abstract data
        max_rows: Maximum number of rows to load (None for all)
        skip_rows: Number of rows to skip from the beginning

    Returns:
        List of dictionaries with abstract data
    """
    abstracts = []

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return abstracts

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')

        # Skip rows if specified
        if skip_rows > 0:
            df = df.iloc[skip_rows:]

        # Limit rows if specified
        if max_rows is not None:
            df = df.head(max_rows)

        for idx, row in df.iterrows():
            # Get basic fields - handle BOM in column names
            abstract_id = row.get('abstract_id', row.get('\ufeffabstract_id', ''))
            abstract_title = row.get('abstract_title', '')

            abstract = {
                'abstract_id': str(abstract_id) if pd.notna(abstract_id) else '',
                'abstract_title': str(abstract_title) if pd.notna(abstract_title) else '',
                'original_row': row.to_dict(),
            }
            abstracts.append(abstract)

        return abstracts

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_single_abstract(
    abstract: Dict,
    extractor: DrugClassExtractionTitleAgent,
    index: int,
) -> Dict:
    """Extract drug classes from a single abstract title.

    Args:
        abstract: Abstract data dictionary
        extractor: Initialized DrugClassExtractionTitleAgent
        index: Index for logging

    Returns:
        Dictionary with original data plus extraction results
    """
    abstract_id = abstract['abstract_id']
    abstract_title = abstract['abstract_title']
    print(f"[{index}] Extracting from: {abstract_title[:80]}..." if len(abstract_title) > 80 else f"[{index}] Extracting from: {abstract_title}")

    result_row = abstract['original_row'].copy()

    try:
        # Run extraction
        extraction_result = extractor.invoke(
            abstract_title=abstract_title,
            abstract_id=abstract_id,
        )

        # Add extraction result columns
        result_row['drug_classes'] = json.dumps(extraction_result.get('drug_classes', ['NA']))
        result_row['raw_json_response'] = extraction_result.get('raw_json_response', '')
        result_row['extraction_success'] = extraction_result.get('extraction_success', False)
        result_row['llm_calls'] = extraction_result.get('llm_calls', 0)

        # Log result summary
        drug_classes = extraction_result.get('drug_classes', ['NA'])
        status = "✓" if extraction_result.get('extraction_success', False) else "✗"
        print(f"  {status} Extracted: {drug_classes}")

    except Exception as e:
        print(f"  ✗ Extraction error for {abstract_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Create default error JSON response
        error_json = {
            "drug_classes": ["NA"],
            "source": "abstract_title",
            "confidence_score": 0.0,
            "reasoning": f'Extraction failed: {str(e)}',
            "extraction_details": [],
        }
        result_row['drug_classes'] = json.dumps(['NA'])
        result_row['raw_json_response'] = json.dumps(error_json, indent=2)
        result_row['extraction_success'] = False
        result_row['llm_calls'] = 0

    return result_row


def extract_batch(
    abstracts: List[Dict],
    extractor: DrugClassExtractionTitleAgent,
    output_file: str = None,
    num_workers: int = 3,
    save_interval: int = 10,
) -> pd.DataFrame:
    """Extract drug classes from a batch of abstracts and return results DataFrame.

    Args:
        abstracts: List of abstract dictionaries
        extractor: Initialized DrugClassExtractionTitleAgent
        output_file: Optional output file path to save intermediate results
        num_workers: Number of parallel workers (default: 3)
        save_interval: Save intermediate results every N records (default: 10)

    Returns:
        DataFrame with extraction results
    """
    print(f"Extracting drug classes from {len(abstracts)} abstracts (using {num_workers} parallel threads)")

    results = []

    # Process abstracts in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(
                extract_single_abstract,
                abstract,
                extractor,
                i
            ): i
            for i, abstract in enumerate(abstracts, 1)
        }

        # Collect results as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_row = future.result()
                results.append((index, result_row))
                completed_count += 1

                # Save intermediate results periodically
                if output_file and completed_count % save_interval == 0:
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
    """Main extraction processing function."""
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Extract Drug Classes from Abstract Titles')
    parser.add_argument('--input_file', default='data/drug_class_input_regimen_150.csv',
                        help='Input CSV file with abstract titles')
    parser.add_argument('--output_file', default=None,
                        help='Output CSV file (default: auto-generated)')
    parser.add_argument('--llm_model', default="gemini/gemini-3-flash-preview",
                        help='LLM model name to use for extraction (default: from settings)')
    parser.add_argument('--temperature', type=float, default=0,
                        help='LLM temperature (default: 0)')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='LLM max tokens (default: from settings)')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Maximum number of rows to process (default: all)')
    parser.add_argument('--skip_rows', type=int, default=0,
                        help='Number of rows to skip from the beginning')
    parser.add_argument('--num_workers', type=int, default=3,
                        help='Number of parallel workers (default: 3)')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save intermediate results every N records (default: 10)')
    parser.add_argument('--enable_caching', action='store_true',
                        help='Enable prompt caching for Anthropic models (reduces costs)')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        # Get model name for filename (extract just model name without provider)
        model_suffix = ""
        if args.llm_model:
            model_suffix = f"_{args.llm_model.split('/')[-1]}"
        args.output_file = f"data/{base_name}_extracted{model_suffix}_{timestamp}.csv"

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print("🔬 Drug Class Extraction from Title Processor")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"LLM model: {args.llm_model or 'from settings'}")
    print(f"Temperature: {args.temperature}")
    print(f"Max rows: {args.max_rows or 'all'}")
    print(f"Skip rows: {args.skip_rows}")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Save interval: {args.save_interval}")
    print(f"Prompt caching: {'enabled' if args.enable_caching else 'disabled'}")
    print()

    # Load abstract titles
    print("Loading abstract titles...")
    abstracts = load_abstracts_from_csv(
        args.input_file,
        max_rows=args.max_rows,
        skip_rows=args.skip_rows,
    )

    if not abstracts:
        print("No abstracts loaded. Exiting.")
        return

    print(f"Loaded {len(abstracts)} abstracts")
    print()

    # Initialize extraction agent
    print("Initializing Drug Class Extraction Title Agent...")
    extractor = DrugClassExtractionTitleAgent(
        agent_name="DrugClassExtractionTitleProcessor",
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_caching=args.enable_caching,
    )
    print("✓ Extraction Agent initialized successfully!")
    print()

    # Extract drug classes
    print("-" * 80)
    results_df = extract_batch(
        abstracts,
        extractor,
        args.output_file,
        args.num_workers,
        args.save_interval,
    )

    # Summary
    print()
    print("=" * 80)
    total_processed = len(results_df)

    if 'extraction_success' in results_df.columns:
        success_count = results_df['extraction_success'].sum()
        fail_count = total_processed - success_count

        print("📊 Extraction Summary:")
        print(f"Total abstracts processed: {total_processed}")
        print(f"  Success: {int(success_count)} ({success_count/total_processed*100:.1f}%)")
        print(f"  Failed: {int(fail_count)} ({fail_count/total_processed*100:.1f}%)")

    # Count NA vs extracted
    if 'drug_classes' in results_df.columns:
        na_count = 0
        extracted_count = 0
        for dc in results_df['drug_classes']:
            try:
                classes = json.loads(dc) if isinstance(dc, str) else dc
                if classes == ['NA'] or classes == 'NA':
                    na_count += 1
                else:
                    extracted_count += 1
            except (json.JSONDecodeError, TypeError):
                na_count += 1
        
        print(f"  With drug classes: {extracted_count} ({extracted_count/total_processed*100:.1f}%)")
        print(f"  No drug class (NA): {na_count} ({na_count/total_processed*100:.1f}%)")

    # Total LLM calls
    if 'llm_calls' in results_df.columns:
        total_llm_calls = results_df['llm_calls'].sum()
        print(f"  Total LLM calls: {int(total_llm_calls)}")

    print()
    print(f"Results saved to: {args.output_file}")

    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print()
    print(f"⏱️  Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")


if __name__ == "__main__":
    main()

