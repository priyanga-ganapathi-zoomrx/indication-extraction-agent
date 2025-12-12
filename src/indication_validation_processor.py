#!/usr/bin/env python3
"""
Indication Validation Processor

This script validates previously extracted indications from a CSV file
using the IndicationValidationAgent. It reads extraction results and
flags potential errors for manual QC review.

Input: CSV file with extraction results (e.g., data/indication_validation.csv)
Output: CSV file with validation results added
"""

import argparse
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List
import concurrent.futures

from src.indication_validator_agent import IndicationValidationAgent
from src.config import settings


def load_extractions_from_csv(
    csv_path: str,
    model_name: str = "model",
    max_rows: int = None,
    skip_rows: int = 0,
) -> List[Dict]:
    """Load extraction results from CSV file.

    Args:
        csv_path: Path to the CSV file with extraction results
        model_name: Model name prefix used in column names (fallbacks to 'model')
        max_rows: Maximum number of rows to load (None for all)
        skip_rows: Number of rows to skip from the beginning

    Returns:
        List of dictionaries with extraction data
    """
    extractions = []

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return extractions

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Skip rows if specified
        if skip_rows > 0:
            df = df.iloc[skip_rows:]
        
        # Limit rows if specified
        if max_rows is not None:
            df = df.head(max_rows)

        model_prefixes = [model_name, "model"]

        def get_prefixed_value(row, suffix: str, default=None):
            """Fetch a value using the provided model prefix with fallbacks."""
            for prefix in model_prefixes:
                col = f"{prefix}_{suffix}"
                if col in row and pd.notna(row[col]):
                    return row[col]
            if suffix in row and pd.notna(row[suffix]):
                return row[suffix]
            return default

        for idx, row in df.iterrows():
            # Get column values, handling potential column name variations
            abstract_id = row.get('abstract_id', row.get('\ufeffabstract_id', ''))
            session_title = row.get('session_title', '')
            abstract_title = row.get('abstract_title', '')
            
            # Get extraction result columns
            indication = get_prefixed_value(row, 'indication_response', '')
            selected_source = get_prefixed_value(row, 'selected_source', '')
            confidence_score = get_prefixed_value(row, 'confidence_score', None)
            reasoning = get_prefixed_value(row, 'reasoning', '')
            rules_retrieved = get_prefixed_value(row, 'rules_retrieved', '[]')
            components_identified = get_prefixed_value(row, 'components_identified', '[]')
            success = get_prefixed_value(row, 'success', True)

            # Normalize boolean-like values
            if isinstance(success, str):
                success = success.strip().lower() in ("true", "1", "yes", "y", "t")
            
            # Parse JSON strings if needed
            if isinstance(rules_retrieved, str):
                try:
                    rules_retrieved = json.loads(rules_retrieved)
                except json.JSONDecodeError:
                    rules_retrieved = []
            
            if isinstance(components_identified, str):
                try:
                    components_identified = json.loads(components_identified)
                except json.JSONDecodeError:
                    components_identified = []

            extraction = {
                'abstract_id': abstract_id,
                'session_title': session_title,
                'abstract_title': abstract_title,
                'extraction_result': {
                    'indication': str(indication) if pd.notna(indication) else '',
                    'success': bool(success) if pd.notna(success) else False,
                    'selected_source': str(selected_source) if pd.notna(selected_source) else '',
                    'confidence_score': float(confidence_score) if pd.notna(confidence_score) else None,
                    'reasoning': str(reasoning) if pd.notna(reasoning) else '',
                    'rules_retrieved': rules_retrieved if rules_retrieved else [],
                    'components_identified': components_identified if components_identified else [],
                },
                'original_row': row.to_dict(),  # Keep original row for output
            }
            extractions.append(extraction)

        return extractions

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return []


def validate_single_extraction(
    extraction: Dict,
    validator: IndicationValidationAgent,
    index: int,
) -> Dict:
    """Validate a single extraction result.

    Args:
        extraction: Extraction data dictionary
        validator: Initialized IndicationValidationAgent
        index: Index for logging

    Returns:
        Dictionary with original data plus validation results
    """
    abstract_id = extraction['abstract_id']
    print(f"Validating extraction {index}: ID {abstract_id}")

    result_row = extraction['original_row'].copy()

    try:
        # Skip validation if extraction failed
        if not extraction['extraction_result'].get('success', False):
            print("  Skipping validation - extraction failed")
            result_row['validation_status'] = 'FAIL'
            result_row['validation_confidence'] = 0.0
            result_row['validation_issues'] = json.dumps([{
                'check_type': 'extraction_failure',
                'severity': 'high',
                'description': 'Extraction failed - no indication generated',
                'evidence': '',
                'component': ''
            }])
            result_row['validation_reasoning'] = 'Extraction failed, validation skipped'
            result_row['validation_llm_calls'] = 0
            result_row['needs_qc'] = True
            return result_row

        # Skip validation if indication is empty
        indication = extraction['extraction_result'].get('indication', '')
        if not indication or indication.strip() == '':
            print("  Skipping validation - empty indication")
            result_row['validation_status'] = 'FAIL'
            result_row['validation_confidence'] = 0.0
            result_row['validation_issues'] = json.dumps([{
                'check_type': 'empty_indication',
                'severity': 'high',
                'description': 'Indication is empty',
                'evidence': '',
                'component': ''
            }])
            result_row['validation_reasoning'] = 'Empty indication, validation skipped'
            result_row['validation_llm_calls'] = 0
            result_row['needs_qc'] = True
            return result_row

        # Run validation
        validation_result = validator.invoke(
            session_title=extraction['session_title'],
            abstract_title=extraction['abstract_title'],
            extraction_result=extraction['extraction_result'],
            abstract_id=abstract_id
        )

        # Add validation columns
        result_row['validation_status'] = validation_result.get('validation_status', 'REVIEW')
        result_row['validation_confidence'] = validation_result.get('validation_confidence', 0.0)
        result_row['validation_issues'] = json.dumps(validation_result.get('issues_found', []))
        result_row['validation_reasoning'] = validation_result.get('validation_reasoning', '')
        result_row['validation_llm_calls'] = validation_result.get('llm_calls', 0)

        # Determine if QC is needed
        validation_status = validation_result.get('validation_status', 'REVIEW')
        issues_found = validation_result.get('issues_found', [])
        needs_qc = (
            validation_status == "FAIL" or
            validation_status == "REVIEW" or
            any(issue.get('severity') == 'high' for issue in issues_found)
        )
        result_row['needs_qc'] = needs_qc

        print(f"  Validation complete: {validation_status} (needs_qc: {needs_qc})")

    except Exception as e:
        print(f"  Validation error for abstract {abstract_id}: {e}")
        result_row['validation_status'] = 'REVIEW'
        result_row['validation_confidence'] = 0.0
        result_row['validation_issues'] = json.dumps([{
            'check_type': 'system_error',
            'severity': 'high',
            'description': f'Validation failed: {str(e)}',
            'evidence': '',
            'component': ''
        }])
        result_row['validation_reasoning'] = f'Validation error: {str(e)}'
        result_row['validation_llm_calls'] = 0
        result_row['needs_qc'] = True

    return result_row


def validate_extractions_batch(
    extractions: List[Dict],
    validator: IndicationValidationAgent,
    output_file: str = None,
    num_workers: int = 3,
) -> pd.DataFrame:
    """Validate a batch of extractions and return results DataFrame.

    Args:
        extractions: List of extraction dictionaries
        validator: Initialized IndicationValidationAgent
        output_file: Optional output file path to save intermediate results
        num_workers: Number of parallel workers (default: 3)

    Returns:
        DataFrame with validation results
    """
    print(f"Validating {len(extractions)} extractions (using {num_workers} parallel threads)")

    results = []

    # Process extractions in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(validate_single_extraction, extraction, validator, i): i
            for i, extraction in enumerate(extractions, 1)
        }

        # Collect results as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_row = future.result()
                results.append((index, result_row))
                completed_count += 1

                # Save intermediate results every 10 extractions
                if output_file and completed_count % 10 == 0:
                    sorted_results = [result for _, result in sorted(results)]
                    temp_df = pd.DataFrame(sorted_results)
                    temp_df.to_csv(output_file, index=False)
                    print(f"Saved intermediate results ({completed_count}/{len(extractions)}) to {output_file}")

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
    """Main validation processing function."""
    parser = argparse.ArgumentParser(description='Validate Indication Extractions')
    parser.add_argument('--input_file', default='data/indication_validation_input_failure.csv',
                       help='Input CSV file with extraction results')
    parser.add_argument('--output_file', default=None,
                       help='Output CSV file (default: auto-generated)')
    parser.add_argument('--model_name', default='model',
                       help="Model name prefix used in input columns (falls back to 'model')")
    parser.add_argument('--llm_model', default="anthropic/claude-sonnet-4-5-20250929",
                       help='LLM model name to use for validation calls')
    parser.add_argument('--max_rows', type=int, default=None,
                       help='Maximum number of rows to validate (default: all)')
    parser.add_argument('--skip_rows', type=int, default=0,
                       help='Number of rows to skip from the beginning')
    parser.add_argument('--num_workers', type=int, default=3,
                       help='Number of parallel workers (default: 3)')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = f"data/{base_name}_validated_{timestamp}.csv"

    print("🔍 Indication Validation Processor")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model name: {args.model_name}")
    print(f"LLM model: {args.llm_model}")
    print(f"Max rows: {args.max_rows or 'all'}")
    print(f"Skip rows: {args.skip_rows}")
    print(f"Parallel workers: {args.num_workers}")
    print()

    # Load extraction results
    print("Loading extraction results...")
    extractions = load_extractions_from_csv(
        args.input_file,
        model_name=args.model_name,
        max_rows=args.max_rows,
        skip_rows=args.skip_rows
    )

    if not extractions:
        print("No extraction results loaded. Exiting.")
        return

    print(f"Loaded {len(extractions)} extraction results")
    print()

    # Initialize validation agent
    print("Initializing Indication Validation Agent...")
    validator = IndicationValidationAgent(agent_name="ValidationProcessor", llm_model=args.llm_model)
    print("✓ Validation Agent initialized successfully!")
    print()

    # Validate extractions
    results_df = validate_extractions_batch(
        extractions,
        validator,
        args.output_file,
        args.num_workers
    )

    # Summary
    total_validated = len(results_df)
    
    if 'validation_status' in results_df.columns:
        pass_count = (results_df['validation_status'] == 'PASS').sum()
        review_count = (results_df['validation_status'] == 'REVIEW').sum()
        fail_count = (results_df['validation_status'] == 'FAIL').sum()
        
        print()
        print("📊 Validation Summary:")
        print(f"Total extractions validated: {total_validated}")
        print(f"  PASS: {int(pass_count)} ({pass_count/total_validated*100:.1f}%)")
        print(f"  REVIEW: {int(review_count)} ({review_count/total_validated*100:.1f}%)")
        print(f"  FAIL: {int(fail_count)} ({fail_count/total_validated*100:.1f}%)")
        
        if 'needs_qc' in results_df.columns:
            needs_qc_count = results_df['needs_qc'].sum()
            print(f"  Flagged for QC: {int(needs_qc_count)} ({needs_qc_count/total_validated*100:.1f}%)")
    
    print()
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

