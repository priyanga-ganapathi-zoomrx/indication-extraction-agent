#!/usr/bin/env python3
"""
Drug Class Validation Processor

This script validates previously extracted drug classes from a CSV file
using the DrugClassValidationAgent. It reads extraction results and
flags potential errors for manual QC review.

Input: CSV file with extraction results (from drug_class_react_batch_processor.py)
Output: CSV file with validation results added

Features:
- Reads extraction results from CSV
- Loads search results from cache JSON file
- Validates each extraction using the 3-check structure
- Supports parallel processing with ThreadPoolExecutor
- Saves intermediate results incrementally
"""

import argparse
import concurrent.futures
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from src.drug_class_validation_agent import DrugClassValidationAgent


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
    drug_data = drugs_cache.get(drug_name, {})

    if not drug_data:
        return [], []

    # Get drug class search results
    drug_class_results = drug_data.get("drug_class_search", {}).get("results", [])

    # Get firm-specific results
    firms_key = get_firms_key(firms)
    firm_results = drug_data.get("firm_searches", {}).get(firms_key, {}).get("results", [])

    return drug_class_results, firm_results


def load_extractions_from_csv(
    csv_path: str,
    max_rows: int = None,
    skip_rows: int = 0,
) -> List[Dict]:
    """Load extraction results from CSV file.

    Args:
        csv_path: Path to the CSV file with extraction results
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

        for idx, row in df.iterrows():
            # Get basic fields
            abstract_id = row.get('abstract_id', row.get('\ufeffabstract_id', ''))
            abstract_title = row.get('abstract_title', '')
            full_abstract = row.get('full_abstract', '')
            firm = row.get('firm', '')
            
            # Use flattened_components for drug names (JSON array like ["Drug1", "Drug2"])
            # Falls back to drug_name if flattened_components not available
            flattened_components = row.get('flattened_components', '')
            drug_name_raw = row.get('drug_name', '')

            # Parse firms list
            if isinstance(firm, str) and firm:
                firms = [f.strip() for f in firm.replace(';', ',').split(',') if f.strip()]
            else:
                firms = []

            # Get extraction result columns
            drug_classes_grouped = row.get('drug_classes_grouped', '{}')
            selected_sources_grouped = row.get('selected_sources_grouped', '{}')
            confidence_scores_grouped = row.get('confidence_scores_grouped', '{}')
            reasoning_grouped = row.get('reasoning_grouped', '{}')
            extraction_details_grouped = row.get('extraction_details_grouped', '{}')
            drug_classes = row.get('drug_classes', '["NA"]')
            success = row.get('success', True)

            # Parse JSON fields
            def safe_json_loads(val, default):
                if pd.isna(val) or val == '':
                    return default
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except json.JSONDecodeError:
                        return default
                return val

            drug_classes_grouped = safe_json_loads(drug_classes_grouped, {})
            selected_sources_grouped = safe_json_loads(selected_sources_grouped, {})
            confidence_scores_grouped = safe_json_loads(confidence_scores_grouped, {})
            reasoning_grouped = safe_json_loads(reasoning_grouped, {})
            extraction_details_grouped = safe_json_loads(extraction_details_grouped, {})
            drug_classes_list = safe_json_loads(drug_classes, ["NA"])
            
            # Parse flattened_components as JSON array for drug names
            flattened_components_list = safe_json_loads(flattened_components, [])
            
            # Determine drug_name: prefer flattened_components, fallback to drug_name column
            if flattened_components_list:
                # Join components for display, keep list for processing
                drug_name = ', '.join(flattened_components_list)
            else:
                drug_name = drug_name_raw

            # Normalize boolean-like values
            if isinstance(success, str):
                success = success.strip().lower() in ("true", "1", "yes", "y", "t")

            extraction = {
                'abstract_id': abstract_id,
                'abstract_title': abstract_title,
                'drug_name': drug_name,
                'flattened_components': flattened_components_list,  # Store as list for processing
                'full_abstract': full_abstract,
                'firms': firms,
                'extraction_result': {
                    'drug_classes': drug_classes_list,
                    'drug_classes_grouped': drug_classes_grouped,
                    'selected_sources_grouped': selected_sources_grouped,
                    'confidence_scores_grouped': confidence_scores_grouped,
                    'reasoning_grouped': reasoning_grouped,
                    'extraction_details_grouped': extraction_details_grouped,
                    'success': bool(success) if pd.notna(success) else False,
                },
                'original_row': row.to_dict(),
            }
            extractions.append(extraction)

        return extractions

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return []


def validate_single_drug(
    drug_name: str,
    extraction: Dict,
    cache_data: Dict,
    validator: DrugClassValidationAgent,
    drug_classes_grouped: Dict,
    selected_sources_grouped: Dict,
    confidence_scores_grouped: Dict,
    reasoning_grouped: Dict,
    extraction_details_grouped: Dict,
) -> Dict:
    """Validate extraction for a single drug within a row.

    Args:
        drug_name: The specific drug to validate
        extraction: Full extraction data dictionary
        cache_data: Cache dictionary with search results
        validator: Initialized DrugClassValidationAgent
        drug_classes_grouped: Grouped drug classes dict
        selected_sources_grouped: Grouped selected sources dict
        confidence_scores_grouped: Grouped confidence scores dict
        reasoning_grouped: Grouped reasoning dict
        extraction_details_grouped: Grouped extraction details dict

    Returns:
        Dictionary with validation results for this drug
    """
    # Get search results from cache for this specific drug
    drug_class_results, firm_results = get_search_results_from_cache(
        cache_data, drug_name, extraction['firms']
    )
    all_search_results = (drug_class_results or []) + (firm_results or [])

    # Build extraction result for this specific drug
    validation_extraction_result = {
        'drug_classes': drug_classes_grouped.get(drug_name, ['NA']),
        'selected_sources': selected_sources_grouped.get(drug_name, []),
        'confidence_score': confidence_scores_grouped.get(drug_name),
        'reasoning': reasoning_grouped.get(drug_name, ''),
        'extraction_details': extraction_details_grouped.get(drug_name, []),
    }

    # Run validation
    validation_result = validator.invoke(
        drug_name=drug_name,
        abstract_title=extraction['abstract_title'],
        full_abstract=extraction['full_abstract'],
        search_results=all_search_results,
        extraction_result=validation_extraction_result,
        abstract_id=extraction['abstract_id'],
    )

    return {
        'status': validation_result.get('validation_status', 'REVIEW'),
        'confidence': validation_result.get('validation_confidence', 0.0),
        'extraction_performed': validation_result.get('extraction_performed', False),
        'extracted_drug_classes': validation_result.get('extracted_drug_classes', []),
        'missed_drug_classes': validation_result.get('missed_drug_classes', []),
        'issues': validation_result.get('issues_found', []),
        'checks': validation_result.get('checks_performed', {}),
        'reasoning': validation_result.get('validation_reasoning', ''),
        'llm_calls': validation_result.get('llm_calls', 0),
        'validation_success': validation_result.get('validation_success', True),
        'raw_llm_response': validation_result.get('raw_llm_response'),
    }


def validate_single_extraction(
    extraction: Dict,
    cache_data: Dict,
    validator: DrugClassValidationAgent,
    index: int,
) -> Dict:
    """Validate a single extraction result (may contain multiple drugs).

    For rows with multiple drugs in drug_classes_grouped, validates each drug
    separately and produces grouped validation output columns.

    Args:
        extraction: Extraction data dictionary
        cache_data: Cache dictionary with search results
        validator: Initialized DrugClassValidationAgent
        index: Index for logging

    Returns:
        Dictionary with original data plus validation results
    """
    abstract_id = extraction['abstract_id']
    drug_name_col = extraction['drug_name']  # May be comma-separated or regimen name
    print(f"[{index}] Validating extraction for: {drug_name_col} (ID: {abstract_id})")

    result_row = extraction['original_row'].copy()

    try:
        # Get extraction result
        extraction_result = extraction['extraction_result']

        # Log if original extraction failed - but continue with validation
        # The validator can still perform grounded search extraction to find drug classes
        if not extraction_result.get('success', False):
            print("  Note: Original extraction failed - will attempt grounded search extraction")

        # Get grouped extraction data
        drug_classes_grouped = extraction_result.get('drug_classes_grouped', {})
        reasoning_grouped = extraction_result.get('reasoning_grouped', {})
        extraction_details_grouped = extraction_result.get('extraction_details_grouped', {})
        selected_sources_grouped = extraction_result.get('selected_sources_grouped', {})
        confidence_scores_grouped = extraction_result.get('confidence_scores_grouped', {})

        # Determine which drugs to validate
        # Use keys from drug_classes_grouped as the authoritative list of drugs
        drugs_to_validate = list(drug_classes_grouped.keys())

        if not drugs_to_validate:
            # Fallback: prefer flattened_components, then drug_name column
            flattened_components = extraction.get('flattened_components', [])
            if flattened_components:
                drugs_to_validate = flattened_components
            elif drug_name_col:
                drugs_to_validate = [d.strip() for d in drug_name_col.replace(';', ',').split(',') if d.strip()]
            else:
                drugs_to_validate = []

        if not drugs_to_validate:
            print("  No drugs to validate")
            result_row['validation_status'] = 'REVIEW'
            result_row['validation_confidence_grouped'] = '{}'
            result_row['validation_issues_grouped'] = '{}'
            result_row['hallucinated_drug_classes_grouped'] = '{}'
            result_row['rule_compliance_drug_classes_grouped'] = '{}'
            result_row['missed_drug_classes_grouped'] = '{}'
            result_row['validation_checks_grouped'] = '{}'
            result_row['validation_reasoning_grouped'] = '{}'
            result_row['validation_status_grouped'] = '{}'
            result_row['validation_llm_calls'] = 0
            result_row['validation_success_grouped'] = '{}'
            result_row['extraction_performed_grouped'] = '{}'
            result_row['extracted_drug_classes_grouped'] = '{}'
            result_row['extracted_classes'] = '{}'
            result_row['raw_llm_response_grouped'] = '{}'
            result_row['needs_qc'] = True
            return result_row

        print(f"  Validating {len(drugs_to_validate)} drugs: {drugs_to_validate}")

        # Validate each drug and collect results
        validation_status_grouped = {}
        validation_confidence_grouped = {}
        validation_issues_grouped = {}
        validation_checks_grouped = {}
        validation_reasoning_grouped = {}
        extraction_performed_grouped = {}
        extracted_drug_classes_grouped = {}
        missed_drug_classes_grouped = {}
        validation_success_grouped = {}
        raw_llm_response_grouped = {}
        total_llm_calls = 0

        for drug in drugs_to_validate:
            print(f"    - Validating drug: {drug}")
            try:
                drug_validation = validate_single_drug(
                    drug_name=drug,
                    extraction=extraction,
                    cache_data=cache_data,
                    validator=validator,
                    drug_classes_grouped=drug_classes_grouped,
                    selected_sources_grouped=selected_sources_grouped,
                    confidence_scores_grouped=confidence_scores_grouped,
                    reasoning_grouped=reasoning_grouped,
                    extraction_details_grouped=extraction_details_grouped,
                )

                validation_status_grouped[drug] = drug_validation['status']
                validation_confidence_grouped[drug] = drug_validation['confidence']
                validation_issues_grouped[drug] = drug_validation['issues']
                validation_checks_grouped[drug] = drug_validation['checks']
                validation_reasoning_grouped[drug] = drug_validation['reasoning']
                extraction_performed_grouped[drug] = drug_validation['extraction_performed']
                extracted_drug_classes_grouped[drug] = drug_validation['extracted_drug_classes']
                missed_drug_classes_grouped[drug] = drug_validation['missed_drug_classes']
                validation_success_grouped[drug] = drug_validation['validation_success']
                total_llm_calls += drug_validation['llm_calls']
                
                # Store raw LLM response for debugging (only if available)
                if drug_validation.get('raw_llm_response'):
                    raw_llm_response_grouped[drug] = drug_validation['raw_llm_response']

                # Log extraction and missed classes if any
                missed_count = len(drug_validation['missed_drug_classes'])
                if drug_validation['extraction_performed']:
                    extracted_count = len(drug_validation['extracted_drug_classes'])
                    print(f"      Status: {drug_validation['status']} (extraction performed, found {extracted_count} classes, {missed_count} missed)")
                elif missed_count > 0:
                    print(f"      Status: {drug_validation['status']} ({missed_count} missed classes: {drug_validation['missed_drug_classes']})")
                else:
                    print(f"      Status: {drug_validation['status']}")

            except Exception as drug_error:
                print(f"      Error validating {drug}: {drug_error}")
                validation_status_grouped[drug] = 'REVIEW'
                validation_confidence_grouped[drug] = 0.0
                validation_issues_grouped[drug] = [{
                    'check_type': 'system_error',
                    'severity': 'high',
                    'description': f'Validation failed for {drug}: {str(drug_error)}',
                    'evidence': '',
                    'drug_class': '',
                    'transformed_drug_class': None,
                    'rule_reference': ''
                }]
                validation_checks_grouped[drug] = {}
                validation_reasoning_grouped[drug] = f'Validation error: {str(drug_error)}'
                extraction_performed_grouped[drug] = False
                extracted_drug_classes_grouped[drug] = []
                missed_drug_classes_grouped[drug] = []
                validation_success_grouped[drug] = False
                # No raw_llm_response available for exception case

        # Calculate cumulative validation_status
        # FAIL if ANY drug fails, REVIEW if any drug needs review (and none fail), PASS only if all pass
        all_statuses = list(validation_status_grouped.values())
        if 'FAIL' in all_statuses:
            cumulative_status = 'FAIL'
        elif 'REVIEW' in all_statuses:
            cumulative_status = 'REVIEW'
        else:
            cumulative_status = 'PASS'

        # Add validation columns (grouped format) - pretty printed for readability
        result_row['validation_status'] = cumulative_status  # Cumulative
        result_row['validation_status_grouped'] = json.dumps(validation_status_grouped, indent=2)
        result_row['validation_confidence_grouped'] = json.dumps(validation_confidence_grouped, indent=2)
        result_row['validation_issues_grouped'] = json.dumps(validation_issues_grouped, indent=2)
        
        # Extract hallucinated drug classes from validation issues (grouped by drug)
        # Only include drugs that have hallucination issues (non-empty arrays)
        hallucinated_drug_classes_grouped = {}
        for drug, issues in validation_issues_grouped.items():
            hallucinated = []
            for issue in issues:
                if issue.get('check_type') == 'hallucination' and issue.get('drug_class'):
                    hallucinated.append(issue.get('drug_class'))
            if hallucinated:  # Only add if non-empty
                hallucinated_drug_classes_grouped[drug] = hallucinated
        result_row['hallucinated_drug_classes_grouped'] = json.dumps(hallucinated_drug_classes_grouped, indent=2)
        
        # Extract rule compliance issues from validation issues (grouped by drug)
        # Only include drugs that have rule_compliance issues (non-empty arrays)
        rule_compliance_drug_classes_grouped = {}
        for drug, issues in validation_issues_grouped.items():
            rule_compliance_items = []
            for issue in issues:
                if issue.get('check_type') == 'rule_compliance' and issue.get('drug_class'):
                    rule_compliance_items.append({
                        'drug_class': issue.get('drug_class', ''),
                        'transformed_drug_class': issue.get('transformed_drug_class', ''),
                        'rule_reference': issue.get('rule_reference', '')
                    })
            if rule_compliance_items:  # Only add if non-empty
                rule_compliance_drug_classes_grouped[drug] = rule_compliance_items
        result_row['rule_compliance_drug_classes_grouped'] = json.dumps(rule_compliance_drug_classes_grouped, indent=2)
        
        # Add missed drug classes column (filtered to only include non-empty arrays)
        missed_drug_classes_filtered = {
            drug: classes for drug, classes in missed_drug_classes_grouped.items() if classes
        }
        result_row['missed_drug_classes_grouped'] = json.dumps(missed_drug_classes_filtered, indent=2)
        
        result_row['validation_checks_grouped'] = json.dumps(validation_checks_grouped, indent=2)
        # For reasoning, preserve actual newlines for CSV readability
        # Convert escaped \n back to real newlines after JSON serialization
        reasoning_json = json.dumps(validation_reasoning_grouped, indent=2)
        result_row['validation_reasoning_grouped'] = reasoning_json.replace('\\n', '\n')
        result_row['validation_llm_calls'] = total_llm_calls
        
        # Add validation success column (boolean per drug indicating if LLM call succeeded)
        result_row['validation_success_grouped'] = json.dumps(validation_success_grouped, indent=2)
        
        # Add extraction columns (for when validator extracts missing drug classes)
        # Pretty print for object columns (with indent=2)
        result_row['extraction_performed_grouped'] = json.dumps(extraction_performed_grouped, indent=2)
        result_row['extracted_drug_classes_grouped'] = json.dumps(extracted_drug_classes_grouped, indent=2)
        
        # Add simplified extracted classes column (just class names as arrays, no pretty print)
        # Extract only class_name from each extracted drug class for readability
        extracted_classes_simple = {}
        for drug, classes in extracted_drug_classes_grouped.items():
            if classes:
                extracted_classes_simple[drug] = [c.get('class_name', '') for c in classes if c.get('class_name')]
            else:
                extracted_classes_simple[drug] = []
        result_row['extracted_classes'] = json.dumps(extracted_classes_simple)
        
        # Add raw LLM response column for debugging (only includes drugs with responses)
        result_row['raw_llm_response_grouped'] = json.dumps(raw_llm_response_grouped, indent=2) if raw_llm_response_grouped else '{}'

        # Determine if QC is needed
        needs_qc = (
            cumulative_status == "FAIL" or
            cumulative_status == "REVIEW" or
            any(
                any(issue.get('severity') == 'high' for issue in issues)
                for issues in validation_issues_grouped.values()
            )
        )
        result_row['needs_qc'] = needs_qc

        print(f"  Validation complete: {cumulative_status} (needs_qc: {needs_qc})")

    except Exception as e:
        print(f"  Validation error for {drug_name_col}: {e}")
        import traceback
        traceback.print_exc()
        result_row['validation_status'] = 'REVIEW'
        result_row['validation_status_grouped'] = '{}'
        result_row['validation_confidence_grouped'] = '{}'
        result_row['validation_issues_grouped'] = json.dumps({
            '_error': [{
                'check_type': 'system_error',
                'severity': 'high',
                'description': f'Validation failed: {str(e)}',
                'evidence': '',
                'drug_class': '',
                'transformed_drug_class': None,
                'rule_reference': ''
            }]
        }, indent=2)
        result_row['hallucinated_drug_classes_grouped'] = '{}'
        result_row['rule_compliance_drug_classes_grouped'] = '{}'
        result_row['missed_drug_classes_grouped'] = '{}'
        result_row['validation_checks_grouped'] = '{}'
        result_row['validation_reasoning_grouped'] = '{}'
        result_row['validation_llm_calls'] = 0
        result_row['validation_success_grouped'] = '{}'
        result_row['extraction_performed_grouped'] = '{}'
        result_row['extracted_drug_classes_grouped'] = '{}'
        result_row['extracted_classes'] = '{}'
        result_row['raw_llm_response_grouped'] = '{}'
        result_row['needs_qc'] = True

    return result_row


def validate_extractions_batch(
    extractions: List[Dict],
    cache_data: Dict,
    validator: DrugClassValidationAgent,
    output_file: str = None,
    num_workers: int = 3,
) -> pd.DataFrame:
    """Validate a batch of extractions and return results DataFrame.

    Args:
        extractions: List of extraction dictionaries
        cache_data: Cache dictionary with search results
        validator: Initialized DrugClassValidationAgent
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
            executor.submit(
                validate_single_extraction,
                extraction,
                cache_data,
                validator,
                i
            ): i
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
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Validate Drug Class Extractions')
    parser.add_argument('--input_file', default='data/drug_class_validation_input.csv',
                        help='Input CSV file with extraction results')
    parser.add_argument('--cache_file', default='data/drug_search_cache.json',
                        help='Input JSON cache file with search results')
    parser.add_argument('--output_file', default=None,
                        help='Output CSV file (default: auto-generated)')
    parser.add_argument('--llm_model', default="claude-haiku-4-5",
                        help='LLM model name to use for validation calls')
    parser.add_argument('--temperature', type=float, default=0,
                        help='LLM temperature (default: from settings)')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='LLM max tokens (default: from settings)')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Maximum number of rows to validate (default: all)')
    parser.add_argument('--skip_rows', type=int, default=0,
                        help='Number of rows to skip from the beginning')
    parser.add_argument('--num_workers', type=int, default=3,
                        help='Number of parallel workers (default: 3)')
    parser.add_argument('--enable_caching', action='store_true',
                        help='Enable prompt caching for Anthropic models (reduces costs)')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = f"data/{base_name}_validated_{timestamp}.csv"

    print("🔍 Drug Class Validation Processor")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Cache file: {args.cache_file}")
    print(f"Output file: {args.output_file}")
    print(f"LLM model: {args.llm_model}")
    print(f"Max rows: {args.max_rows or 'all'}")
    print(f"Skip rows: {args.skip_rows}")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Prompt caching: {'enabled' if args.enable_caching else 'disabled'}")
    print()

    # Load extraction results
    print("Loading extraction results...")
    extractions = load_extractions_from_csv(
        args.input_file,
        max_rows=args.max_rows,
        skip_rows=args.skip_rows,
    )

    if not extractions:
        print("No extraction results loaded. Exiting.")
        return

    print(f"Loaded {len(extractions)} extraction results")
    print()

    # Load cache
    print("Loading search results cache...")
    cache_data = load_cache(args.cache_file)
    print()

    # Initialize validation agent
    print("Initializing Drug Class Validation Agent...")
    validator = DrugClassValidationAgent(
        agent_name="DrugClassValidationProcessor",
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_caching=args.enable_caching,
    )
    print("✓ Validation Agent initialized successfully!")
    print()

    # Validate extractions
    print("-" * 80)
    results_df = validate_extractions_batch(
        extractions,
        cache_data,
        validator,
        args.output_file,
        args.num_workers,
    )

    # Summary
    print()
    print("=" * 80)
    total_validated = len(results_df)

    if 'validation_status' in results_df.columns:
        pass_count = (results_df['validation_status'] == 'PASS').sum()
        review_count = (results_df['validation_status'] == 'REVIEW').sum()
        fail_count = (results_df['validation_status'] == 'FAIL').sum()

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
    
    # Calculate and display execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print()
    print(f"⏱️  Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")


if __name__ == "__main__":
    main()

