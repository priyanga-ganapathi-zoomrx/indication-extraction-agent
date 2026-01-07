#!/usr/bin/env python3
"""
Drug Class Selection Processor

This script post-processes drug class extraction results by selecting
the most appropriate drug class(es) from extracted candidates using
priority and specificity rules.

Input: CSV file with extraction results (from drug_class_react_batch_processor.py)
Output: CSV file with selected drug classes added

Features:
- Reads extraction results from CSV
- Transforms extraction_details_grouped to selection input format
- Applies priority rules (MoA > Chemical > Mode > Therapeutic)
- Applies specificity rules (child over parent)
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

from src.drug_class_selection_agent import DrugClassSelectionAgent


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


def transform_extraction_to_selection_input(
    extraction_details: List[Dict],
) -> List[Dict[str, str]]:
    """Transform extraction_details to selection input format.

    Args:
        extraction_details: List of extraction detail dicts from extraction output
            Each dict has: extracted_text, class_type, normalized_form, evidence, source, rules_applied

    Returns:
        List of dicts with 'drug_class' and 'class_type' for selection input
    """
    selection_input = []
    seen_classes = set()  # Deduplicate

    for detail in extraction_details:
        drug_class = detail.get('normalized_form', detail.get('extracted_text', ''))
        class_type = detail.get('class_type', 'Therapeutic')  # Default to Therapeutic if missing

        if drug_class and drug_class not in seen_classes:
            seen_classes.add(drug_class)
            selection_input.append({
                'drug_class': drug_class,
                'class_type': class_type,
            })

    return selection_input


def group_classes_by_type(
    extraction_details_grouped: Dict[str, List[Dict]],
) -> Dict[str, Dict[str, List[str]]]:
    """Group drug classes by class_type for each drug.

    Args:
        extraction_details_grouped: Dict mapping drug names to list of extraction details

    Returns:
        Dict mapping drug names to dict of class_type -> list of drug classes
        Example: {"Folinic acid": {"Therapeutic": ["Antidote"], "Chemical": ["Folate Analog"]}}
    """
    result = {}

    for drug_name, details in extraction_details_grouped.items():
        classes_by_type = {}
        seen_classes = set()  # Deduplicate within each drug

        for detail in details:
            drug_class = detail.get('normalized_form', detail.get('extracted_text', ''))
            class_type = detail.get('class_type', 'Therapeutic')

            if drug_class and drug_class not in seen_classes:
                seen_classes.add(drug_class)
                if class_type not in classes_by_type:
                    classes_by_type[class_type] = []
                classes_by_type[class_type].append(drug_class)

        result[drug_name] = classes_by_type

    return result


def select_drug_class_for_single_drug(
    drug_name: str,
    extraction_details: List[Dict],
    selector: DrugClassSelectionAgent,
    abstract_id: str = None,
) -> Dict:
    """Select drug class for a single drug.

    Args:
        drug_name: The drug name
        extraction_details: List of extraction detail dicts
        selector: Initialized DrugClassSelectionAgent
        abstract_id: Abstract ID for tracking

    Returns:
        Dictionary with selection results
    """
    # Transform extraction details to selection input format
    extracted_classes = transform_extraction_to_selection_input(extraction_details)

    # Run selection
    selection_result = selector.invoke(
        drug_name=drug_name,
        extracted_classes=extracted_classes,
        abstract_id=abstract_id,
    )

    return {
        'selected_drug_classes': selection_result.get('selected_drug_classes', ['NA']),
        'reasoning': selection_result.get('reasoning', ''),
        'llm_calls': selection_result.get('llm_calls', 0),
        'selection_success': selection_result.get('selection_success', True),
    }


def process_single_extraction(
    extraction: Dict,
    selector: DrugClassSelectionAgent,
    index: int,
) -> Dict:
    """Process a single extraction result (may contain multiple drugs).

    For rows with multiple drugs in extraction_details_grouped, processes each drug
    separately and produces grouped selection output columns.

    Args:
        extraction: Extraction data dictionary
        selector: Initialized DrugClassSelectionAgent
        index: Index for logging

    Returns:
        Dictionary with original data plus selection results
    """
    abstract_id = extraction['abstract_id']
    drug_name_col = extraction['drug_name']  # May be comma-separated or regimen name
    print(f"[{index}] Processing selection for: {drug_name_col} (ID: {abstract_id})")

    result_row = extraction['original_row'].copy()

    try:
        # Get extraction result
        extraction_result = extraction['extraction_result']

        # Get grouped extraction data
        extraction_details_grouped = extraction_result.get('extraction_details_grouped', {})
        drug_classes_grouped = extraction_result.get('drug_classes_grouped', {})

        # Determine which drugs to process
        # Use keys from extraction_details_grouped as the authoritative list of drugs
        drugs_to_process = list(extraction_details_grouped.keys())

        if not drugs_to_process:
            # Fallback: use drug_classes_grouped keys
            drugs_to_process = list(drug_classes_grouped.keys())

        if not drugs_to_process:
            # Fallback: prefer flattened_components, then drug_name column
            flattened_components = extraction.get('flattened_components', [])
            if flattened_components:
                drugs_to_process = flattened_components
            elif drug_name_col:
                drugs_to_process = [d.strip() for d in drug_name_col.replace(';', ',').split(',') if d.strip()]
            else:
                drugs_to_process = []

        if not drugs_to_process:
            print("  No drugs to process")
            result_row['drug_classes_by_type_grouped'] = '{}'
            result_row['selected_drug_classes_grouped'] = '{}'
            result_row['selection_reasoning_grouped'] = '{}'
            result_row['selection_llm_calls'] = 0
            result_row['selection_success_grouped'] = '{}'
            result_row['selected_drug_classes'] = '["NA"]'
            return result_row

        print(f"  Processing {len(drugs_to_process)} drugs: {drugs_to_process}")

        # Process each drug and collect results
        selected_drug_classes_grouped = {}
        selection_reasoning_grouped = {}
        selection_success_grouped = {}
        total_llm_calls = 0

        for drug in drugs_to_process:
            print(f"    - Selecting for drug: {drug}")
            try:
                # Get extraction details for this specific drug
                extraction_details = extraction_details_grouped.get(drug, [])

                # If no extraction details, try to construct from drug_classes_grouped
                if not extraction_details:
                    drug_classes = drug_classes_grouped.get(drug, [])
                    # Create minimal extraction details with default class_type
                    extraction_details = [
                        {'normalized_form': dc, 'class_type': 'Therapeutic'}
                        for dc in drug_classes if dc and dc != 'NA'
                    ]

                if not extraction_details:
                    print(f"      No extraction details for {drug}, returning NA")
                    selected_drug_classes_grouped[drug] = ['NA']
                    selection_reasoning_grouped[drug] = 'No extraction details available for selection.'
                    selection_success_grouped[drug] = True
                    continue

                # Get unique drug classes from extraction details
                unique_classes = list(set(
                    detail.get('normalized_form', detail.get('extracted_text', ''))
                    for detail in extraction_details
                    if detail.get('normalized_form') or detail.get('extracted_text')
                ))

                # If only one unique drug class, skip selection - no LLM call needed
                if len(unique_classes) == 1:
                    print(f"      Only one class extracted, skipping selection: {unique_classes[0]}")
                    selected_drug_classes_grouped[drug] = [unique_classes[0]]
                    selection_reasoning_grouped[drug] = 'Only one class was extracted. No selection logic needed.'
                    selection_success_grouped[drug] = True
                    continue

                drug_selection = select_drug_class_for_single_drug(
                    drug_name=drug,
                    extraction_details=extraction_details,
                    selector=selector,
                    abstract_id=str(abstract_id),
                )

                selected_drug_classes_grouped[drug] = drug_selection['selected_drug_classes']
                selection_reasoning_grouped[drug] = drug_selection['reasoning']
                selection_success_grouped[drug] = drug_selection['selection_success']
                total_llm_calls += drug_selection['llm_calls']

                print(f"      Selected: {drug_selection['selected_drug_classes']}")

            except Exception as drug_error:
                print(f"      Error selecting for {drug}: {drug_error}")
                selected_drug_classes_grouped[drug] = ['NA']
                selection_reasoning_grouped[drug] = f'Selection failed for {drug}: {str(drug_error)}'
                selection_success_grouped[drug] = False

        # Add drug classes grouped by type column
        drug_classes_by_type = group_classes_by_type(extraction_details_grouped)
        result_row['drug_classes_by_type_grouped'] = json.dumps(drug_classes_by_type, indent=2)

        # Add selection columns (grouped format) - pretty printed for readability
        result_row['selected_drug_classes_grouped'] = json.dumps(selected_drug_classes_grouped, indent=2)
        result_row['selection_reasoning_grouped'] = json.dumps(selection_reasoning_grouped, indent=2)
        result_row['selection_llm_calls'] = total_llm_calls
        result_row['selection_success_grouped'] = json.dumps(selection_success_grouped, indent=2)

        # Flatten selected drug classes for convenience (combine all drugs)
        all_selected = []
        for drug, classes in selected_drug_classes_grouped.items():
            for cls in classes:
                if cls and cls != 'NA' and cls not in all_selected:
                    all_selected.append(cls)
        if not all_selected:
            all_selected = ['NA']
        result_row['selected_drug_classes'] = json.dumps(all_selected)

        print(f"  Selection complete: {total_llm_calls} LLM calls")

    except Exception as e:
        print(f"  Selection error for {drug_name_col}: {e}")
        import traceback
        traceback.print_exc()
        result_row['drug_classes_by_type_grouped'] = '{}'
        result_row['selected_drug_classes_grouped'] = '{}'
        result_row['selection_reasoning_grouped'] = json.dumps({
            '_error': f'Selection failed: {str(e)}'
        }, indent=2)
        result_row['selection_llm_calls'] = 0
        result_row['selection_success_grouped'] = '{}'
        result_row['selected_drug_classes'] = '["NA"]'

    return result_row


def process_extractions_batch(
    extractions: List[Dict],
    selector: DrugClassSelectionAgent,
    output_file: str = None,
    num_workers: int = 3,
) -> pd.DataFrame:
    """Process a batch of extractions and return results DataFrame.

    Args:
        extractions: List of extraction dictionaries
        selector: Initialized DrugClassSelectionAgent
        output_file: Optional output file path to save intermediate results
        num_workers: Number of parallel workers (default: 3)

    Returns:
        DataFrame with selection results
    """
    print(f"Processing {len(extractions)} extractions (using {num_workers} parallel threads)")

    results = []

    # Process extractions in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(
                process_single_extraction,
                extraction,
                selector,
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
    """Main selection processing function."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Drug Class Selection Post-Processor')
    parser.add_argument('--input_file', default='data/drug_class_validation_input_gemini-3-pro_150.csv',
                        help='Input CSV file with extraction results')
    parser.add_argument('--output_file', default=None,
                        help='Output CSV file (default: auto-generated)')
    parser.add_argument('--llm_model', default="gemini/gemini-3-flash-preview",
                        help='LLM model name to use for selection calls')
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
    parser.add_argument('--enable_caching', action='store_true',
                        help='Enable prompt caching for Anthropic models (reduces costs)')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = f"data/{base_name}_selected_{timestamp}.csv"

    print("🎯 Drug Class Selection Processor")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
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

    # Initialize selection agent
    print("Initializing Drug Class Selection Agent...")
    selector = DrugClassSelectionAgent(
        agent_name="DrugClassSelectionProcessor",
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_caching=args.enable_caching,
    )
    print("✓ Selection Agent initialized successfully!")
    print()

    # Process extractions
    print("-" * 80)
    results_df = process_extractions_batch(
        extractions,
        selector,
        args.output_file,
        args.num_workers,
    )

    # Summary
    print()
    print("=" * 80)
    total_processed = len(results_df)

    if 'selection_success_grouped' in results_df.columns:
        # Count successful selections
        success_count = 0
        for _, row in results_df.iterrows():
            try:
                success_grouped = json.loads(row.get('selection_success_grouped', '{}'))
                if all(success_grouped.values()):
                    success_count += 1
            except (json.JSONDecodeError, TypeError):
                pass

        print("📊 Selection Summary:")
        print(f"Total rows processed: {total_processed}")
        print(f"  Successful: {success_count} ({success_count/total_processed*100:.1f}%)")
        print(f"  With issues: {total_processed - success_count} ({(total_processed - success_count)/total_processed*100:.1f}%)")

    if 'selection_llm_calls' in results_df.columns:
        total_llm_calls = results_df['selection_llm_calls'].sum()
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

