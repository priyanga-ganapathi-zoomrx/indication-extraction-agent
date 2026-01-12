#!/usr/bin/env python3
"""
Drug Class Consolidation Only Processor

This script processes abstracts to consolidate and deduplicate drug classes.
It expects PRE-SELECTED drug classes as input and filters extraction_details
to only include the selected classes (preserving evidence and rules_applied).

Input:
- CSV with pre-selected drug classes (selected_drug_classes_grouped)
- CSV with explicit drug classes from title extraction

Output:
- Consolidated drug-class mappings
- Refined explicit drug classes (after deduplication)

Features:
- Filters extraction_details_grouped by selected_drug_classes_grouped
- Preserves evidence, source, rules_applied for consolidation decisions
- Supports parallel processing with ThreadPoolExecutor
- Saves intermediate results incrementally
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drug_class_consolidation_agent import DrugClassConsolidationOnlyAgent


def parse_json_safely(json_str: str, default: Any = None) -> Any:
    """Safely parse a JSON string.

    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    if pd.isna(json_str) or not json_str:
        return default
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def filter_extraction_details_by_selected(
    extraction_details: List[Dict],
    selected_classes: List[str]
) -> List[Dict]:
    """Filter extraction details to only include selected drug classes.

    Args:
        extraction_details: List of extraction detail dicts with normalized_form/drug_class
        selected_classes: List of selected drug class names

    Returns:
        Filtered list of extraction details matching selected classes
    """
    if not extraction_details or not selected_classes:
        return []

    # Normalize selected classes for comparison (case-insensitive, ignore hyphens)
    def normalize_class_name(name: str) -> str:
        if not name:
            return ""
        return name.lower().replace("-", "").replace(" ", "")

    selected_normalized = {normalize_class_name(c) for c in selected_classes if c and c != "NA"}

    filtered = []
    for detail in extraction_details:
        if not isinstance(detail, dict):
            continue

        # Check normalized_form or drug_class
        class_name = detail.get("normalized_form") or detail.get("drug_class", "")
        if normalize_class_name(class_name) in selected_normalized:
            filtered.append(detail)

    return filtered


def load_drug_selections(csv_path: str) -> Dict[str, Dict]:
    """Load drug selections from CSV with filtered extraction details.

    Reads:
    - selected_drug_classes_grouped: Already selected classes per drug
    - extraction_details_grouped: Full extraction details (to be filtered)
    - selection_reasoning_grouped: Reasoning for selection

    Filters extraction_details to only include classes in selected_drug_classes.

    Args:
        csv_path: Path to drug selections CSV

    Returns:
        Dict mapping abstract_id -> {abstract_title, drug_selections}
    """
    if not os.path.exists(csv_path):
        print(f"Error: Drug selections file not found at {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"✓ Loaded {len(df)} rows from drug selections file")

        abstracts = {}

        for _, row in df.iterrows():
            abstract_id = str(row.get('abstract_id', ''))
            if not abstract_id:
                continue

            abstract_title = str(row.get('abstract_title', ''))

            # Parse selected_drug_classes_grouped: {"drug_name": ["Class1", "Class2"]}
            selected_classes_grouped = parse_json_safely(
                row.get('selected_drug_classes_grouped', '{}'), {}
            )

            # Parse extraction_details_grouped: {"drug_name": [{...}, {...}]}
            extraction_details_grouped = parse_json_safely(
                row.get('extraction_details_grouped', '{}'), {}
            )

            # Parse selection_reasoning_grouped: {"drug_name": "reasoning"}
            selection_reasoning_grouped = parse_json_safely(
                row.get('selection_reasoning_grouped', '{}'), {}
            )

            # Transform to drug_selections format with filtered extraction details
            drug_selections = []
            for drug_name, selected_classes in selected_classes_grouped.items():
                # Get full extraction details for this drug
                full_details = extraction_details_grouped.get(drug_name, [])

                # Filter to only include selected classes
                filtered_details = filter_extraction_details_by_selected(
                    full_details, selected_classes
                )

                # Get selection reasoning
                selection_reasoning = selection_reasoning_grouped.get(drug_name, "")

                drug_selections.append({
                    "drug_name": drug_name,
                    "selected_drug_classes": selected_classes if isinstance(selected_classes, list) else [selected_classes],
                    "selection_reasoning": selection_reasoning,
                    "extraction_details": filtered_details,  # Filtered details with evidence
                })

            # Store by abstract_id
            if abstract_id not in abstracts:
                abstracts[abstract_id] = {
                    "abstract_title": abstract_title,
                    "drug_selections": drug_selections,
                    "original_row": row.to_dict(),
                }
            else:
                # Merge drug selections if same abstract has multiple rows
                abstracts[abstract_id]["drug_selections"].extend(drug_selections)

        print(f"✓ Parsed {len(abstracts)} unique abstracts with drug selections")
        return abstracts

    except Exception as e:
        print(f"Error loading drug selections: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_explicit_drug_classes(csv_path: str) -> Dict[str, Dict]:
    """Load explicit drug classes from CSV, grouped by abstract_id.

    Expected columns:
    - abstract_id, abstract_title
    - drug_classes: JSON array ["Class1", "Class2"]
    - raw_json_response: Full extraction JSON with details

    Args:
        csv_path: Path to explicit drug classes CSV

    Returns:
        Dict mapping abstract_id -> {drug_classes, extraction_details}
    """
    if not os.path.exists(csv_path):
        print(f"Error: Explicit drug classes file not found at {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"✓ Loaded {len(df)} rows from explicit drug classes file")

        # Group by abstract_id to get unique abstracts
        abstracts = {}

        for _, row in df.iterrows():
            abstract_id = str(row.get('abstract_id', ''))
            if not abstract_id:
                continue

            # Skip if already processed this abstract_id
            if abstract_id in abstracts:
                continue

            # Parse drug_classes: ["Class1", "Class2"]
            drug_classes = parse_json_safely(
                row.get('drug_classes', '["NA"]'), ["NA"]
            )

            # Parse raw_json_response for extraction details and reasoning
            raw_json = parse_json_safely(
                row.get('raw_json_response', '{}'), {}
            )

            extraction_details = raw_json.get('extraction_details', [])
            reasoning = raw_json.get('reasoning', '')

            abstracts[abstract_id] = {
                "drug_classes": drug_classes if isinstance(drug_classes, list) else [drug_classes],
                "reasoning": reasoning,
                "extraction_details": extraction_details,
                "abstract_title": str(row.get('abstract_title', '')),
            }

        print(f"✓ Parsed {len(abstracts)} unique abstracts with explicit drug classes")
        return abstracts

    except Exception as e:
        print(f"Error loading explicit drug classes: {e}")
        import traceback
        traceback.print_exc()
        return {}


def merge_by_abstract_id(
    drug_selections: Dict[str, Dict],
    explicit_classes: Dict[str, Dict]
) -> List[Dict]:
    """Merge drug selections and explicit classes by abstract_id.

    Args:
        drug_selections: Dict from load_drug_selections
        explicit_classes: Dict from load_explicit_drug_classes

    Returns:
        List of merged abstract records ready for consolidation
    """
    merged = []

    # Get all unique abstract_ids
    all_ids = set(drug_selections.keys()) | set(explicit_classes.keys())

    for abstract_id in all_ids:
        selection_data = drug_selections.get(abstract_id, {})
        explicit_data = explicit_classes.get(abstract_id, {})

        # Determine abstract_title (prefer from drug_selections)
        abstract_title = selection_data.get("abstract_title") or explicit_data.get("abstract_title", "")

        # Build explicit_drug_classes input
        explicit_drug_classes = {
            "drug_classes": explicit_data.get("drug_classes", ["NA"]),
            "reasoning": explicit_data.get("reasoning", ""),
            "extraction_details": explicit_data.get("extraction_details", [])
        }

        # Build drug_selections input (already has filtered extraction details)
        drug_selections_list = selection_data.get("drug_selections", [])

        # Handle missing data
        if not drug_selections_list:
            # No drug selections - might be procedure-only abstract
            drug_selections_list = []

        merged.append({
            "abstract_id": abstract_id,
            "abstract_title": abstract_title,
            "explicit_drug_classes": explicit_drug_classes,
            "drug_selections": drug_selections_list,
            "original_row": selection_data.get("original_row", {}),
        })

    # Sort by abstract_id for consistent ordering
    merged.sort(key=lambda x: x["abstract_id"])

    print(f"✓ Merged {len(merged)} abstracts for consolidation")
    return merged


def consolidate_single_abstract(
    abstract: Dict,
    agent: DrugClassConsolidationOnlyAgent,
    index: int,
) -> Dict:
    """Consolidate drug classes for a single abstract.

    Args:
        abstract: Merged abstract data with pre-selected classes
        agent: Initialized DrugClassConsolidationOnlyAgent
        index: Index for logging

    Returns:
        Dictionary with original data plus consolidation results
    """
    abstract_id = abstract["abstract_id"]
    abstract_title = abstract["abstract_title"]

    print(f"[{index}] Consolidating: {abstract_title[:80]}..." if len(abstract_title) > 80 else f"[{index}] Consolidating: {abstract_title}")

    result_row = abstract.get("original_row", {}).copy()
    result_row["abstract_id"] = abstract_id
    result_row["abstract_title"] = abstract_title

    # Store original explicit drug classes (input) for comparison
    input_explicit_classes = abstract["explicit_drug_classes"].get("drug_classes", ["NA"])
    result_row["input_explicit_drug_classes"] = json.dumps(input_explicit_classes)

    try:
        # Run consolidation
        consolidation_result = agent.invoke(
            abstract_title=abstract_title,
            explicit_drug_classes=abstract["explicit_drug_classes"],
            drug_selections=abstract["drug_selections"],
            abstract_id=abstract_id,
        )

        # Add consolidation result columns (simplified output)
        refined_explicit = consolidation_result.get("refined_explicit_drug_classes", {})
        result_row["refined_explicit_classes"] = json.dumps(refined_explicit, indent=2)
        result_row["consolidation_reasoning"] = consolidation_result.get("reasoning", "")
        result_row["consolidation_raw_response"] = consolidation_result.get("raw_json_response", "")
        result_row["consolidation_success"] = consolidation_result.get("consolidation_success", False)
        result_row["consolidation_llm_calls"] = consolidation_result.get("llm_calls", 0)
        result_row["duplicates_removed"] = consolidation_result.get("duplicates_removed", 0)

        # Build combined drug classes column
        # Collect all drug classes from input drug_selections (already selected upstream)
        all_drug_classes = []
        for drug_sel in abstract["drug_selections"]:
            selected_classes = drug_sel.get("selected_drug_classes", [])
            for cls in selected_classes:
                if cls and cls != "NA" and cls not in all_drug_classes:
                    all_drug_classes.append(cls)

        # Add refined explicit drug classes (after consolidation)
        explicit_classes = refined_explicit.get("drug_classes", [])
        for cls in explicit_classes:
            if cls and cls != "NA" and cls not in all_drug_classes:
                all_drug_classes.append(cls)

        # Store combined column (as JSON array for consistency, or "NA" if empty)
        if all_drug_classes:
            result_row["combined_drug_classes"] = json.dumps(all_drug_classes)
        else:
            result_row["combined_drug_classes"] = json.dumps(["NA"])

        # Add explicit drug classes only column
        explicit_only = [c for c in explicit_classes if c and c != "NA"]
        if explicit_only:
            result_row["explicit_drug_classes_only"] = json.dumps(explicit_only)
        else:
            result_row["explicit_drug_classes_only"] = json.dumps(["NA"])

        # Log result summary
        status = "✓" if consolidation_result.get("consolidation_success", False) else "✗"
        drugs_count = len(abstract["drug_selections"])
        explicit_count = len([c for c in explicit_classes if c != "NA"])
        removed_count = len(refined_explicit.get("removed_classes", []))
        print(f"  {status} Drugs: {drugs_count}, Explicit: {explicit_count}, Removed: {removed_count}, Combined: {len(all_drug_classes)}")

    except Exception as e:
        print(f"  ✗ Consolidation error for {abstract_id}: {e}")
        import traceback
        traceback.print_exc()

        # Create default error response (simplified)
        # Note: input_explicit_drug_classes already set above
        result_row["refined_explicit_classes"] = json.dumps({"drug_classes": ["NA"], "removed_classes": []}, indent=2)
        result_row["consolidation_reasoning"] = f"Consolidation failed: {str(e)}"
        result_row["consolidation_raw_response"] = ""
        result_row["consolidation_success"] = False
        result_row["consolidation_llm_calls"] = 0
        result_row["duplicates_removed"] = 0
        result_row["combined_drug_classes"] = json.dumps(["NA"])
        result_row["explicit_drug_classes_only"] = json.dumps(["NA"])

    return result_row


def consolidate_batch(
    abstracts: List[Dict],
    agent: DrugClassConsolidationOnlyAgent,
    output_file: str = None,
    num_workers: int = 3,
    save_interval: int = 5,
) -> pd.DataFrame:
    """Consolidate drug classes for a batch of abstracts.

    Args:
        abstracts: List of merged abstract dictionaries
        agent: Initialized DrugClassConsolidationOnlyAgent
        output_file: Optional output file path to save intermediate results
        num_workers: Number of parallel workers (default: 3)
        save_interval: Save intermediate results every N records (default: 5)

    Returns:
        DataFrame with consolidation results
    """
    print(f"Consolidating drug classes for {len(abstracts)} abstracts (using {num_workers} parallel threads)")

    results = []

    # Process abstracts in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(
                consolidate_single_abstract,
                abstract,
                agent,
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
    """Main consolidation processing function."""
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Consolidate Drug Class Selections (Consolidation Only)')
    parser.add_argument('--drug_selections_file',
                        default='data/drug_class_validation_input_gemini-3-pro_150_selected.csv',
                        help='Input CSV file with pre-selected drug classes')
    parser.add_argument('--explicit_classes_file',
                        default='data/drug_class_input_regimen_150_explicit_drug_class.csv',
                        help='Input CSV file with explicit drug classes from title')
    parser.add_argument('--output_file', default=None,
                        help='Output CSV file (default: auto-generated)')
    parser.add_argument('--llm_model', default="gemini/gemini-3-flash-preview",
                        help='LLM model name to use for consolidation')
    parser.add_argument('--temperature', type=float, default=0,
                        help='LLM temperature (default: 0)')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='LLM max tokens (default: from settings)')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Maximum number of abstracts to process (default: all)')
    parser.add_argument('--skip_rows', type=int, default=0,
                        help='Number of abstracts to skip from the beginning')
    parser.add_argument('--num_workers', type=int, default=3,
                        help='Number of parallel workers (default: 3)')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save intermediate results every N records (default: 5)')
    parser.add_argument('--enable_caching', action='store_true',
                        help='Enable prompt caching for Anthropic models')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = ""
        if args.llm_model:
            model_suffix = f"_{args.llm_model.split('/')[-1]}"
        args.output_file = f"data/drug_class_consolidated_only{model_suffix}_{timestamp}.csv"

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print("🔬 Drug Class Consolidation Only Processor")
    print("=" * 80)
    print(f"Drug selections file: {args.drug_selections_file}")
    print(f"Explicit classes file: {args.explicit_classes_file}")
    print(f"Output file: {args.output_file}")
    print(f"LLM model: {args.llm_model or 'from settings'}")
    print(f"Temperature: {args.temperature}")
    print(f"Max abstracts: {args.max_rows or 'all'}")
    print(f"Skip abstracts: {args.skip_rows}")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Save interval: {args.save_interval}")
    print(f"Prompt caching: {'enabled' if args.enable_caching else 'disabled'}")
    print()

    # Load drug selections (with filtered extraction details)
    print("Loading drug selections...")
    drug_selections = load_drug_selections(args.drug_selections_file)
    if not drug_selections:
        print("No drug selections loaded. Exiting.")
        return

    # Load explicit drug classes
    print("\nLoading explicit drug classes...")
    explicit_classes = load_explicit_drug_classes(args.explicit_classes_file)
    if not explicit_classes:
        print("No explicit drug classes loaded. Exiting.")
        return

    # Merge by abstract_id
    print("\nMerging datasets...")
    merged_abstracts = merge_by_abstract_id(drug_selections, explicit_classes)

    if not merged_abstracts:
        print("No abstracts to process after merge. Exiting.")
        return

    # Apply skip and limit
    if args.skip_rows > 0:
        merged_abstracts = merged_abstracts[args.skip_rows:]
    if args.max_rows is not None:
        merged_abstracts = merged_abstracts[:args.max_rows]

    print(f"\nProcessing {len(merged_abstracts)} abstracts")
    print()

    # Initialize consolidation agent
    print("Initializing Drug Class Consolidation Only Agent...")
    agent = DrugClassConsolidationOnlyAgent(
        agent_name="DrugClassConsolidationOnlyProcessor",
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_caching=args.enable_caching,
    )
    print("✓ Consolidation Only Agent initialized successfully!")
    print()

    # Consolidate drug classes
    print("-" * 80)
    results_df = consolidate_batch(
        merged_abstracts,
        agent,
        args.output_file,
        args.num_workers,
        args.save_interval,
    )

    # Summary
    print()
    print("=" * 80)
    total_processed = len(results_df)

    if 'consolidation_success' in results_df.columns:
        success_count = results_df['consolidation_success'].sum()
        fail_count = total_processed - success_count

        print("📊 Consolidation Summary:")
        print(f"Total abstracts processed: {total_processed}")
        print(f"  Success: {int(success_count)} ({success_count/total_processed*100:.1f}%)")
        print(f"  Failed: {int(fail_count)} ({fail_count/total_processed*100:.1f}%)")

    # Count duplicates removed
    if 'duplicates_removed' in results_df.columns:
        total_duplicates = results_df['duplicates_removed'].sum()
        print(f"  Total duplicates removed: {int(total_duplicates)}")

    # Total LLM calls
    if 'consolidation_llm_calls' in results_df.columns:
        total_llm_calls = results_df['consolidation_llm_calls'].sum()
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

