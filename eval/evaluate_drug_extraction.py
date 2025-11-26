"""
Drug Extraction Evaluation Script

Compares LLM drug extraction results (result/output.csv) with golden data (data/golden_data_drug.csv).
Performs exact match analysis (case-insensitive) for Primary, Secondary, and Comparator drugs.

Outputs:
- Detailed evaluation CSV with matched/missed/hallucinated/misclassified columns
- Summary metrics (Precision, Recall, F1) per drug category
"""

import pandas as pd
import ast
from typing import Set, Tuple, Dict
import os
import argparse


def parse_drug_list_llm(drug_str: str) -> Set[str]:
    """
    Parse drug list from LLM output (JSON array format).
    Returns a set of lowercase drug names.
    """
    if pd.isna(drug_str) or drug_str == '' or drug_str == '[]':
        return set()
    
    try:
        # Try parsing as JSON array
        drugs = ast.literal_eval(drug_str)
        if isinstance(drugs, list):
            return {drug.strip().lower() for drug in drugs if drug.strip()}
    except (ValueError, SyntaxError):
        pass
    
    # Fallback: try as comma-separated string
    return {drug.strip().lower() for drug in drug_str.split(',') if drug.strip()}


def parse_drug_list_golden(drug_str: str) -> Set[str]:
    """
    Parse drug list from golden data (comma-separated format).
    Returns a set of lowercase drug names.
    """
    if pd.isna(drug_str) or drug_str == '':
        return set()
    
    return {drug.strip().lower() for drug in str(drug_str).split(',') if drug.strip()}


def compare_drug_sets(
    llm_drugs: Set[str], 
    golden_drugs: Set[str],
    all_golden_drugs: Set[str],
    all_llm_drugs: Set[str]
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    Compare LLM drugs with golden drugs for a specific category.
    
    Args:
        llm_drugs: Drugs extracted by LLM for this category
        golden_drugs: Golden truth drugs for this category
        all_golden_drugs: All golden drugs across all categories (for misclassification detection)
        all_llm_drugs: All LLM drugs across all categories
    
    Returns:
        Tuple of (matched, missed, hallucinated, misclassified)
    """
    matched = llm_drugs & golden_drugs
    missed = golden_drugs - llm_drugs
    
    # Hallucinated: in LLM but not in golden (for this category)
    not_in_golden_category = llm_drugs - golden_drugs
    
    # Misclassified: drug is in LLM for this category but exists in golden in a DIFFERENT category
    misclassified = not_in_golden_category & (all_golden_drugs - golden_drugs)
    
    # Pure hallucination: drug is in LLM but doesn't exist anywhere in golden
    hallucinated = not_in_golden_category - all_golden_drugs
    
    return matched, missed, hallucinated, misclassified


def set_to_str(drug_set: Set[str]) -> str:
    """Convert a set of drugs to a sorted, comma-separated string."""
    if not drug_set:
        return ""
    return ", ".join(sorted(drug_set))


def evaluate_drug_extraction(
    llm_output_path: str,
    golden_data_path: str,
    output_path: str
) -> Dict:
    """
    Main evaluation function.
    
    Args:
        llm_output_path: Path to LLM output CSV
        golden_data_path: Path to golden data CSV
        output_path: Path for evaluation output CSV
    
    Returns:
        Dictionary with summary metrics
    """
    # Load data
    print("Loading data...")
    llm_df = pd.read_csv(llm_output_path)
    golden_df = pd.read_csv(golden_data_path)
    
    print(f"LLM output: {len(llm_df)} records")
    print(f"Golden data: {len(golden_df)} records")
    
    # Rename columns for consistency
    llm_df = llm_df.rename(columns={'abstract_id': 'ID'})
    
    # Merge on ID
    merged_df = pd.merge(
        golden_df, 
        llm_df[['ID', 'default_model_primary_drugs', 'default_model_secondary_drugs', 
                'default_model_comparator_drugs', 'default_model_reasoning']],
        on='ID',
        how='outer',
        indicator=True
    )
    
    print(f"\nMerge results:")
    print(f"  Both: {(merged_df['_merge'] == 'both').sum()}")
    print(f"  Golden only: {(merged_df['_merge'] == 'left_only').sum()}")
    print(f"  LLM only: {(merged_df['_merge'] == 'right_only').sum()}")
    
    # Initialize counters for metrics
    metrics = {
        'primary': {'tp': 0, 'fp': 0, 'fn': 0, 'misclassified': 0},
        'secondary': {'tp': 0, 'fp': 0, 'fn': 0, 'misclassified': 0},
        'comparator': {'tp': 0, 'fp': 0, 'fn': 0, 'misclassified': 0}
    }
    
    # Lists for new columns
    results = []
    
    for idx, row in merged_df.iterrows():
        result_row = {
            'ID': row['ID'],
            'Abstract Title': row['Abstract Title'],
        }
        
        # Parse golden drugs
        golden_primary = parse_drug_list_golden(row.get('Primary Drug', ''))
        golden_secondary = parse_drug_list_golden(row.get('Secondary Drug', ''))
        golden_comparator = parse_drug_list_golden(row.get('Comparator Drug', ''))
        all_golden = golden_primary | golden_secondary | golden_comparator
        
        # Parse LLM drugs
        llm_primary = parse_drug_list_llm(row.get('default_model_primary_drugs', ''))
        llm_secondary = parse_drug_list_llm(row.get('default_model_secondary_drugs', ''))
        llm_comparator = parse_drug_list_llm(row.get('default_model_comparator_drugs', ''))
        all_llm = llm_primary | llm_secondary | llm_comparator
        
        # Store original values
        result_row['golden_primary'] = set_to_str(golden_primary)
        result_row['golden_secondary'] = set_to_str(golden_secondary)
        result_row['golden_comparator'] = set_to_str(golden_comparator)
        result_row['llm_primary'] = set_to_str(llm_primary)
        result_row['llm_secondary'] = set_to_str(llm_secondary)
        result_row['llm_comparator'] = set_to_str(llm_comparator)
        
        # Compare Primary drugs
        matched_p, missed_p, halluc_p, misclass_p = compare_drug_sets(
            llm_primary, golden_primary, all_golden, all_llm
        )
        result_row['matched_primary'] = set_to_str(matched_p)
        result_row['missed_primary'] = set_to_str(missed_p)
        result_row['hallucinated_primary'] = set_to_str(halluc_p)
        result_row['misclassified_primary'] = set_to_str(misclass_p)
        
        metrics['primary']['tp'] += len(matched_p)
        metrics['primary']['fn'] += len(missed_p)
        metrics['primary']['fp'] += len(halluc_p)
        metrics['primary']['misclassified'] += len(misclass_p)
        
        # Compare Secondary drugs
        matched_s, missed_s, halluc_s, misclass_s = compare_drug_sets(
            llm_secondary, golden_secondary, all_golden, all_llm
        )
        result_row['matched_secondary'] = set_to_str(matched_s)
        result_row['missed_secondary'] = set_to_str(missed_s)
        result_row['hallucinated_secondary'] = set_to_str(halluc_s)
        result_row['misclassified_secondary'] = set_to_str(misclass_s)
        
        metrics['secondary']['tp'] += len(matched_s)
        metrics['secondary']['fn'] += len(missed_s)
        metrics['secondary']['fp'] += len(halluc_s)
        metrics['secondary']['misclassified'] += len(misclass_s)
        
        # Compare Comparator drugs
        matched_c, missed_c, halluc_c, misclass_c = compare_drug_sets(
            llm_comparator, golden_comparator, all_golden, all_llm
        )
        result_row['matched_comparator'] = set_to_str(matched_c)
        result_row['missed_comparator'] = set_to_str(missed_c)
        result_row['hallucinated_comparator'] = set_to_str(halluc_c)
        result_row['misclassified_comparator'] = set_to_str(misclass_c)
        
        metrics['comparator']['tp'] += len(matched_c)
        metrics['comparator']['fn'] += len(missed_c)
        metrics['comparator']['fp'] += len(halluc_c)
        metrics['comparator']['misclassified'] += len(misclass_c)
        
        # Calculate row-level accuracy
        total_golden = len(all_golden)
        total_matched = len(matched_p) + len(matched_s) + len(matched_c)
        result_row['total_golden_drugs'] = total_golden
        result_row['total_matched'] = total_matched
        result_row['row_accuracy'] = f"{total_matched}/{total_golden}" if total_golden > 0 else "N/A"
        
        # Add LLM reasoning for debugging
        reasoning = row.get('default_model_reasoning', '')
        result_row['LLM_Reasoning'] = reasoning if not pd.isna(reasoning) else ''
        
        results.append(result_row)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        'ID', 'Abstract Title',
        'golden_primary', 'llm_primary', 'matched_primary', 'missed_primary', 'hallucinated_primary', 'misclassified_primary',
        'golden_secondary', 'llm_secondary', 'matched_secondary', 'missed_secondary', 'hallucinated_secondary', 'misclassified_secondary',
        'golden_comparator', 'llm_comparator', 'matched_comparator', 'missed_comparator', 'hallucinated_comparator', 'misclassified_comparator',
        'total_golden_drugs', 'total_matched', 'row_accuracy', 'LLM_Reasoning'
    ]
    results_df = results_df[column_order]
    
    # Don't save individual CSV - will be included in Excel
    # results_df.to_csv(output_path, index=False)
    # print(f"\nDetailed evaluation saved to: {output_path}")
    
    # Calculate summary metrics
    summary = {}
    for category in ['primary', 'secondary', 'comparator']:
        tp = metrics[category]['tp']
        fp = metrics[category]['fp']
        fn = metrics[category]['fn']
        misclass = metrics[category]['misclassified']
        
        precision = tp / (tp + fp + misclass) if (tp + fp + misclass) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        summary[category] = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'misclassified': misclass,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        }
    
    # Overall metrics
    total_tp = sum(metrics[cat]['tp'] for cat in metrics)
    total_fp = sum(metrics[cat]['fp'] for cat in metrics)
    total_fn = sum(metrics[cat]['fn'] for cat in metrics)
    total_misclass = sum(metrics[cat]['misclassified'] for cat in metrics)
    
    overall_precision = total_tp / (total_tp + total_fp + total_misclass) if (total_tp + total_fp + total_misclass) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    summary['overall'] = {
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'misclassified': total_misclass,
        'precision': round(overall_precision, 4),
        'recall': round(overall_recall, 4),
        'f1_score': round(overall_f1, 4)
    }
    
    return summary, results_df


def print_summary(summary: Dict):
    """Print formatted summary metrics."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print("\nMetric Definitions:")
    print("  - True Positives (TP): Drugs correctly extracted in the right category")
    print("  - False Positives (FP): Drugs extracted that don't exist in golden data (hallucinations)")
    print("  - False Negatives (FN): Drugs in golden data that were missed by LLM")
    print("  - Misclassified: Drugs extracted in wrong category (e.g., Primary instead of Secondary)")
    
    for category in ['primary', 'secondary', 'comparator', 'overall']:
        cat_summary = summary[category]
        print(f"\n{'-'*40}")
        print(f"{category.upper()} DRUGS")
        print(f"{'-'*40}")
        print(f"  True Positives (Matched):    {cat_summary['true_positives']}")
        print(f"  False Positives (Halluc.):   {cat_summary['false_positives']}")
        print(f"  False Negatives (Missed):    {cat_summary['false_negatives']}")
        print(f"  Misclassified:               {cat_summary['misclassified']}")
        print(f"  Precision:                   {cat_summary['precision']:.2%}")
        print(f"  Recall:                      {cat_summary['recall']:.2%}")
        print(f"  F1 Score:                    {cat_summary['f1_score']:.2%}")
    
    print("\n" + "="*80)


def save_summary_to_csv(summary: Dict, output_path: str):
    """Save summary metrics to a separate CSV file."""
    rows = []
    for category in ['primary', 'secondary', 'comparator', 'overall']:
        row = {'category': category}
        row.update(summary[category])
        rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path, index=False)
    print(f"Summary metrics saved to: {output_path}")


def generate_simplified_report(
    llm_output_path: str,
    golden_data_path: str,
    output_path: str = None,
    errors_only: bool = False
):
    """
    Generate a simplified, readable evaluation report.
    
    Format: One row per abstract with combined analysis columns.
    Easier to manually review and identify issues.
    """
    # Load data
    llm_df = pd.read_csv(llm_output_path)
    golden_df = pd.read_csv(golden_data_path)
    
    llm_df = llm_df.rename(columns={'abstract_id': 'ID'})
    
    merged_df = pd.merge(
        golden_df, 
        llm_df[['ID', 'default_model_primary_drugs', 'default_model_secondary_drugs', 
                'default_model_comparator_drugs', 'default_model_reasoning']],
        on='ID',
        how='outer'
    )
    
    results = []
    
    for idx, row in merged_df.iterrows():
        # Parse all drugs
        golden_primary = parse_drug_list_golden(row.get('Primary Drug', ''))
        golden_secondary = parse_drug_list_golden(row.get('Secondary Drug', ''))
        golden_comparator = parse_drug_list_golden(row.get('Comparator Drug', ''))
        all_golden = golden_primary | golden_secondary | golden_comparator
        
        llm_primary = parse_drug_list_llm(row.get('default_model_primary_drugs', ''))
        llm_secondary = parse_drug_list_llm(row.get('default_model_secondary_drugs', ''))
        llm_comparator = parse_drug_list_llm(row.get('default_model_comparator_drugs', ''))
        all_llm = llm_primary | llm_secondary | llm_comparator
        
        # Compare each category
        matched_p, missed_p, halluc_p, misclass_p = compare_drug_sets(
            llm_primary, golden_primary, all_golden, all_llm
        )
        matched_s, missed_s, halluc_s, misclass_s = compare_drug_sets(
            llm_secondary, golden_secondary, all_golden, all_llm
        )
        matched_c, missed_c, halluc_c, misclass_c = compare_drug_sets(
            llm_comparator, golden_comparator, all_golden, all_llm
        )
        
        # Calculate totals
        total_matched = len(matched_p) + len(matched_s) + len(matched_c)
        total_missed = len(missed_p) + len(missed_s) + len(missed_c)
        total_halluc = len(halluc_p) + len(halluc_s) + len(halluc_c)
        total_misclass = len(misclass_p) + len(misclass_s) + len(misclass_c)
        total_golden = len(all_golden)
        
        # Determine status
        if total_golden == 0 and len(all_llm) == 0:
            status = "✓ CORRECT (No drugs)"
        elif total_missed == 0 and total_halluc == 0 and total_misclass == 0:
            status = "✓ PERFECT"
        elif total_matched > 0 and (total_missed > 0 or total_halluc > 0 or total_misclass > 0):
            status = "⚠ PARTIAL"
        else:
            status = "✗ MISMATCH"
        
        has_errors = total_missed > 0 or total_halluc > 0 or total_misclass > 0
        
        if errors_only and not has_errors:
            continue
        
        # Build combined columns
        def format_comparison(golden_set, llm_set, matched, missed, halluc, misclass):
            """Format a single category comparison."""
            if not golden_set and not llm_set:
                return "", "", "", ""
            
            golden_str = set_to_str(golden_set) if golden_set else "-"
            llm_str = set_to_str(llm_set) if llm_set else "-"
            
            issues = []
            if missed:
                issues.append(f"MISSED: {set_to_str(missed)}")
            if halluc:
                issues.append(f"HALLUC: {set_to_str(halluc)}")
            if misclass:
                issues.append(f"WRONG_CAT: {set_to_str(misclass)}")
            
            issues_str = " | ".join(issues) if issues else "✓"
            matched_str = set_to_str(matched) if matched else "-"
            
            return golden_str, llm_str, matched_str, issues_str
        
        # Format each category
        g_p, l_p, m_p, i_p = format_comparison(golden_primary, llm_primary, matched_p, missed_p, halluc_p, misclass_p)
        g_s, l_s, m_s, i_s = format_comparison(golden_secondary, llm_secondary, matched_s, missed_s, halluc_s, misclass_s)
        g_c, l_c, m_c, i_c = format_comparison(golden_comparator, llm_comparator, matched_c, missed_c, halluc_c, misclass_c)
        
        # Truncate title for readability
        title = str(row['Abstract Title'])
        title_short = title
        
        # Get reasoning (handle NaN)
        reasoning = row.get('default_model_reasoning', '')
        if pd.isna(reasoning):
            reasoning = ''
        
        result_row = {
            'ID': row['ID'],
            'Title': title_short,
            'Status': status,
            'Score': f"{total_matched}/{total_golden}" if total_golden > 0 else "N/A",
            # Primary
            'Primary_Golden': g_p,
            'Primary_LLM': l_p,
            'Primary_Issues': i_p,
            # Secondary  
            'Secondary_Golden': g_s,
            'Secondary_LLM': l_s,
            'Secondary_Issues': i_s,
            # Comparator
            'Comparator_Golden': g_c,
            'Comparator_LLM': l_c,
            'Comparator_Issues': i_c,
            # Reasoning for debugging
            'LLM_Reasoning': reasoning,
        }
        
        results.append(result_row)
    
    results_df = pd.DataFrame(results)
    # Don't save individual CSV - will be included in Excel
    # results_df.to_csv(output_path, index=False)
    
    return results_df


def generate_errors_only_report(
    llm_output_path: str,
    golden_data_path: str
):
    """
    Generate a flat report showing only errors - one row per error.
    Very easy to review and fix issues.
    """
    llm_df = pd.read_csv(llm_output_path)
    golden_df = pd.read_csv(golden_data_path)
    
    llm_df = llm_df.rename(columns={'abstract_id': 'ID'})
    
    merged_df = pd.merge(
        golden_df, 
        llm_df[['ID', 'default_model_primary_drugs', 'default_model_secondary_drugs', 
                'default_model_comparator_drugs', 'default_model_reasoning']],
        on='ID',
        how='outer'
    )
    
    errors = []
    
    for idx, row in merged_df.iterrows():
        abstract_id = row['ID']
        title = str(row['Abstract Title'])
        title_short = title
        
        # Get reasoning (handle NaN)
        reasoning = row.get('default_model_reasoning', '')
        if pd.isna(reasoning):
            reasoning = ''
        
        # Parse all drugs
        golden_primary = parse_drug_list_golden(row.get('Primary Drug', ''))
        golden_secondary = parse_drug_list_golden(row.get('Secondary Drug', ''))
        golden_comparator = parse_drug_list_golden(row.get('Comparator Drug', ''))
        all_golden = golden_primary | golden_secondary | golden_comparator
        
        llm_primary = parse_drug_list_llm(row.get('default_model_primary_drugs', ''))
        llm_secondary = parse_drug_list_llm(row.get('default_model_secondary_drugs', ''))
        llm_comparator = parse_drug_list_llm(row.get('default_model_comparator_drugs', ''))
        all_llm = llm_primary | llm_secondary | llm_comparator
        
        categories = [
            ('Primary', golden_primary, llm_primary),
            ('Secondary', golden_secondary, llm_secondary),
            ('Comparator', golden_comparator, llm_comparator)
        ]
        
        for cat_name, golden_set, llm_set in categories:
            matched, missed, halluc, misclass = compare_drug_sets(
                llm_set, golden_set, all_golden, all_llm
            )
            
            # Log each missed drug
            for drug in missed:
                errors.append({
                    'ID': abstract_id,
                    'Title': title_short,
                    'Category': cat_name,
                    'Error_Type': 'MISSED',
                    'Drug': drug,
                    'Details': f"In Golden {cat_name}, not found in LLM",
                    'LLM_Reasoning': reasoning
                })
            
            # Log each hallucinated drug
            for drug in halluc:
                errors.append({
                    'ID': abstract_id,
                    'Title': title_short,
                    'Category': cat_name,
                    'Error_Type': 'HALLUCINATED',
                    'Drug': drug,
                    'Details': f"In LLM {cat_name}, not in Golden anywhere",
                    'LLM_Reasoning': reasoning
                })
            
            # Log each misclassified drug
            for drug in misclass:
                # Find where it actually belongs in golden
                actual_cat = []
                if drug in golden_primary:
                    actual_cat.append('Primary')
                if drug in golden_secondary:
                    actual_cat.append('Secondary')
                if drug in golden_comparator:
                    actual_cat.append('Comparator')
                
                errors.append({
                    'ID': abstract_id,
                    'Title': title_short,
                    'Category': cat_name,
                    'Error_Type': 'MISCLASSIFIED',
                    'Drug': drug,
                    'Details': f"LLM: {cat_name}, Golden: {', '.join(actual_cat)}",
                    'LLM_Reasoning': reasoning
                })
    
    errors_df = pd.DataFrame(errors)
    # Don't save individual CSV - will be included in Excel
    # errors_df.to_csv(output_path, index=False)
    
    return errors_df


def export_to_excel(
    summary: Dict,
    simplified_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    llm_output_df: pd.DataFrame,
    excel_output_path: str
):
    """
    Export all evaluation results to a single Excel file with multiple sheets.
    """
    # Create summary DataFrame
    summary_rows = []
    for category in ['primary', 'secondary', 'comparator', 'overall']:
        row = {'Category': category.upper()}
        row['True Positives (Matched)'] = summary[category]['true_positives']
        row['False Positives (Hallucinated)'] = summary[category]['false_positives']
        row['False Negatives (Missed)'] = summary[category]['false_negatives']
        row['Misclassified'] = summary[category]['misclassified']
        row['Precision'] = f"{summary[category]['precision']:.2%}"
        row['Recall'] = f"{summary[category]['recall']:.2%}"
        row['F1 Score'] = f"{summary[category]['f1_score']:.2%}"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    
    # Create status distribution DataFrame
    status_dist = simplified_df['Status'].value_counts().reset_index()
    status_dist.columns = ['Status', 'Count']
    
    # Create error type distribution DataFrame
    if len(errors_df) > 0:
        error_dist = errors_df['Error_Type'].value_counts().reset_index()
        error_dist.columns = ['Error Type', 'Count']
    else:
        error_dist = pd.DataFrame({'Error Type': [], 'Count': []})
    
    # Write to Excel with multiple sheets
    with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
        # Sheet 1: Summary Metrics
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Add status distribution below summary (with some spacing)
        startrow = len(summary_df) + 3
        status_dist.to_excel(writer, sheet_name='Summary', startrow=startrow, index=False)
        
        # Add error distribution below status
        startrow = startrow + len(status_dist) + 3
        error_dist.to_excel(writer, sheet_name='Summary', startrow=startrow, index=False)
        
        # Sheet 2: Simplified Report (easy to review)
        simplified_df.to_excel(writer, sheet_name='Simplified', index=False)
        
        # Sheet 3: Errors Only (for debugging)
        errors_df.to_excel(writer, sheet_name='Errors', index=False)
        
        # Sheet 4: Detailed Evaluation (full data)
        detailed_df.to_excel(writer, sheet_name='Detailed', index=False)
        
        # Sheet 5: Original LLM Output (raw data)
        llm_output_df.to_excel(writer, sheet_name='LLM_Output', index=False)
    
    print(f"\n{'='*60}")
    print(f"Excel report saved to: {excel_output_path}")
    print(f"{'='*60}")
    print("Sheets included:")
    print("  1. Summary    - Metrics, status distribution, error distribution")
    print("  2. Simplified - Easy-to-read per-abstract analysis")
    print("  3. Errors     - Flat list of all errors (filterable)")
    print("  4. Detailed   - Full evaluation with all columns")
    print("  5. LLM_Output - Original LLM output (raw data)")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate drug extraction results.')
    parser.add_argument('input_file', nargs='?', help='Path to the LLM output CSV file')
    args = parser.parse_args()

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.input_file:
        llm_output_path = args.input_file
        # Generate report path based on input filename
        input_dir = os.path.dirname(llm_output_path)
        input_filename = os.path.basename(llm_output_path)
        filename_no_ext = os.path.splitext(input_filename)[0]
        excel_output_path = os.path.join(input_dir, f"{filename_no_ext}_report.xlsx")
    else:
        # Default path
        llm_output_path = os.path.join(script_dir, "result", "claude-sonnet-4-5.csv")
        excel_output_path = os.path.join(script_dir, "result", "claude-sonnet-4-5_report.xlsx")
    
    golden_data_path = os.path.join(script_dir, "data", "golden_data_drug.csv")
    
    # Run detailed evaluation
    print("Running evaluation...")
    summary, detailed_df = evaluate_drug_extraction(
        llm_output_path=llm_output_path,
        golden_data_path=golden_data_path,
        output_path=None  # Not saving CSV
    )
    
    # Print summary
    print_summary(summary)
    
    # Generate simplified report (all rows)
    print("\nGenerating simplified report...")
    simplified_df = generate_simplified_report(
        llm_output_path=llm_output_path,
        golden_data_path=golden_data_path,
        output_path=None  # Not saving CSV
    )
    
    # Print status distribution
    if 'Status' in simplified_df.columns:
        print("\nStatus Distribution:")
        for status, count in simplified_df['Status'].value_counts().items():
            print(f"  {status}: {count}")
    
    # Generate errors-only report
    print("\nGenerating errors report...")
    errors_df = generate_errors_only_report(
        llm_output_path=llm_output_path,
        golden_data_path=golden_data_path
    )
    print(f"Total errors found: {len(errors_df)}")
    
    if len(errors_df) > 0:
        print("\nError Type Distribution:")
        for error_type, count in errors_df['Error_Type'].value_counts().items():
            print(f"  {error_type}: {count}")
    
    # Load original LLM output for inclusion in Excel
    llm_output_df = pd.read_csv(llm_output_path)
    
    # Export all to single Excel file
    print("\nExporting to Excel...")
    export_to_excel(
        summary=summary,
        simplified_df=simplified_df,
        errors_df=errors_df,
        detailed_df=detailed_df,
        llm_output_df=llm_output_df,
        excel_output_path=excel_output_path
    )

