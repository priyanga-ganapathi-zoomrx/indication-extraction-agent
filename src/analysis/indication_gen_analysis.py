#!/usr/bin/env python3
"""
Indication Generation Analysis Script

This script analyzes model-generated indications from CSV data, comparing performance
across multiple models and against ground truth data. It provides comprehensive
metrics, visualizations, and export capabilities.

Usage:
    python indication_gen_analysis.py --input_file path/to/data.csv --output_dir ./results/
"""

import pandas as pd
import json
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
from difflib import SequenceMatcher
import string
import argparse
import os
from datetime import datetime

# For ground truth analysis
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    print("✅ Ground truth analysis libraries available")
    GROUND_TRUTH_AVAILABLE = True
except ImportError:
    print("⚠️ Install nltk and scikit-learn for full ground truth analysis: pip install nltk scikit-learn")
    sentence_bleu = None
    TfidfVectorizer = None
    cosine_similarity = None
    GROUND_TRUTH_AVAILABLE = False

# For Excel export
try:
    import openpyxl
    print("✅ openpyxl available for Excel export")
    EXCEL_AVAILABLE = True
except ImportError:
    print("⚠️ openpyxl not available. Install with: pip install openpyxl")
    print("Will fallback to CSV export if needed")
    EXCEL_AVAILABLE = False

# For visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✅ Visualization libraries available")
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("⚠️ Install matplotlib and seaborn for visualizations: pip install matplotlib seaborn")
    print("Will skip visualization generation")
    plt = None
    sns = None
    VISUALIZATION_AVAILABLE = False

print("=== Indication Generation Analysis Script ===")


def parse_indication_response(response_text: str) -> Dict:
    """
    Parse model JSON response and extract generated indication.
    Handles both JSON format and direct text format.

    Args:
        response_text: Raw model response text

    Returns:
        Dictionary with success status and extracted indication
    """
    try:
        # Handle missing/NaN responses as successful "no indication found"
        if pd.isna(response_text):
            return {
                "success": True,
                "indication": "No indication found",
                "confidence_score": None,
                "components_used": [],
                "clinical_notes": ["No medical indication identified in abstract"],
                "quality_assessment": {"no_indication": True},
                "full_response": {"no_indication_found": True},
                "error": None
            }

        # Handle empty/whitespace-only responses as successful "no indication found"
        if isinstance(response_text, str) and not response_text.strip():
            return {
                "success": True,
                "indication": "No indication found",
                "confidence_score": None,
                "components_used": [],
                "clinical_notes": ["No medical indication identified in abstract"],
                "quality_assessment": {"no_indication": True},
                "full_response": {"no_indication_found": True},
                "error": None
            }

        # Clean the response text
        clean_response = response_text
        if isinstance(response_text, str):
            clean_response = re.sub(r'```json\n?', '', response_text)
            clean_response = re.sub(r'```\n?', '', clean_response).strip()

        # Try to parse as JSON first
        try:
            parsed = json.loads(clean_response)

            # Extract indication from JSON format
            indication = ""
            if 'generated_indication' in parsed:
                indication = parsed['generated_indication'].strip()

            return {
                "success": True,
                "indication": indication,
                "confidence_score": parsed.get('confidence_score', None),
                "components_used": parsed.get('components_used', []),
                "clinical_notes": parsed.get('clinical_notes', []),
                "quality_assessment": parsed.get('quality_assessment', {}),
                "full_response": parsed,
                "error": None
            }

        except json.JSONDecodeError:
            # If JSON parsing fails, treat as direct text response
            indication = clean_response.strip()

            # Handle cases where model correctly found no indication
            if not indication:
                # This should not happen since we handle empty responses above
                return {
                    "success": True,
                    "indication": "No indication found",
                    "confidence_score": None,
                    "components_used": [],
                    "clinical_notes": ["No medical indication identified in abstract"],
                    "quality_assessment": {"no_indication": True},
                    "full_response": {"no_indication_found": True},
                    "error": None
                }
            elif len(indication) <= 2:
                return {
                    "success": False,
                    "indication": "",
                    "error": "Invalid direct text response (too short)",
                    "full_response": {}
                }
            else:
                return {
                    "success": True,
                    "indication": indication,
                    "confidence_score": None,  # No confidence score in direct text format
                    "components_used": [],
                    "clinical_notes": [],
                    "quality_assessment": {},
                    "full_response": {"direct_text": indication},
                    "error": None
                }

    except Exception as e:
        return {
            "success": False,
            "indication": "",
            "error": f"General error: {str(e)}",
            "full_response": {}
        }


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings using SequenceMatcher.

    Args:
        text1, text2: Text strings to compare

    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0

    # Normalize text (lowercase, remove punctuation)
    def normalize_text(text):
        text = text.lower().strip()
        # Remove punctuation but keep spaces
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        # Remove extra spaces
        text = ' '.join(text.split())
        return text

    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)

    return SequenceMatcher(None, norm_text1, norm_text2).ratio()


def analyze_indication_characteristics(indication: str) -> Dict:
    """
    Analyze characteristics of an indication text.

    Args:
        indication: The indication text to analyze

    Returns:
        Dictionary with various metrics
    """
    if not indication or pd.isna(indication):
        return {
            "length": 0,
            "word_count": 0,
            "sentence_count": 0,
            "has_parentheses": False,
            "has_abbreviations": False,
            "clinical_terms": [],
            "clinical_term_count": 0,
            "complexity_score": 0
        }

    # Basic metrics
    length = len(indication)
    words = indication.split()
    word_count = len(words)

    # Sentence count (approximate)
    sentence_count = indication.count('.') + indication.count('!') + indication.count('?') + 1

    # Check for parentheses and abbreviations
    has_parentheses = '(' in indication and ')' in indication
    has_abbreviations = bool(re.search(r'\b[A-Z]{2,}\b', indication))

    # Common clinical terms (basic list)
    clinical_keywords = [
        'acute', 'chronic', 'severe', 'moderate', 'mild', 'advanced', 'early',
        'disease', 'syndrome', 'disorder', 'cancer', 'tumor', 'leukemia',
        'myeloma', 'lymphoma', 'carcinoma', 'therapy', 'treatment', 'patients',
        'diagnosis', 'stage', 'grade', 'positive', 'negative', 'deficient',
        'resistant', 'refractory', 'relapsed', 'newly', 'previously'
    ]

    clinical_terms = [term for term in clinical_keywords
                     if term.lower() in indication.lower()]

    # Simple complexity score (0-1)
    complexity_score = min(1.0, (word_count / 20.0) + (len(clinical_terms) / 10.0))

    return {
        "length": length,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "has_parentheses": has_parentheses,
        "has_abbreviations": has_abbreviations,
        "clinical_terms": clinical_terms,
        "clinical_term_count": len(clinical_terms),
        "complexity_score": complexity_score
    }


def compare_indications_between_models(indications: Dict[str, str]) -> Dict:
    """
    Compare indications between multiple models.

    Args:
        indications: Dictionary with model names as keys and indications as values

    Returns:
        Dictionary with pairwise similarity scores and consensus metrics
    """
    model_names = list(indications.keys())
    similarities = {}

    # Calculate pairwise similarities
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:  # Avoid duplicates
                sim = calculate_text_similarity(indications[model1], indications[model2])
                similarities[f"{model1}_vs_{model2}"] = sim

    # Calculate consensus metrics
    all_similarities = list(similarities.values())
    avg_similarity = np.mean(all_similarities) if all_similarities else 0
    min_similarity = min(all_similarities) if all_similarities else 0
    max_similarity = max(all_similarities) if all_similarities else 0

    # Simple consensus indicator
    high_consensus = avg_similarity > 0.7
    moderate_consensus = 0.4 <= avg_similarity <= 0.7
    low_consensus = avg_similarity < 0.4

    consensus_level = "High" if high_consensus else "Moderate" if moderate_consensus else "Low"

    return {
        "pairwise_similarities": similarities,
        "average_similarity": avg_similarity,
        "min_similarity": min_similarity,
        "max_similarity": max_similarity,
        "consensus_level": consensus_level,
        "model_count": len([ind for ind in indications.values() if ind and ind.strip()])
    }


def calculate_ground_truth_metrics(generated_indication: str, ground_truth: str) -> Dict:
    """
    Calculate comprehensive metrics between generated indication and ground truth.

    Args:
        generated_indication: Model-generated indication text
        ground_truth: Ground truth indication text

    Returns:
        Dictionary with various similarity and accuracy metrics
    """
    if not generated_indication or not ground_truth or pd.isna(generated_indication) or pd.isna(ground_truth):
        return {
            "exact_match": False,
            "text_similarity": 0.0,
            "bleu_score": 0.0,
            "cosine_similarity": 0.0,
            "length_ratio": 0.0,
            "word_overlap": 0.0,
            "error": "Missing text"
        }

    # Clean and normalize texts
    gen_clean = str(generated_indication).strip()
    gt_clean = str(ground_truth).strip()

    if not gen_clean or not gt_clean:
        return {
            "exact_match": False,
            "text_similarity": 0.0,
            "bleu_score": 0.0,
            "cosine_similarity": 0.0,
            "length_ratio": 0.0,
            "word_overlap": 0.0,
            "error": "Empty text"
        }

    gen_norm = gen_clean.lower()
    gt_norm = gt_clean.lower()

    # 1. Exact match
    exact_match = gen_norm == gt_norm

    # 2. Text similarity (SequenceMatcher)
    text_similarity = SequenceMatcher(None, gen_norm, gt_norm).ratio()

    # 3. BLEU Score
    bleu_score = 0.0
    if sentence_bleu is not None:
        try:
            reference = [gt_norm.split()]
            candidate = gen_norm.split()
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
        except:
            bleu_score = 0.0

    # 4. Cosine Similarity (TF-IDF)
    cosine_sim = 0.0
    if TfidfVectorizer is not None and cosine_similarity is not None:
        try:
            vectorizer = TfidfVectorizer().fit([gen_norm, gt_norm])
            vectors = vectorizer.transform([gen_norm, gt_norm])
            cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except:
            cosine_sim = 0.0

    # 5. Length ratio
    length_ratio = len(gen_clean) / len(gt_clean) if len(gt_clean) > 0 else 0

    # 6. Word overlap
    gen_words = set(gen_norm.split())
    gt_words = set(gt_norm.split())
    if len(gt_words) > 0:
        word_overlap = len(gen_words.intersection(gt_words)) / len(gt_words)
    else:
        word_overlap = 0.0

    return {
        "exact_match": exact_match,
        "text_similarity": text_similarity,
        "bleu_score": bleu_score,
        "cosine_similarity": cosine_sim,
        "length_ratio": length_ratio,
        "word_overlap": word_overlap,
        "error": None
    }


def create_visualizations(summary_df: pd.DataFrame, gt_summary_df: pd.DataFrame,
                         inter_model_comparisons: List[Dict], output_dir: str) -> None:
    """
    Create comprehensive visualizations and save to files.

    Args:
        summary_df: Model performance summary DataFrame
        gt_summary_df: Ground truth performance summary DataFrame
        inter_model_comparisons: List of inter-model comparison results
        output_dir: Directory to save visualization files
    """
    if not VISUALIZATION_AVAILABLE:
        print("\n⚠️ Visualization libraries not available. Skipping visualization generation.")
        return

    print("\n=== Creating Visualizations ===")

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    # Row 1: Basic Model Performance
    # Success Rate
    axes[0, 0].bar(summary_df['Model'], summary_df['Success_Rate_%'], color='skyblue')
    axes[0, 0].set_title('Success Rate (%)')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Average Word Count
    axes[0, 1].bar(summary_df['Model'], summary_df['Avg_Word_Count'], color='lightgreen')
    axes[0, 1].set_title('Average Word Count')
    axes[0, 1].set_ylabel('Word Count')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Average Clinical Terms
    axes[0, 2].bar(summary_df['Model'], summary_df['Avg_Clinical_Terms'], color='lightcoral')
    axes[0, 2].set_title('Average Clinical Terms')
    axes[0, 2].set_ylabel('Clinical Terms')
    axes[0, 2].tick_params(axis='x', rotation=45)

    # Row 2: Ground Truth Analysis
    if not gt_summary_df.empty and gt_summary_df['GT_Comparisons'].sum() > 0:
        # Exact Match Rate
        gt_models = gt_summary_df[gt_summary_df['GT_Comparisons'] > 0]
        if not gt_models.empty:
            axes[1, 0].bar(gt_models['Model'], gt_models['Exact_Match_Rate_%'], color='gold')
            axes[1, 0].set_title('Exact Match Rate (%)')
            axes[1, 0].set_ylabel('Exact Match Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Text Similarity
            axes[1, 1].bar(gt_models['Model'], gt_models['Avg_Text_Similarity'], color='orange')
            axes[1, 1].set_title('Avg Text Similarity vs GT')
            axes[1, 1].set_ylabel('Similarity Score')
            axes[1, 1].tick_params(axis='x', rotation=45)

            # BLEU Score
            axes[1, 2].bar(gt_models['Model'], gt_models['Avg_BLEU_Score'], color='purple')
            axes[1, 2].set_title('Avg BLEU Score vs GT')
            axes[1, 2].set_ylabel('BLEU Score')
            axes[1, 2].tick_params(axis='x', rotation=45)
        else:
            for i in range(3):
                axes[1, i].text(0.5, 0.5, 'No Ground Truth\nData Available',
                               ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'Ground Truth Analysis {i+1}')
    else:
        for i in range(3):
            axes[1, i].text(0.5, 0.5, 'No Ground Truth\nData Available',
                           ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title(f'Ground Truth Analysis {i+1}')

    # Row 3: Advanced Metrics
    # Complexity Score
    axes[2, 0].bar(summary_df['Model'], summary_df['Avg_Complexity'], color='mediumpurple')
    axes[2, 0].set_title('Average Complexity Score')
    axes[2, 0].set_ylabel('Complexity Score')
    axes[2, 0].tick_params(axis='x', rotation=45)

    # Abbreviations Usage
    axes[2, 1].bar(summary_df['Model'], summary_df['Pct_With_Abbreviations'], color='cyan')
    axes[2, 1].set_title('% Using Abbreviations')
    axes[2, 1].set_ylabel('Percentage')
    axes[2, 1].tick_params(axis='x', rotation=45)

    # Inter-Model Consensus Distribution
    if inter_model_comparisons:
        consensus_data = [comp['average_similarity'] for comp in inter_model_comparisons]
        axes[2, 2].hist(consensus_data, bins=10, color='salmon', alpha=0.7)
        axes[2, 2].set_title('Inter-Model Similarity Distribution')
        axes[2, 2].set_xlabel('Similarity Score')
        axes[2, 2].set_ylabel('Frequency')
    else:
        axes[2, 2].text(0.5, 0.5, 'No Inter-Model\nComparisons Available',
                       ha='center', va='center', transform=axes[2, 2].transAxes)
        axes[2, 2].set_title('Inter-Model Consensus')

    plt.tight_layout()

    # Save the comprehensive plot
    viz_path = os.path.join(output_dir, 'comprehensive_analysis.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Comprehensive visualization saved to: {viz_path}")


def export_results(consolidated_df: pd.DataFrame, summary_df: pd.DataFrame,
                  gt_summary_df: pd.DataFrame, all_gt_matches: List[Dict],
                  inter_model_comparisons: List[Dict], df: pd.DataFrame,
                  models: List[str], output_dir: str) -> None:
    """
    Export comprehensive results to Excel and CSV files.

    Args:
        consolidated_df: Consolidated results DataFrame
        summary_df: Model performance summary
        gt_summary_df: Ground truth performance summary
        all_gt_matches: All ground truth match results
        inter_model_comparisons: Inter-model comparison results
        df: Original data DataFrame
        models: List of model names
        output_dir: Output directory for results
    """
    print("\n=== Exporting Results ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"Indication_Generation_Analysis_{timestamp}"

    try:
        if EXCEL_AVAILABLE:
            # Create Excel writer object
            excel_filename = os.path.join(output_dir, f"{base_filename}.xlsx")
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:

                # Sheet 1: Consolidated Results (main sheet with all metrics)
                consolidated_df.to_excel(writer, sheet_name='Consolidated_Results', index=False)

                # Sheet 2: Model Summary (basic performance)
                summary_df.to_excel(writer, sheet_name='Model_Summary', index=False)

                # Sheet 3: Ground Truth Summary (NEW)
                if not gt_summary_df.empty:
                    gt_summary_df.to_excel(writer, sheet_name='Ground_Truth_Summary', index=False)
                else:
                    # Create empty GT summary with structure
                    empty_gt_df = pd.DataFrame({
                        'Model': models,
                        'Total_Successful': [0] * len(models),
                        'GT_Comparisons': [0] * len(models),
                        'Exact_Matches': [0] * len(models),
                        'Exact_Match_Rate_%': [0.0] * len(models),
                        'Avg_Text_Similarity': [0.0] * len(models),
                        'Avg_BLEU_Score': [0.0] * len(models),
                        'Avg_Cosine_Similarity': [0.0] * len(models),
                        'Avg_Length_Ratio': [0.0] * len(models),
                        'Avg_Word_Overlap': [0.0] * len(models)
                    })
                    empty_gt_df.to_excel(writer, sheet_name='Ground_Truth_Summary', index=False)

                # Sheet 4: Best GT Matches (NEW)
                if all_gt_matches:
                    best_gt_matches = sorted(all_gt_matches, key=lambda x: x['Text_Similarity'], reverse=True)[:20]
                    best_gt_df = pd.DataFrame(best_gt_matches)
                    best_gt_df.to_excel(writer, sheet_name='Best_GT_Matches', index=False)
                else:
                    # Create empty structure
                    empty_best_gt = pd.DataFrame(columns=[
                        'Model', 'Abstract_ID', 'Generated_Indication',
                        'Text_Similarity', 'BLEU_Score', 'Cosine_Similarity', 'Exact_Match',
                        'Word_Overlap', 'Length_Ratio'
                    ])
                    empty_best_gt.to_excel(writer, sheet_name='Best_GT_Matches', index=False)

                # Sheet 5: Worst GT Matches (NEW)
                if all_gt_matches:
                    worst_gt_matches = sorted(all_gt_matches, key=lambda x: x['Text_Similarity'])[:20]
                    worst_gt_df = pd.DataFrame(worst_gt_matches)
                    worst_gt_df.to_excel(writer, sheet_name='Worst_GT_Matches', index=False)
                else:
                    # Create empty structure
                    empty_worst_gt = pd.DataFrame(columns=[
                        'Model', 'Abstract_ID', 'Generated_Indication',
                        'Text_Similarity', 'BLEU_Score', 'Cosine_Similarity', 'Exact_Match',
                        'Word_Overlap', 'Length_Ratio'
                    ])
                    empty_worst_gt.to_excel(writer, sheet_name='Worst_GT_Matches', index=False)

                # Sheet 6: Exact Matches (NEW)
                if all_gt_matches:
                    exact_matches = [match for match in all_gt_matches if match['Exact_Match']]
                    if exact_matches:
                        exact_matches_df = pd.DataFrame(exact_matches)
                        exact_matches_df.to_excel(writer, sheet_name='Exact_Matches', index=False)
                    else:
                        empty_exact = pd.DataFrame(columns=[
                            'Model', 'Abstract_ID', 'Generated_Indication',
                            'Text_Similarity', 'BLEU_Score', 'Cosine_Similarity', 'Exact_Match',
                            'Word_Overlap', 'Length_Ratio'
                        ])
                        empty_exact.to_excel(writer, sheet_name='Exact_Matches', index=False)
                else:
                    empty_exact = pd.DataFrame(columns=[
                        'Model', 'Abstract_ID', 'Generated_Indication',
                        'Text_Similarity', 'BLEU_Score', 'Cosine_Similarity', 'Exact_Match',
                        'Word_Overlap', 'Length_Ratio'
                    ])
                    empty_exact.to_excel(writer, sheet_name='Exact_Matches', index=False)

                # Sheet 7: Inter-Model Comparisons
                if inter_model_comparisons:
                    comparison_df = pd.DataFrame(inter_model_comparisons)
                    comparison_df.to_excel(writer, sheet_name='Inter_Model_Comparisons', index=False)
                else:
                    # Create empty structure
                    empty_comparison = pd.DataFrame(columns=[
                        'abstract_id', 'pairwise_similarities', 'average_similarity',
                        'min_similarity', 'max_similarity', 'consensus_level', 'model_count'
                    ])
                    empty_comparison.to_excel(writer, sheet_name='Inter_Model_Comparisons', index=False)

                print(f"✅ Excel file created: {excel_filename}")

        # Always export to CSV as backup
        csv_dir = os.path.join(output_dir, "csv_exports")
        os.makedirs(csv_dir, exist_ok=True)

        consolidated_df.to_csv(os.path.join(csv_dir, 'consolidated_results.csv'), index=False)
        summary_df.to_csv(os.path.join(csv_dir, 'model_summary.csv'), index=False)

        if not gt_summary_df.empty:
            gt_summary_df.to_csv(os.path.join(csv_dir, 'ground_truth_summary.csv'), index=False)

        if all_gt_matches:
            pd.DataFrame(all_gt_matches).to_csv(os.path.join(csv_dir, 'ground_truth_matches.csv'), index=False)

        if inter_model_comparisons:
            pd.DataFrame(inter_model_comparisons).to_csv(os.path.join(csv_dir, 'inter_model_comparisons.csv'), index=False)

        print(f"✅ CSV files exported to: {csv_dir}")

    except Exception as e:
        print(f"⚠️ Error during export: {str(e)}")
        # Fallback to basic CSV export
        try:
            consolidated_df.to_csv(os.path.join(output_dir, 'consolidated_results.csv'), index=False)
            summary_df.to_csv(os.path.join(output_dir, 'model_summary.csv'), index=False)
            print(f"✅ Fallback CSV export completed in: {output_dir}")
        except Exception as e2:
            print(f"❌ Export failed completely: {str(e2)}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Indication Generation Analysis')
    parser.add_argument('--input_file', required=True,
                       help='Path to input CSV file with indication results')
    parser.add_argument('--output_dir', default='./results',
                       help='Output directory for results (default: ./results)')
    parser.add_argument('--models', nargs='+', default=['o3', 'Gemini-2.5-pro'],
                       help='Model names to analyze (default: o3 Gemini-2.5-pro)')
    parser.add_argument('--successful_only', action='store_true',
                       help='Only analyze abstracts where all specified models responded successfully')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models to analyze: {', '.join(args.models)}")

    # Load the data
    try:
        df = pd.read_csv(args.input_file)
        original_count = len(df)
        print(f"✅ Dataset loaded: {original_count} records")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return

    print(f"- Columns: {len(df.columns)}")
    print(f"- Models analyzed: {', '.join(args.models)}")

    # Filter for successful responses only if requested
    if args.successful_only:
        success_conditions = []
        for model in args.models:
            success_col = f'{model}_success'
            if success_col in df.columns:
                success_conditions.append(df[success_col].isin(['True', True]))
            else:
                print(f"⚠️ Warning: Column '{success_col}' not found in data")
                continue

        if success_conditions:
            combined_condition = pd.concat(success_conditions, axis=1).all(axis=1)
            df = df[combined_condition].copy()
            filtered_count = len(df)
            print(f"🔍 Filtered for successful responses only: {filtered_count}/{original_count} records retained")

    # Check for ground truth
    ground_truth_available = df['ground_truth'].notna().sum()
    print(f"- Ground truth available: {ground_truth_available}/{len(df)} records")

    # Display basic info
    print(f"\nSample data:")
    print(df[['abstract_id', 'abstract_title']].head(3))

    # Model columns
    models = args.models
    success_columns = [f'{model}_success' for model in models]
    response_columns = [f'{model}_indication_response' for model in models]

    # Check success rates
    print(f"\nInitial Success Rates:")
    for model, col in zip(models, success_columns):
        if col in df.columns:
            success_count = (df[col] == 'True').sum() if df[col].dtype == 'object' else df[col].sum()
            success_rate = success_count / len(df) * 100
            print(f"- {model}: {success_count}/{len(df)} ({success_rate:.1f}%)")

    # Initialize results storage
    results = {}
    for model in models:
        results[model] = {
            'successes': 0,
            'parsing_errors': 0,
            'indications': [],
            'characteristics': [],
            'details': []
        }

    # Store inter-model comparisons
    inter_model_comparisons = []

    print("\n=== Processing All Records ===")

    # Process each record
    for idx, record in df.iterrows():

        # Parse each model's response
        record_indications = {}
        record_details = {
            'abstract_id': record['abstract_id'],
            'abstract_title': record['abstract_title'],
            'ground_truth': record.get('ground_truth', None)
        }

        for model in models:
            response_col = f'{model}_indication_response'
            success_col = f'{model}_success'

            if record[success_col] in ['True', True]:
                parsed = parse_indication_response(record[response_col])

                if parsed['success']:
                    results[model]['successes'] += 1
                    indication = parsed['indication']
                    results[model]['indications'].append(indication)

                    # Analyze characteristics
                    characteristics = analyze_indication_characteristics(indication)
                    results[model]['characteristics'].append(characteristics)

                    # GROUND TRUTH ANALYSIS
                    gt_metrics = {}
                    if record.get('ground_truth') and pd.notna(record['ground_truth']) and str(record['ground_truth']).strip():
                        gt_metrics = calculate_ground_truth_metrics(indication, record['ground_truth'])
                    else:
                        gt_metrics = {
                            "exact_match": False,
                            "text_similarity": 0.0,
                            "bleu_score": 0.0,
                            "cosine_similarity": 0.0,
                            "length_ratio": 0.0,
                            "word_overlap": 0.0,
                            "error": "No ground truth available"
                        }

                    # Store details
                    detail = {
                        'abstract_id': record['abstract_id'],
                        'indication': indication,
                        'confidence_score': parsed['confidence_score'],
                        'characteristics': characteristics,
                        'quality_assessment': parsed['quality_assessment'],
                        'ground_truth_metrics': gt_metrics
                    }
                    results[model]['details'].append(detail)

                    # Store for inter-model comparison
                    record_indications[model] = indication
                    record_details[f'{model}_indication'] = indication
                    record_details[f'{model}_confidence'] = parsed['confidence_score']
                    record_details[f'{model}_word_count'] = characteristics['word_count']
                    record_details[f'{model}_clinical_terms'] = characteristics['clinical_term_count']

                    # Add ground truth metrics to record details
                    for gt_key, gt_value in gt_metrics.items():
                        record_details[f'{model}_gt_{gt_key}'] = gt_value

                else:
                    results[model]['parsing_errors'] += 1
                    record_details[f'{model}_indication'] = ""
                    record_details[f'{model}_error'] = parsed['error']
                    # Add empty ground truth metrics
                    for gt_key in ['exact_match', 'text_similarity', 'bleu_score', 'cosine_similarity', 'length_ratio', 'word_overlap']:
                        record_details[f'{model}_gt_{gt_key}'] = 0.0 if gt_key != 'exact_match' else False
            else:
                record_details[f'{model}_indication'] = ""
                record_details[f'{model}_error'] = "Model failed"
                # Add empty ground truth metrics
                for gt_key in ['exact_match', 'text_similarity', 'bleu_score', 'cosine_similarity', 'length_ratio', 'word_overlap']:
                    record_details[f'{model}_gt_{gt_key}'] = 0.0 if gt_key != 'exact_match' else False

        # Compare indications between models for this record
        if len(record_indications) > 1:
            comparison = compare_indications_between_models(record_indications)
            record_details['inter_model_comparison'] = comparison
            inter_model_comparisons.append({
                'abstract_id': record['abstract_id'],
                **comparison
            })

    print(f"Processing complete! Analyzed {len(df)} abstracts.")

    # Generate summaries
    print("\n=== MODEL PERFORMANCE SUMMARY ===")

    summary_data = []
    total_records = len(df)

    for model in models:
        model_results = results[model]

        print(f"\n--- {model} ---")
        success_rate = (model_results['successes'] / total_records) * 100
        print(f"Success Rate: {model_results['successes']}/{total_records} ({success_rate:.1f}%)")
        print(f"Parsing Errors: {model_results['parsing_errors']}")

        if model_results['successes'] > 0:
            # Calculate indication characteristics
            all_chars = model_results['characteristics']

            avg_word_count = np.mean([c['word_count'] for c in all_chars])
            avg_length = np.mean([c['length'] for c in all_chars])
            avg_clinical_terms = np.mean([c['clinical_term_count'] for c in all_chars])
            avg_complexity = np.mean([c['complexity_score'] for c in all_chars])

            pct_with_abbrev = (sum([c['has_abbreviations'] for c in all_chars]) / len(all_chars)) * 100
            pct_with_parens = (sum([c['has_parentheses'] for c in all_chars]) / len(all_chars)) * 100

            print(f"Average Word Count: {avg_word_count:.1f}")
            print(f"Average Length: {avg_length:.1f} characters")
            print(f"Average Clinical Terms: {avg_clinical_terms:.1f}")
            print(f"Average Complexity Score: {avg_complexity:.3f}")
            print(f"With Abbreviations: {pct_with_abbrev:.1f}%")
            print(f"With Parentheses: {pct_with_parens:.1f}%")

            summary_data.append({
                'Model': model,
                'Success_Rate_%': round(success_rate, 1),
                'Parsing_Errors': model_results['parsing_errors'],
                'Avg_Word_Count': round(avg_word_count, 1),
                'Avg_Length': round(avg_length, 1),
                'Avg_Clinical_Terms': round(avg_clinical_terms, 1),
                'Avg_Complexity': round(avg_complexity, 3),
                'Pct_With_Abbreviations': round(pct_with_abbrev, 1),
                'Pct_With_Parentheses': round(pct_with_parens, 1)
            })

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    print(f"\n=== SUMMARY TABLE ===")
    print(summary_df)

    # Ground truth analysis
    print("\n=== GROUND TRUTH PERFORMANCE ANALYSIS ===")

    # Check ground truth availability
    total_gt_available = df['ground_truth'].notna().sum()
    total_gt_with_text = df[df['ground_truth'].notna() & (df['ground_truth'].str.strip() != '')].shape[0] if total_gt_available > 0 else 0

    print(f"Ground Truth Availability: {total_gt_available}/{len(df)} records have non-null values")
    print(f"Ground Truth with Text: {total_gt_with_text}/{len(df)} records have actual text")

    # Analyze ground truth performance for each model
    gt_summary_data = []
    all_gt_matches = []

    for model in models:
        model_results = results[model]

        if model_results['successes'] > 0:
            # Get ground truth metrics from model details
            gt_metrics_list = [detail['ground_truth_metrics'] for detail in model_results['details']]

            # Filter out cases with no ground truth
            valid_gt_metrics = [m for m in gt_metrics_list if m['error'] != "No ground truth available" and m['error'] != "Missing text" and m['error'] != "Empty text"]

            print(f"\n--- {model} Ground Truth Analysis ---")
            print(f"Total Successful Indications: {model_results['successes']}")
            print(f"With Ground Truth for Comparison: {len(valid_gt_metrics)}")

            if len(valid_gt_metrics) > 0:
                # Calculate averages
                exact_matches = sum([1 for m in valid_gt_metrics if m['exact_match']])
                avg_text_similarity = np.mean([m['text_similarity'] for m in valid_gt_metrics])
                avg_bleu_score = np.mean([m['bleu_score'] for m in valid_gt_metrics])
                avg_cosine_similarity = np.mean([m['cosine_similarity'] for m in valid_gt_metrics])
                avg_length_ratio = np.mean([m['length_ratio'] for m in valid_gt_metrics])
                avg_word_overlap = np.mean([m['word_overlap'] for m in valid_gt_metrics])

                exact_match_rate = (exact_matches / len(valid_gt_metrics)) * 100

                print(f"Exact Matches: {exact_matches}/{len(valid_gt_metrics)} ({exact_match_rate:.1f}%)")
                print(f"Average Text Similarity: {avg_text_similarity:.3f}")
                print(f"Average BLEU Score: {avg_bleu_score:.3f}")
                print(f"Average Cosine Similarity: {avg_cosine_similarity:.3f}")
                print(f"Average Length Ratio: {avg_length_ratio:.3f}")
                print(f"Average Word Overlap: {avg_word_overlap:.3f}")

                gt_summary_data.append({
                    'Model': model,
                    'Total_Successful': model_results['successes'],
                    'GT_Comparisons': len(valid_gt_metrics),
                    'Exact_Matches': exact_matches,
                    'Exact_Match_Rate_%': round(exact_match_rate, 1),
                    'Avg_Text_Similarity': round(avg_text_similarity, 3),
                    'Avg_BLEU_Score': round(avg_bleu_score, 3),
                    'Avg_Cosine_Similarity': round(avg_cosine_similarity, 3),
                    'Avg_Length_Ratio': round(avg_length_ratio, 3),
                    'Avg_Word_Overlap': round(avg_word_overlap, 3)
                })

                # Collect all GT matches for detailed analysis
                for i, detail in enumerate(model_results['details']):
                    gt_metrics = detail['ground_truth_metrics']
                    if gt_metrics['error'] is None:  # Valid ground truth comparison
                        all_gt_matches.append({
                            'Model': model,
                            'Abstract_ID': detail['abstract_id'],
                            'Generated_Indication': detail['indication'],
                            'Ground_Truth': df[df['abstract_id'] == detail['abstract_id']]['ground_truth'].iloc[0],
                            'Text_Similarity': gt_metrics['text_similarity'],
                            'BLEU_Score': gt_metrics['bleu_score'],
                            'Cosine_Similarity': gt_metrics['cosine_similarity'],
                            'Exact_Match': gt_metrics['exact_match'],
                            'Word_Overlap': gt_metrics['word_overlap'],
                            'Length_Ratio': gt_metrics['length_ratio']
                        })
            else:
                print("No valid ground truth comparisons available")
                gt_summary_data.append({
                    'Model': model,
                    'Total_Successful': model_results['successes'],
                    'GT_Comparisons': 0,
                    'Exact_Matches': 0,
                    'Exact_Match_Rate_%': 0.0,
                    'Avg_Text_Similarity': 0.0,
                    'Avg_BLEU_Score': 0.0,
                    'Avg_Cosine_Similarity': 0.0,
                    'Avg_Length_Ratio': 0.0,
                    'Avg_Word_Overlap': 0.0
                })

    # Create ground truth summary DataFrame
    gt_summary_df = pd.DataFrame(gt_summary_data)
    if not gt_summary_df.empty:
        print(f"\n=== GROUND TRUTH SUMMARY TABLE ===")
        print(gt_summary_df)

    # Inter-model consensus analysis
    print("\n=== INTER-MODEL CONSENSUS ANALYSIS ===")

    if inter_model_comparisons:
        # Calculate overall consensus statistics
        avg_similarities = [comp['average_similarity'] for comp in inter_model_comparisons]
        consensus_levels = [comp['consensus_level'] for comp in inter_model_comparisons]

        overall_avg_similarity = np.mean(avg_similarities)

        print(f"Overall Average Similarity: {overall_avg_similarity:.3f}")
        print(f"Total Comparisons: {len(inter_model_comparisons)}")

        # Consensus level distribution
        consensus_counts = {}
        for level in consensus_levels:
            consensus_counts[level] = consensus_counts.get(level, 0) + 1

        print(f"\nConsensus Distribution:")
        for level, count in consensus_counts.items():
            pct = (count / len(consensus_levels)) * 100
            print(f"  {level}: {count} ({pct:.1f}%)")

    # Prepare consolidated results for export
    print("\n=== Preparing Export Data ===")

    # Create consolidated results
    consolidated_results = []

    for idx, record in df.iterrows():
        result_row = {
            'abstract_id': record['abstract_id'],
            'abstract_title': record['abstract_title'],
            'session_title': record.get('session_title', ''),
            'ground_truth': record.get('ground_truth', None)
        }

        # Add results for each model
        for model in models:
            response_col = f'{model}_indication_response'
            success_col = f'{model}_success'

            if record[success_col] in ['True', True]:
                parsed = parse_indication_response(record[response_col])

                if parsed['success']:
                    chars = analyze_indication_characteristics(parsed['indication'])
                    result_row.update({
                        f'{model}_success': True,
                        f'{model}_indication': parsed['indication'],
                        f'{model}_confidence': parsed['confidence_score'],
                        f'{model}_word_count': chars['word_count'],
                        f'{model}_length': chars['length'],
                        f'{model}_clinical_terms': chars['clinical_term_count'],
                        f'{model}_complexity': chars['complexity_score'],
                        f'{model}_has_abbreviations': chars['has_abbreviations'],
                        f'{model}_has_parentheses': chars['has_parentheses'],
                        f'{model}_error': None
                    })
                else:
                    result_row.update({
                        f'{model}_success': False,
                        f'{model}_indication': "",
                        f'{model}_confidence': None,
                        f'{model}_word_count': 0,
                        f'{model}_length': 0,
                        f'{model}_clinical_terms': 0,
                        f'{model}_complexity': 0,
                        f'{model}_has_abbreviations': False,
                        f'{model}_has_parentheses': False,
                        f'{model}_error': parsed['error']
                    })
            else:
                result_row.update({
                    f'{model}_success': False,
                    f'{model}_indication': "",
                    f'{model}_confidence': None,
                    f'{model}_word_count': 0,
                    f'{model}_length': 0,
                    f'{model}_clinical_terms': 0,
                    f'{model}_complexity': 0,
                    f'{model}_has_abbreviations': False,
                    f'{model}_has_parentheses': False,
                    f'{model}_error': "Model failed"
                })

        # Add inter-model comparison if available
        record_indications = {}
        for model in models:
            if result_row[f'{model}_success']:
                record_indications[model] = result_row[f'{model}_indication']

        if len(record_indications) > 1:
            comparison = compare_indications_between_models(record_indications)
            result_row['inter_model_avg_similarity'] = comparison['average_similarity']
            result_row['inter_model_consensus'] = comparison['consensus_level']

            # Add pairwise similarities
            for pair, sim in comparison['pairwise_similarities'].items():
                result_row[f'similarity_{pair}'] = sim
        else:
            result_row['inter_model_avg_similarity'] = None
            result_row['inter_model_consensus'] = 'No comparison'

        consolidated_results.append(result_row)

    consolidated_df = pd.DataFrame(consolidated_results)

    # Create visualizations
    create_visualizations(summary_df, gt_summary_df, inter_model_comparisons, args.output_dir)

    # Export results
    export_results(consolidated_df, summary_df, gt_summary_df, all_gt_matches,
                  inter_model_comparisons, df, models, args.output_dir)

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
