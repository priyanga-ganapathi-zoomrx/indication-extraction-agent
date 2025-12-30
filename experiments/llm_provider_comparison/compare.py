#!/usr/bin/env python3
"""
LLM Provider Comparison Script

Compare Gemini model performance between LangChain and LiteLLM.

Metrics compared:
- Response time (ms)
- Token consumption (prompt, completion, total)
- Success rate

Usage:
    python experiments/llm_provider_comparison/compare.py
    python experiments/llm_provider_comparison/compare.py --model "gemini/gemini-3-flash-preview" --runs 3
    python experiments/llm_provider_comparison/compare.py --drug "Pembrolizumab" --context "Phase 3 melanoma study"
"""

import argparse
import csv
import os
import sys
from dataclasses import asdict
from datetime import datetime
from statistics import mean, stdev
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.llm_provider_comparison.config import (
    DEFAULT_ENABLE_LANGFUSE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
)
from experiments.llm_provider_comparison.prompts import TEST_INPUTS
from experiments.llm_provider_comparison.langchain_runner import run_langchain, RunResult
from experiments.llm_provider_comparison.litellm_runner import run_litellm


def run_comparison(
    drug: str,
    context: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    runs: int = 1,
    enable_langfuse: bool = DEFAULT_ENABLE_LANGFUSE,
) -> Dict[str, List[RunResult]]:
    """Run comparison between LangChain and LiteLLM for a single drug.
    
    Args:
        drug: Drug name to classify
        context: Clinical context
        model: Model name to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        runs: Number of runs per provider
        enable_langfuse: Enable Langfuse tracing (default: True if configured)
        
    Returns:
        Dictionary with 'langchain' and 'litellm' keys containing list of results
    """
    results = {
        "langchain": [],
        "litellm": [],
    }
    
    for run_num in range(runs):
        print(f"\n  Run {run_num + 1}/{runs}")
        
        # LangChain
        print("    Running LangChain...", end=" ", flush=True)
        lc_result = run_langchain(drug, context, model, temperature, max_tokens, enable_langfuse)
        results["langchain"].append(lc_result)
        if lc_result.success:
            print(f"✓ {lc_result.response_time_ms:.0f}ms (prompt:{lc_result.prompt_tokens}, completion:{lc_result.completion_tokens}, total:{lc_result.total_tokens})")
        else:
            print(f"✗ Error: {lc_result.error}")
        
        # LiteLLM
        print("    Running LiteLLM...", end=" ", flush=True)
        ll_result = run_litellm(drug, context, model, temperature, max_tokens, enable_langfuse)
        results["litellm"].append(ll_result)
        if ll_result.success:
            print(f"✓ {ll_result.response_time_ms:.0f}ms (prompt:{ll_result.prompt_tokens}, completion:{ll_result.completion_tokens}, total:{ll_result.total_tokens})")
        else:
            print(f"✗ Error: {ll_result.error}")
    
    return results


def calculate_stats(results: List[RunResult]) -> Dict[str, float]:
    """Calculate statistics for a list of results.
    
    Args:
        results: List of RunResult objects
        
    Returns:
        Dictionary with statistics
    """
    successful = [r for r in results if r.success]
    
    if not successful:
        return {
            "success_rate": 0.0,
            "avg_response_time_ms": 0.0,
            "std_response_time_ms": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
            "avg_total_tokens": 0.0,
        }
    
    response_times = [r.response_time_ms for r in successful]
    
    return {
        "success_rate": len(successful) / len(results) * 100,
        "avg_response_time_ms": mean(response_times),
        "std_response_time_ms": stdev(response_times) if len(response_times) > 1 else 0.0,
        "avg_prompt_tokens": mean([r.prompt_tokens for r in successful]),
        "avg_completion_tokens": mean([r.completion_tokens for r in successful]),
        "avg_total_tokens": mean([r.total_tokens for r in successful]),
    }


def print_comparison_table(
    drug: str,
    langchain_stats: Dict[str, float],
    litellm_stats: Dict[str, float],
) -> None:
    """Print a formatted comparison table.
    
    Args:
        drug: Drug name
        langchain_stats: Statistics for LangChain
        litellm_stats: Statistics for LiteLLM
    """
    print("\n" + "=" * 70)
    print(f"  Comparison Results for: {drug}")
    print("=" * 70)
    print(f"{'Metric':<30} {'LangChain':>18} {'LiteLLM':>18}")
    print("-" * 70)
    
    metrics = [
        ("Success Rate (%)", "success_rate", ".1f"),
        ("Avg Response Time (ms)", "avg_response_time_ms", ".0f"),
        ("Std Response Time (ms)", "std_response_time_ms", ".0f"),
        ("Avg Prompt Tokens", "avg_prompt_tokens", ".0f"),
        ("Avg Completion Tokens", "avg_completion_tokens", ".0f"),
        ("Avg Total Tokens", "avg_total_tokens", ".0f"),
    ]
    
    for label, key, fmt in metrics:
        lc_val = langchain_stats[key]
        ll_val = litellm_stats[key]
        
        # Calculate difference
        if lc_val > 0:
            diff = ((ll_val - lc_val) / lc_val) * 100
            diff_str = f"({diff:+.1f}%)" if key != "success_rate" else ""
        else:
            diff_str = ""
        
        print(f"{label:<30} {lc_val:>{18}{fmt}} {ll_val:>{12}{fmt}} {diff_str}")
    
    print("=" * 70)
    
    # Determine winner
    if langchain_stats["avg_response_time_ms"] < litellm_stats["avg_response_time_ms"]:
        time_winner = "LangChain"
        time_diff = litellm_stats["avg_response_time_ms"] - langchain_stats["avg_response_time_ms"]
    else:
        time_winner = "LiteLLM"
        time_diff = langchain_stats["avg_response_time_ms"] - litellm_stats["avg_response_time_ms"]
    
    print(f"\n⚡ Faster: {time_winner} (by {time_diff:.0f}ms)")


def save_results_to_csv(
    all_results: List[Dict],
    output_dir: str,
) -> str:
    """Save all results to a CSV file.
    
    Args:
        all_results: List of result dictionaries
        output_dir: Directory to save the CSV file
        
    Returns:
        Path to the saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"comparison_{timestamp}.csv")
    
    fieldnames = [
        "drug", "context", "provider", "run_number",
        "success", "error", "response_time_ms",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "model", "langfuse_enabled", "response_content",
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    return output_file


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description='Compare LLM Providers')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'Model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs per provider per drug (default: 3)')
    parser.add_argument('--drug', type=str, default=None,
                        help='Single drug to test (optional)')
    parser.add_argument('--context', type=str, default=None,
                        help='Context for single drug test')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help=f'Temperature (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help=f'Max tokens (default: {DEFAULT_MAX_TOKENS})')
    parser.add_argument('--langfuse', dest='enable_langfuse', action='store_true',
                        default=DEFAULT_ENABLE_LANGFUSE,
                        help='Enable Langfuse tracing (default: enabled if configured)')
    parser.add_argument('--no-langfuse', dest='enable_langfuse', action='store_false',
                        help='Disable Langfuse tracing')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  LLM Provider Comparison: LangChain vs LiteLLM")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Runs per provider: {args.runs}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Langfuse tracing: {'enabled' if args.enable_langfuse else 'disabled'}")
    
    # Determine test inputs
    if args.drug:
        test_inputs = [{
            "drug": args.drug,
            "context": args.context or "Clinical study context"
        }]
    else:
        test_inputs = TEST_INPUTS
    
    print(f"Test drugs: {len(test_inputs)}")
    print()
    
    all_csv_rows = []
    all_langchain_results = []
    all_litellm_results = []
    
    for input_data in test_inputs:
        drug = input_data["drug"]
        context = input_data["context"]
        
        print(f"\n📊 Testing: {drug}")
        print(f"   Context: {context}")
        
        results = run_comparison(
            drug=drug,
            context=context,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            runs=args.runs,
            enable_langfuse=args.enable_langfuse,
        )
        
        # Collect results
        all_langchain_results.extend(results["langchain"])
        all_litellm_results.extend(results["litellm"])
        
        # Prepare CSV rows
        for run_num, lc_result in enumerate(results["langchain"], 1):
            all_csv_rows.append({
                "drug": drug,
                "context": context,
                "provider": "langchain",
                "run_number": run_num,
                "success": lc_result.success,
                "error": lc_result.error or "",
                "response_time_ms": lc_result.response_time_ms,
                "prompt_tokens": lc_result.prompt_tokens,
                "completion_tokens": lc_result.completion_tokens,
                "total_tokens": lc_result.total_tokens,
                "model": lc_result.model,
                "langfuse_enabled": lc_result.langfuse_enabled,
                "response_content": lc_result.content,
            })
        
        for run_num, ll_result in enumerate(results["litellm"], 1):
            all_csv_rows.append({
                "drug": drug,
                "context": context,
                "provider": "litellm",
                "run_number": run_num,
                "success": ll_result.success,
                "error": ll_result.error or "",
                "response_time_ms": ll_result.response_time_ms,
                "prompt_tokens": ll_result.prompt_tokens,
                "completion_tokens": ll_result.completion_tokens,
                "total_tokens": ll_result.total_tokens,
                "model": ll_result.model,
                "langfuse_enabled": ll_result.langfuse_enabled,
                "response_content": ll_result.content,
            })
        
        # Calculate and print stats for this drug
        lc_stats = calculate_stats(results["langchain"])
        ll_stats = calculate_stats(results["litellm"])
        print_comparison_table(drug, lc_stats, ll_stats)
    
    # Overall summary
    if len(test_inputs) > 1:
        print("\n" + "=" * 70)
        print("  OVERALL SUMMARY")
        print("=" * 70)
        
        overall_lc_stats = calculate_stats(all_langchain_results)
        overall_ll_stats = calculate_stats(all_litellm_results)
        print_comparison_table("All Drugs", overall_lc_stats, overall_ll_stats)
    
    # Save results to CSV
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results"
    )
    csv_file = save_results_to_csv(all_csv_rows, output_dir)
    print(f"\n📁 Results saved to: {csv_file}")


if __name__ == "__main__":
    main()

