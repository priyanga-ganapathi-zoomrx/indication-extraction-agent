#!/usr/bin/env python3
"""
Drug Class LLM Extractor

This script reads an input CSV with drug/firm/abstract info, looks up cached
Tavily search results, and uses LLM to extract drug classes.

Features:
- Reads abstract_id, abstract_title, drug_name, firm, full_abstract from input CSV
- Looks up search results from cache (keyed by drug name)
- Handles multiple drugs per row (comma/semicolon separated)
- Groups results by drug with flattened drug_classes column
- Preserves all original input columns in output
"""

import argparse
import concurrent.futures
import csv
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.config import settings
from src.langfuse_config import get_langfuse_config
from src.llm_handler import LLMConfig, create_llm
from src.prompts import get_system_prompt


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


def load_input_csv(csv_path: str, max_entries: int = None) -> List[Dict]:
    """Load input CSV with drug/firm/abstract information.

    Handles multiple drugs in a single drug_name cell by creating separate entries.
    Each row maps to one or more processing entries, grouped by row_id.

    Args:
        csv_path: Path to input CSV
        max_entries: Maximum number of CSV rows to load

    Returns:
        List of dictionaries, each representing a processing entry
    """
    if not os.path.exists(csv_path):
        print(f"Error: Input file not found at {csv_path}")
        return []

    entries = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Detect column names (case-insensitive matching)
        fieldnames_lower = {name.lower(): name for name in reader.fieldnames} if reader.fieldnames else {}

        # Map expected columns
        drug_col = next((fieldnames_lower[k] for k in ['drug_name', 'drug'] if k in fieldnames_lower), None)
        firm_col = next((fieldnames_lower[k] for k in ['firm', 'company'] if k in fieldnames_lower), None)
        abstract_id_col = next((fieldnames_lower[k] for k in ['abstract_id', 'id'] if k in fieldnames_lower), None)
        abstract_title_col = next((fieldnames_lower[k] for k in ['abstract_title', 'title'] if k in fieldnames_lower), None)
        full_abstract_col = next((fieldnames_lower[k] for k in ['full_abstract', 'abstract'] if k in fieldnames_lower), None)
        ground_truth_col = next((fieldnames_lower[k] for k in ['drug class - ground truth (manually extracted)', 'ground_truth'] if k in fieldnames_lower), None)

        row_count = 0
        for row_id, row in enumerate(reader, start=1):
            if max_entries and row_count >= max_entries:
                break

            # Get raw values
            raw_drug_name = row.get(drug_col, '').strip() if drug_col else ''
            raw_firm = row.get(firm_col, '').strip() if firm_col else ''
            abstract_id = row.get(abstract_id_col, '').strip() if abstract_id_col else ''
            abstract_title = row.get(abstract_title_col, '').strip() if abstract_title_col else ''
            full_abstract = row.get(full_abstract_col, '').strip() if full_abstract_col else ''
            ground_truth = row.get(ground_truth_col, '').strip() if ground_truth_col else ''

            if not raw_drug_name:
                continue

            # Parse firms (comma or semicolon separated)
            firms = [f.strip() for f in raw_firm.replace(';', ',').split(',') if f.strip()]

            # Parse multiple drugs in a single cell (comma or semicolon separated)
            individual_drugs = [d.strip() for d in raw_drug_name.replace(';', ',').split(',') if d.strip()]

            # Store original row data for output
            original_row = {
                'abstract_id': abstract_id,
                'abstract_title': abstract_title,
                'drug_name': raw_drug_name,  # Original (may have multiple)
                'Drug Class - Ground truth (Manually extracted)': ground_truth,
                'firm': raw_firm,
                'full_abstract': full_abstract,
            }

            entries.append({
                'row_id': row_id,
                'original_row': original_row,
                'individual_drugs': individual_drugs,
                'firms': firms,
                'abstract_id': abstract_id,
                'abstract_title': abstract_title,
                'full_abstract': full_abstract,
            })

            row_count += 1

    print(f"✓ Loaded {len(entries)} rows from input CSV")
    return entries


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

    # Look up drug in cache
    drug_data = drugs_cache.get(drug_name, {})

    if not drug_data:
        return [], []

    # Get drug class search results (shared for all firms)
    drug_class_results = drug_data.get("drug_class_search", {}).get("results", [])

    # Get firm-specific results
    firms_key = get_firms_key(firms)
    firm_results = drug_data.get("firm_searches", {}).get(firms_key, {}).get("results", [])

    return drug_class_results, firm_results


def format_search_results_for_prompt(
    drug: str,
    drug_class_results: List[Dict],
    firm_results: List[Dict],
    abstract_title: str = "",
    full_abstract: str = ""
) -> str:
    """Format search results according to the prompt's INPUT specification.

    Args:
        drug: The drug name
        drug_class_results: Results from drug class search
        firm_results: Results from firm search
        abstract_title: Abstract title for context
        full_abstract: Full abstract text for context

    Returns:
        str: Formatted input string for the extraction prompt
    """
    all_results = drug_class_results + firm_results

    formatted_parts = [f"Drug: {drug}"]

    # Add abstract title if provided
    if abstract_title:
        formatted_parts.append(f"\nAbstract title: {abstract_title}")

    # Add full abstract if provided
    if full_abstract:
        abstract_text = full_abstract
        if len(abstract_text) > 10000:
            abstract_text = abstract_text[:10000] + "... [truncated]"
        formatted_parts.append(f"\nFull Abstract Text: {abstract_text}")

    # Add search results
    if not all_results:
        formatted_parts.append("\nNo search results available.")
    else:
        for i, result in enumerate(all_results, 1):
            content = result.get("raw_content") or result.get("content", "No content available")
            url = result.get("url", "Unknown URL")

            if len(content) > 5000:
                content = content[:5000] + "... [truncated]"

            formatted_parts.append(f"\nExtracted Content {i}: {content}")
            formatted_parts.append(f"Content {i} URL: {url}")

    return "\n".join(formatted_parts)


def extract_drug_class(
    drug_name: str,
    drug_class_results: List[Dict],
    firm_results: List[Dict],
    abstract_title: str,
    full_abstract: str,
    llm,
    system_prompt: str,
    langfuse_callback=None,
    abstract_id: str = "",
    prompt_version: str = ""
) -> Dict[str, Any]:
    """Extract drug class using LLM.

    Args:
        drug_name: Name of the drug
        drug_class_results: Cached drug class search results
        firm_results: Cached firm search results
        abstract_title: Abstract title for context
        full_abstract: Full abstract text for context
        llm: Initialized LLM
        system_prompt: System prompt for extraction
        langfuse_callback: Optional Langfuse callback handler
        abstract_id: Abstract ID for Langfuse tagging
        prompt_version: Prompt version for Langfuse tagging

    Returns:
        Dictionary with extraction results
    """
    formatted_input = format_search_results_for_prompt(
        drug_name, drug_class_results, firm_results, abstract_title, full_abstract
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=formatted_input),
    ]

    # Build tags for Langfuse tracing
    tags = [drug_name]
    if abstract_id:
        tags.append(f"abstract_id:{abstract_id}")
    if prompt_version:
        tags.append(f"prompt_version:{prompt_version}")

    try:
        if langfuse_callback:
            response: AIMessage = llm.invoke(
                messages,
                config={
                    "callbacks": [langfuse_callback],
                    "metadata": {"langfuse_tags": tags},
                }
            )
        else:
            response: AIMessage = llm.invoke(messages)

        content = response.content

        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "drug_name": drug_name,
                    "drug_classes": parsed.get("drug_classes", ["NA"]),
                    "content_urls": parsed.get("content_urls", ["NA"]),
                    "steps_taken": parsed.get("steps_taken", []),
                    "success": True,
                }
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parsing error for {drug_name}: {e}")

        return {
            "drug_name": drug_name,
            "drug_classes": ["NA"],
            "content_urls": ["NA"],
            "steps_taken": [],
            "success": False,
            "error": "JSON parsing failed",
        }

    except Exception as e:
        print(f"  ✗ Error extracting for {drug_name}: {e}")
        return {
            "drug_name": drug_name,
            "drug_classes": ["NA"],
            "content_urls": ["NA"],
            "steps_taken": [],
            "success": False,
            "error": str(e),
        }


def process_single_row(
    entry: Dict,
    cache_data: Dict,
    llm,
    system_prompt: str,
    index: int,
    langfuse_callback=None,
    prompt_version: str = ""
) -> Dict:
    """Process a single row from input CSV.

    Handles multiple drugs in a row by processing each individually and grouping results.

    Args:
        entry: Entry dict with row info and individual_drugs list
        cache_data: Cache dictionary
        llm: Initialized LLM
        system_prompt: System prompt
        index: Index for logging
        langfuse_callback: Optional Langfuse callback
        prompt_version: Prompt version for Langfuse tagging

    Returns:
        Dictionary with grouped results for the row
    """
    individual_drugs = entry['individual_drugs']
    firms = entry['firms']
    abstract_title = entry['abstract_title']
    full_abstract = entry['full_abstract']
    abstract_id = entry['abstract_id']
    original_row = entry['original_row']

    print(f"[{index}] Processing row with drugs: {individual_drugs}")

    # Process each drug individually - separate groupings for each field
    drug_classes_grouped = {}
    content_urls_grouped = {}
    steps_taken_grouped = {}
    all_drug_classes = []  # For flattened output
    success_flags = []

    for drug in individual_drugs:
        # Get search results from cache
        drug_class_results, firm_results = get_search_results_from_cache(
            cache_data, drug, firms
        )

        # Extract drug class
        result = extract_drug_class(
            drug_name=drug,
            drug_class_results=drug_class_results,
            firm_results=firm_results,
            abstract_title=abstract_title,
            full_abstract=full_abstract,
            llm=llm,
            system_prompt=system_prompt,
            langfuse_callback=langfuse_callback,
            abstract_id=abstract_id,
            prompt_version=prompt_version,
        )

        # Store grouped results separately
        drug_classes_grouped[drug] = result.get("drug_classes", ["NA"])
        content_urls_grouped[drug] = result.get("content_urls", ["NA"])
        steps_taken_grouped[drug] = result.get("steps_taken", [])
        success_flags.append(result.get("success", False))

        # Collect drug classes for flattened output (exclude "NA")
        drug_classes = result.get("drug_classes", [])
        for dc in drug_classes:
            if dc and dc != "NA" and dc not in all_drug_classes:
                all_drug_classes.append(dc)

    # If no valid drug classes found, use ["NA"]
    if not all_drug_classes:
        all_drug_classes = ["NA"]

    # Determine overall success
    overall_success = any(success_flags)

    # Build output row (preserve original columns + add new ones)
    output_row = original_row.copy()
    output_row.update({
        "drug_classes_grouped": json.dumps(drug_classes_grouped),  # Only drug classes grouped by drug
        "content_urls_grouped": json.dumps(content_urls_grouped),  # Content URLs grouped by drug
        "steps_taken_grouped": json.dumps(steps_taken_grouped),  # Steps taken grouped by drug
        "drug_classes": json.dumps(all_drug_classes),  # Flattened
        "success": overall_success,
    })

    return output_row


def main():
    """Main function for LLM extraction from cached data."""
    parser = argparse.ArgumentParser(description='Extract drug classes from cached search results')
    parser.add_argument('--input_file', default='data/drug_class_input_500.csv',
                        help='Input CSV file (default: data/drug_class_input_500.csv)')
    parser.add_argument('--cache_file', default='data/drug_search_cache.json',
                        help='Input JSON cache file (default: data/drug_search_cache.json)')
    parser.add_argument('--output_file', default=None,
                        help='Output CSV file (default: auto-generated)')
    parser.add_argument('--model', default='gpt-4.1',
                        help='LLM model to use (default: gpt-4.1)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='LLM temperature (default: 0.0)')
    parser.add_argument('--max_tokens', type=int, default=4096,
                        help='LLM max tokens (default: 4096)')
    parser.add_argument('--max_entries', type=int, default=None,
                        help='Maximum CSV rows to process (default: all)')
    parser.add_argument('--max_workers', type=int, default=1,
                        help='Parallel workers (default: 1)')
    parser.add_argument('--disable_langfuse', action='store_true',
                        help='Disable Langfuse tracing')

    args = parser.parse_args()

    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model.replace("/", "-")
        args.output_file = f"data/drug_class_extraction_{model_safe}_{timestamp}.csv"

    print("🧬 Drug Class LLM Extractor")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Cache file: {args.cache_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Max entries: {args.max_entries or 'all'}")
    print(f"Max workers: {args.max_workers}")
    print(f"Langfuse: {'disabled' if args.disable_langfuse else 'enabled'}")
    print()

    # Load input CSV
    print("Loading input CSV...")
    entries = load_input_csv(args.input_file, args.max_entries)
    if not entries:
        print("No entries found in input CSV.")
        return

    # Load cache
    print("Loading cache...")
    cache_data = load_cache(args.cache_file)

    # Initialize Langfuse
    langfuse_config = get_langfuse_config() if not args.disable_langfuse else None
    langfuse_callback = None

    if langfuse_config:
        try:
            langfuse = Langfuse(
                public_key=langfuse_config.public_key,
                secret_key=langfuse_config.secret_key,
                host=langfuse_config.host,
            )
            if langfuse.auth_check():
                print("✓ Langfuse initialized")
                os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_config.public_key
                os.environ["LANGFUSE_SECRET_KEY"] = langfuse_config.secret_key
                os.environ["LANGFUSE_HOST"] = langfuse_config.host
                langfuse_callback = CallbackHandler()
        except Exception as e:
            print(f"⚠ Langfuse init failed: {e}")

    # Initialize LLM
    print(f"\nInitializing LLM ({args.model})...")
    llm_config = LLMConfig(
        api_key=settings.llm.LLM_API_KEY,
        model=args.model,
        base_url=settings.llm.LLM_BASE_URL,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        name="DrugClassExtractor",
    )
    llm = create_llm(llm_config)
    print("✓ LLM initialized")

    # Load system prompt
    print("\nLoading prompt...")
    system_prompt, prompt_version = get_system_prompt(
        langfuse_client=None,
        prompt_name="DRUG_CLASS_EXTRACTION_FROM_SEARCH",
        fallback_to_file=True,
    )
    print(f"✓ Prompt loaded (version: {prompt_version})")

    # Process entries
    print(f"\nProcessing {len(entries)} rows...")
    print("-" * 60)

    results = []

    if args.max_workers == 1:
        # Sequential processing
        for i, entry in enumerate(entries, 1):
            result = process_single_row(
                entry, cache_data, llm, system_prompt, i, langfuse_callback, prompt_version
            )
            results.append(result)
    else:
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_row,
                    entry,
                    cache_data,
                    llm,
                    system_prompt,
                    i,
                    langfuse_callback,
                    prompt_version
                ): i
                for i, entry in enumerate(entries, 1)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")

        # Sort by index if parallel
        if results and isinstance(results[0], tuple):
            results.sort(key=lambda x: x[0])
            results = [r[1] for r in results]

    # Save results
    print(f"\n{'=' * 60}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False)

    # Summary
    total = len(results_df)
    successful = results_df['success'].sum() if 'success' in results_df.columns else 0
    success_rate = (successful / total * 100) if total > 0 else 0

    print("📊 Summary:")
    print(f"  Total rows processed: {total}")
    print(f"  Successful: {int(successful)}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
