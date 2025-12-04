#!/usr/bin/env python3
"""
Drug Class LLM Extractor

This script reads cached Tavily search results and uses LLM to extract drug classes.
This allows experimenting with the prompt multiple times without consuming
additional Tavily credits.

Supports both old cache format (flat) and new optimized format (nested).
"""

import argparse
import concurrent.futures
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
        print(f"Error: Cache file not found at {cache_file}")
        return {}

    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            drug_count = len(data.get('drugs', {}))
            print(f"✓ Loaded cache with {drug_count} unique drugs")
            return data
    except Exception as e:
        print(f"Error loading cache: {e}")
        return {}


def get_entries_from_cache(cache_data: Dict) -> List[Tuple[str, List[str], List[Dict], List[Dict]]]:
    """Extract all drug+firm entries from cache (supports both old and new format).

    Args:
        cache_data: Cache dictionary

    Returns:
        List of tuples: (drug_name, firms, drug_class_results, firm_results)
    """
    entries = []
    drugs_cache = cache_data.get("drugs", {})

    # Check cache version/format
    # New format: drugs[drug_name]["drug_class_search"] exists
    # Old format: drugs[key]["drug_class_search_results"] exists (key is "drug|firms")

    for key, value in drugs_cache.items():
        # New optimized format
        if "drug_class_search" in value and "firm_searches" in value:
            drug_name = key
            drug_class_results = value.get("drug_class_search", {}).get("results", [])

            # Iterate through all firm combinations for this drug
            for firms_key, firm_data in value.get("firm_searches", {}).items():
                firms = firm_data.get("firms", [])
                firm_results = firm_data.get("results", [])
                entries.append((drug_name, firms, drug_class_results, firm_results))

        # Old flat format (backward compatibility)
        elif "drug_class_search_results" in value:
            drug_name = value.get("drug_name", key.split("|")[0] if "|" in key else key)
            firms = value.get("firm", [])
            if isinstance(firms, str):
                firms = [firms] if firms else []
            drug_class_results = value.get("drug_class_search_results", [])
            firm_results = value.get("firm_search_results", [])
            entries.append((drug_name, firms, drug_class_results, firm_results))

    return entries


def format_search_results_for_prompt(drug: str, drug_class_results: List[Dict], firm_results: List[Dict]) -> str:
    """Format search results according to the prompt's INPUT specification.

    Args:
        drug: The drug name
        drug_class_results: Results from drug class search
        firm_results: Results from firm search

    Returns:
        str: Formatted input string for the extraction prompt
    """
    all_results = drug_class_results + firm_results

    if not all_results:
        return f"Drug: {drug}\n\nNo search results available."

    formatted_parts = [f"Drug: {drug}"]

    for i, result in enumerate(all_results, 1):
        # Use raw_content if available, otherwise fall back to content
        content = result.get("raw_content") or result.get("content", "No content available")
        url = result.get("url", "Unknown URL")

        # Truncate very long content
        if len(content) > 5000:
            content = content[:5000] + "... [truncated]"

        formatted_parts.append(f"\nExtracted Content {i}: {content}")
        formatted_parts.append(f"Content {i} URL: {url}")

    return "\n".join(formatted_parts)


def extract_drug_class(
    drug_name: str,
    drug_class_results: List[Dict],
    firm_results: List[Dict],
    llm,
    system_prompt: str,
    langfuse_callback=None
) -> Dict[str, Any]:
    """Extract drug class using LLM.

    Args:
        drug_name: Name of the drug
        drug_class_results: Cached drug class search results
        firm_results: Cached firm search results
        llm: Initialized LLM
        system_prompt: System prompt for extraction
        langfuse_callback: Optional Langfuse callback handler

    Returns:
        Dictionary with extraction results
    """
    # Format input for prompt
    formatted_input = format_search_results_for_prompt(drug_name, drug_class_results, firm_results)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=formatted_input),
    ]

    try:
        # Invoke LLM
        if langfuse_callback:
            response: AIMessage = llm.invoke(messages, config={"callbacks": [langfuse_callback]})
        else:
            response: AIMessage = llm.invoke(messages)

        content = response.content

        # Parse JSON response
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
                    "raw_response": content,
                }
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parsing error for {drug_name}: {e}")

        # Return with raw response if parsing fails
        return {
            "drug_name": drug_name,
            "drug_classes": ["NA"],
            "content_urls": ["NA"],
            "steps_taken": [],
            "success": False,
            "raw_response": content,
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


def process_single_entry(
    entry: Tuple[str, List[str], List[Dict], List[Dict]],
    llm,
    system_prompt: str,
    index: int,
    langfuse_callback=None
) -> Dict:
    """Process a single drug+firm entry from cache.

    Args:
        entry: Tuple of (drug_name, firms, drug_class_results, firm_results)
        llm: Initialized LLM
        system_prompt: System prompt
        index: Index for logging
        langfuse_callback: Optional Langfuse callback

    Returns:
        Dictionary with processing result
    """
    drug_name, firms, drug_class_results, firm_results = entry

    print(f"[{index}] Processing: {drug_name}")

    # Extract drug class
    result = extract_drug_class(
        drug_name=drug_name,
        drug_class_results=drug_class_results,
        firm_results=firm_results,
        llm=llm,
        system_prompt=system_prompt,
        langfuse_callback=langfuse_callback,
    )

    # Build output row
    return {
        "drug_name": drug_name,
        "firm": json.dumps(firms),  # Store as JSON array
        "drug_classes": json.dumps(result.get("drug_classes", [])),
        "content_urls": json.dumps(result.get("content_urls", [])),
        "steps_taken": json.dumps(result.get("steps_taken", [])),
        "success": result.get("success", False),
        "raw_response": result.get("raw_response", ""),
        "error": result.get("error", ""),
    }


def main():
    """Main function for LLM extraction from cached data."""
    parser = argparse.ArgumentParser(description='Extract drug classes from cached search results')
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
                        help='Maximum entries to process (default: all)')
    parser.add_argument('--max_workers', type=int, default=3,
                        help='Parallel workers (default: 3)')
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
    print(f"Cache file: {args.cache_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Max entries: {args.max_entries or 'all'}")
    print(f"Max workers: {args.max_workers}")
    print(f"Langfuse: {'disabled' if args.disable_langfuse else 'enabled'}")
    print()

    # Load cache
    print("Loading cache...")
    cache_data = load_cache(args.cache_file)
    if not cache_data or not cache_data.get("drugs"):
        print("No cached data found. Run drug_class_search_fetcher.py first.")
        return

    # Get all entries from cache (supports both formats)
    entries = get_entries_from_cache(cache_data)
    print(f"Found {len(entries)} drug+firm entries in cache")

    if args.max_entries:
        entries = entries[:args.max_entries]

    print(f"Processing {len(entries)} entries")

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
        langfuse_client=None,  # Use local file for experimentation
        prompt_name="DRUG_CLASS_EXTRACTION_FROM_SEARCH",
        fallback_to_file=True,
    )
    print(f"✓ Prompt loaded (version: {prompt_version})")

    # Process entries
    print("\nExtracting drug classes...")
    print("-" * 60)

    results = []

    # Process sequentially for better logging (or use parallel if needed)
    if args.max_workers == 1:
        for i, entry in enumerate(entries, 1):
            result = process_single_entry(
                entry, llm, system_prompt, i, langfuse_callback
            )
            results.append(result)
    else:
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_entry,
                    entry,
                    llm,
                    system_prompt,
                    i,
                    langfuse_callback
                ): i
                for i, entry in enumerate(entries, 1)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    print(f"Error processing entry {idx}: {e}")

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
    print(f"  Total processed: {total}")
    print(f"  Successful: {int(successful)}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
