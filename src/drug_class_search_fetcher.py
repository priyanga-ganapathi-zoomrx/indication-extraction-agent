#!/usr/bin/env python3
"""
Drug Class Search Fetcher

This script fetches Tavily search results for drugs and caches them to a JSON file.
This allows running searches once and then experimenting with LLM extraction
multiple times without consuming additional Tavily credits.

Cache Structure (optimized):
{
  "drugs": {
    "DrugA": {
      "drug_class_search": {
        "fetched_at": "...",
        "results": [...]  # Shared across all firms
      },
      "firm_searches": {
        '["Pfizer"]': {
          "fetched_at": "...",
          "results": [...]
        },
        '["Roche", "Genentech"]': {
          "fetched_at": "...",
          "results": [...]
        }
      }
    }
  }
}
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings


def initialize_tavily():
    """Initialize Tavily client.

    Returns:
        TavilyClient instance or None if initialization fails
    """
    try:
        from tavily import TavilyClient

        api_key = settings.tavily.TAVILY_API_KEY
        if not api_key:
            print("⚠ Tavily API key not configured.")
            return None

        client = TavilyClient(api_key=api_key)
        print("✓ Tavily client initialized")
        return client
    except ImportError:
        print("✗ tavily-python not installed. Run: pip install tavily-python")
        return None
    except Exception as e:
        print(f"✗ Error initializing Tavily client: {e}")
        return None


def load_drugs_from_csv(csv_path: str, max_entries: int = None) -> List[Dict]:
    """Load drugs from CSV file.

    Handles multiple drugs and firms per row by expanding into separate entries.
    For example, a row with "DrugA, DrugB" and "Firm1, Firm2" becomes:
    - Entry 1: drug_name="DrugA", firms=["Firm1", "Firm2"]
    - Entry 2: drug_name="DrugB", firms=["Firm1", "Firm2"]

    Args:
        csv_path: Path to the CSV file
        max_entries: Maximum number of entries to return (after expansion)

    Returns:
        List of dictionaries with drug data (one per drug, may have multiple entries per CSV row)
    """
    entries = []

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return entries

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)

            if reader.fieldnames:
                header_map = {h.lower().strip(): h for h in reader.fieldnames}
            else:
                header_map = {}

            drug_name_col = header_map.get('drug_name') or header_map.get('drug name') or header_map.get('drug')
            firm_col = header_map.get('firm') or header_map.get('company') or header_map.get('sponsor')

            if not drug_name_col:
                print(f"Warning: Could not find drug_name column in {csv_path}")
                return entries

            rows_processed = 0
            for row in reader:
                rows_processed += 1

                # Parse drug_name as list (comma-separated or semicolon-separated)
                drug_value = row.get(drug_name_col, '').strip()
                if drug_value:
                    # Split by comma or semicolon and strip whitespace
                    drug_names = [d.strip() for d in drug_value.replace(';', ',').split(',') if d.strip()]
                else:
                    drug_names = []

                # Parse firm as list (comma-separated or semicolon-separated)
                firm_value = row.get(firm_col, '').strip() if firm_col else ''
                if firm_value:
                    firms = [f.strip() for f in firm_value.replace(';', ',').split(',') if f.strip()]
                else:
                    firms = []

                # Create separate entry for each drug (same firms shared)
                for drug_name in drug_names:
                    entries.append({
                        'drug_name': drug_name,
                        'firm': firms,  # List of firms (shared across drugs from same row)
                    })

            print(f"  Processed {rows_processed} CSV rows -> {len(entries)} drug entries")

        if max_entries and len(entries) > max_entries:
            entries = entries[:max_entries]

        return entries

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def search_drug_class(tavily_client, drug: str) -> List[Dict]:
    """Search for drug class information.

    Query: ("{{DRUG}}" AND ("Mechanism of Action" OR ...))
    Config: search_depth="advanced", include_domains=["nih.gov", "fda.gov", "clinicaltrials.gov"],
            include_raw_content=True, max_results=3

    Args:
        tavily_client: Initialized Tavily client
        drug: Drug name

    Returns:
        List of search results
    """
    query = (
        f'("{drug}" AND '
        f'("Mechanism of Action" OR "12.1 Mechanism of Action" OR "MoA" OR "mode of action" OR '
        f'"Pharmacologic Class" OR "Pharmacological Class" OR "Chemical Class" OR '
        f'"Therapeutic Class" OR "Drug Class"))'
    )

    try:
        print(f"  🔍 Search 1: Drug class info for {drug}")
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            include_domains=["nih.gov", "fda.gov", "clinicaltrials.gov"],
            include_raw_content=True,
            max_results=3,
        )

        results = []
        for result in response.get("results", [])[:3]:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "raw_content": result.get("raw_content", ""),
            })

        print(f"     ✓ Found {len(results)} results")
        return results

    except Exception as e:
        print(f"     ✗ Error: {e}")
        return []


def search_drug_firm(tavily_client, drug: str, firms: List[str]) -> List[Dict]:
    """Search for drug + firm information.

    Query format:
    - No firms: {drug}
    - Single firm: ({drug} AND {firm})
    - Multiple firms: ({drug} AND ({firm1} OR {firm2} OR ...))

    Config: search_depth="advanced", include_raw_content=True, max_results=3

    Args:
        tavily_client: Initialized Tavily client
        drug: Drug name
        firms: List of firm/company names

    Returns:
        List of search results
    """
    # Build the search query based on number of firms
    if not firms or len(firms) == 0:
        query = drug
    elif len(firms) == 1:
        query = f'({drug} AND {firms[0]})'
    else:
        # Multiple firms: ({drug} AND ({firm1} OR {firm2} OR ...))
        firms_or = " OR ".join(firms)
        query = f'({drug} AND ({firms_or}))'

    try:
        print(f"  🔍 Search 2: {query}")
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            include_raw_content=True,
            max_results=3,
        )

        results = []
        for result in response.get("results", [])[:3]:
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "raw_content": result.get("raw_content", ""),
            })

        print(f"     ✓ Found {len(results)} results")
        return results

    except Exception as e:
        print(f"     ✗ Error: {e}")
        return []


def load_existing_cache(cache_file: str) -> Dict:
    """Load existing cache file if it exists.

    Args:
        cache_file: Path to cache file

    Returns:
        Dictionary with cached data or empty structure
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                drug_count = len(data.get('drugs', {}))
                print(f"✓ Loaded existing cache with {drug_count} unique drugs")
                return data
        except Exception as e:
            print(f"⚠ Could not load existing cache: {e}")

    return {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "2.0",  # New optimized structure
        },
        "drugs": {}
    }


def save_cache(cache_data: Dict, cache_file: str):
    """Save cache data to file.

    Args:
        cache_data: Cache dictionary
        cache_file: Path to cache file
    """
    cache_data["metadata"]["updated_at"] = datetime.now().isoformat()

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def get_firms_key(firms: List[str]) -> str:
    """Get a consistent key for firms list.

    Args:
        firms: List of firm names

    Returns:
        JSON string of sorted firms for use as cache key
    """
    return json.dumps(sorted(firms))


def main():
    """Main function to fetch and cache drug searches."""
    parser = argparse.ArgumentParser(description='Fetch and cache Tavily searches for drugs')
    parser.add_argument('--input_file', default='data/drug_class_input_500.csv',
                        help='Input CSV file with drugs')
    parser.add_argument('--cache_file', default='data/drug_search_cache.json',
                        help='Output JSON cache file (default: data/drug_search_cache.json)')
    parser.add_argument('--max_entries', type=int, default=None,
                        help='Max entries to process after expansion (default: all)')
    parser.add_argument('--force_refresh', action='store_true',
                        help='Force re-fetch all searches')
    parser.add_argument('--force_refresh_firm', action='store_true',
                        help='Force re-fetch only firm searches (keep drug class searches)')

    args = parser.parse_args()

    print("🔎 Drug Class Search Fetcher (Optimized)")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Cache file: {args.cache_file}")
    print(f"Max entries: {args.max_entries or 'all'}")
    print(f"Force refresh all: {args.force_refresh}")
    print(f"Force refresh firm only: {args.force_refresh_firm}")
    print()

    # Initialize Tavily
    tavily_client = initialize_tavily()
    if not tavily_client:
        print("Cannot proceed without Tavily client.")
        return

    # Load drugs from CSV (expands rows with multiple drugs into separate entries)
    print("\nLoading drugs from CSV...")
    entries = load_drugs_from_csv(args.input_file, max_entries=args.max_entries)
    if not entries:
        print("No drug entries loaded. Exiting.")
        return
    print(f"Total entries to process: {len(entries)}")

    # Load existing cache
    print("\nLoading cache...")
    cache_data = load_existing_cache(args.cache_file)

    # Process each entry
    print("\nFetching search results...")
    print("-" * 60)

    drug_class_searches_done = 0
    drug_class_searches_skipped = 0
    firm_searches_done = 0
    firm_searches_skipped = 0
    credits_used = 0

    for i, entry in enumerate(entries, 1):
        drug_name = entry['drug_name']
        firms = entry['firm']  # List of firms
        firms_key = get_firms_key(firms)

        firms_display = ', '.join(firms) if firms else 'no firm'
        print(f"\n[{i}/{len(entries)}] {drug_name} ({firms_display})")

        # Initialize drug entry if not exists
        if drug_name not in cache_data["drugs"]:
            cache_data["drugs"][drug_name] = {
                "drug_class_search": None,
                "firm_searches": {}
            }

        drug_cache = cache_data["drugs"][drug_name]

        # Search 1: Drug class search (shared per drug)
        if drug_cache.get("drug_class_search") and not args.force_refresh:
            print("  ⏭ Drug class search already cached")
            drug_class_searches_skipped += 1
        else:
            drug_class_results = search_drug_class(tavily_client, drug_name)
            drug_cache["drug_class_search"] = {
                "fetched_at": datetime.now().isoformat(),
                "results": drug_class_results,
            }
            drug_class_searches_done += 1
            credits_used += 2

        # Search 2: Firm search (per drug+firm combo)
        if firms_key in drug_cache.get("firm_searches", {}) and not args.force_refresh and not args.force_refresh_firm:
            print("  ⏭ Firm search already cached")
            firm_searches_skipped += 1
        else:
            firm_results = search_drug_firm(tavily_client, drug_name, firms)
            if "firm_searches" not in drug_cache:
                drug_cache["firm_searches"] = {}
            drug_cache["firm_searches"][firms_key] = {
                "fetched_at": datetime.now().isoformat(),
                "firms": firms,  # Store original firms list for reference
                "results": firm_results,
            }
            firm_searches_done += 1
            credits_used += 2

        # Save cache after each entry (in case of interruption)
        save_cache(cache_data, args.cache_file)

    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"  Drug class searches: {drug_class_searches_done} done, {drug_class_searches_skipped} skipped (cached)")
    print(f"  Firm searches: {firm_searches_done} done, {firm_searches_skipped} skipped (cached)")
    print(f"  Tavily credits used: ~{credits_used}")
    print(f"  Unique drugs in cache: {len(cache_data['drugs'])}")
    print(f"  Cache file: {args.cache_file}")


if __name__ == "__main__":
    main()
