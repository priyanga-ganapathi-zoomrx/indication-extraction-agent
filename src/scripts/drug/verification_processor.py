#!/usr/bin/env python3
"""
Step 3: Drug Verification Processor

This script runs ONLY the verification step (Step 3) using extraction results from Step 1.
It uses Tavily search to verify if each drug term is a valid drug/regimen.

Usage:
    python src/scripts/drug/verification_processor.py --input_file step1_extraction_results.csv --output_file verification_results.csv
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langfuse.langchain import CallbackHandler

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.agents.core import settings, get_langfuse_config, LLMConfig, create_llm
from src.agents.drug.prompts import get_system_prompt


class DrugVerificationProcessor:
    """Processor for Step 3: Drug Verification only."""

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        max_parallel: int = 5,
        cache_file: str = "data/drug_verification_cache.json",
    ):
        """Initialize the verification processor.

        Args:
            model: Model name (uses default from settings if not specified)
            temperature: Temperature (uses default from settings if not specified)
            max_tokens: Max tokens (uses default from settings if not specified)
            max_parallel: Maximum parallel Tavily queries
            cache_file: Path to cache file for drug verification results
        """
        self.langfuse_config = get_langfuse_config()
        self.max_parallel = max_parallel
        self.cache_file = cache_file
        
        # Initialize verification cache
        self.verification_cache: Dict[str, Dict] = {}
        self._load_cache()
        
        # Create LLM config
        self.llm_config = LLMConfig(
            api_key=settings.llm.LLM_API_KEY,
            model=model or settings.llm.LLM_MODEL,
            base_url=settings.llm.LLM_BASE_URL,
            temperature=temperature if temperature is not None else settings.llm.LLM_TEMPERATURE,
            max_tokens=max_tokens or settings.llm.LLM_MAX_TOKENS,
            name="DrugVerificationProcessor",
        )
        self.llm = create_llm(self.llm_config)

        # Load system prompt
        self.system_prompt, self.prompt_version = get_system_prompt(
            langfuse_client=None,
            prompt_name="DRUG_VERIFICATION_SYSTEM_PROMPT",
            fallback_to_file=True,
        )

        # Initialize Tavily
        self.tavily_client = self._initialize_tavily()
        
        # Setup callbacks for Langfuse
        self.callbacks = []
        if self.langfuse_config:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_config.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_config.secret_key
            os.environ["LANGFUSE_HOST"] = self.langfuse_config.host
            self.callbacks = [CallbackHandler()]

        print(f"✓ Verification processor initialized with model: {self.llm_config.model}")
        print(f"✓ Cache: {self.cache_file} ({len(self.verification_cache)} drugs cached)")

    def _normalize_drug_name(self, drug_name: str) -> str:
        """Normalize drug name for cache key consistency."""
        return drug_name.strip().lower()

    def _load_cache(self):
        """Load verification cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.verification_cache = json.load(f)
                print(f"✓ Loaded {len(self.verification_cache)} cached drug verifications")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠ Could not load cache: {e}")
                self.verification_cache = {}

    def save_cache(self):
        """Save verification cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.verification_cache, f, indent=2)
        except IOError as e:
            print(f"⚠ Could not save cache: {e}")

    def _get_cached(self, drug_name: str) -> Dict | None:
        """Get cached verification result."""
        return self.verification_cache.get(self._normalize_drug_name(drug_name))

    def _set_cached(self, drug_name: str, result: Dict):
        """Cache a verification result."""
        self.verification_cache[self._normalize_drug_name(drug_name)] = result

    def _initialize_tavily(self):
        """Initialize Tavily client."""
        try:
            from tavily import TavilyClient
            
            api_key = settings.tavily.TAVILY_API_KEY
            if not api_key:
                print("⚠ Tavily API key not configured. Verification will fail.")
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

    def _search_drug_term(self, drug_term: str) -> Dict:
        """Search for a drug term using Tavily."""
        if not self.tavily_client:
            return {"drug_term": drug_term, "results": [], "error": "Tavily client not initialized"}

        try:
            query = f"Is {drug_term} a valid drug or drug regimen?"
            response = self.tavily_client.search(
                query=query,
                max_results=settings.tavily.TAVILY_MAX_RESULTS,
                search_depth="basic",
            )
            
            results = []
            for result in response.get("results", [])[:5]:
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("content", "")[:500],
                    "url": result.get("url", ""),
                })
            
            return {"drug_term": drug_term, "results": results, "error": None}
        except Exception as e:
            print(f"✗ Error searching for '{drug_term}': {e}")
            return {"drug_term": drug_term, "results": [], "error": str(e)}

    def _verify_single_drug(self, drug_term: str, search_results: list) -> Dict:
        """Verify a single drug term using LLM with search results."""
        formatted_results = ""
        for i, result in enumerate(search_results, 1):
            formatted_results += f"\n{i}. **{result.get('title', 'No title')}**\n"
            formatted_results += f"   {result.get('snippet', 'No content')}\n"

        if not formatted_results:
            formatted_results = "No search results available."

        verification_input = f"""**Drug Term:** {drug_term}

**Search Results:**
{formatted_results}

Based on the search results, determine if "{drug_term}" is a valid drug or drug regimen."""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=verification_input),
        ]

        config = RunnableConfig(
            callbacks=self.callbacks,
            metadata={"langfuse_tags": ["step:verification", f"drug:{drug_term}"]}
        )

        try:
            response: AIMessage = self.llm.invoke(messages, config=config)
            content = response.content

            # Parse JSON response
            try:
                import re
                json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    return {
                        "drug_term": drug_term,
                        "is_drug": parsed.get("is_drug", False),
                        "reason": parsed.get("reason", "Unable to determine"),
                        "search_results": search_results,
                    }
            except json.JSONDecodeError:
                pass

            return {
                "drug_term": drug_term,
                "is_drug": False,
                "reason": "Failed to parse verification response",
                "search_results": search_results,
            }
        except Exception as e:
            print(f"✗ Error verifying '{drug_term}': {e}")
            return {
                "drug_term": drug_term,
                "is_drug": False,
                "reason": f"Verification error: {str(e)}",
                "search_results": search_results,
            }

    def verify_drugs(self, drug_list: List[str], abstract_id: str = None) -> Dict[str, Any]:
        """Verify a list of drugs using cache-first approach.
        
        Preserves original drug format for Tavily searches and CSV output.

        Args:
            drug_list: List of drug terms to verify
            abstract_id: Optional abstract ID for tracking

        Returns:
            dict: Verification results for all drugs (keyed by ORIGINAL drug format)
        """
        if not drug_list:
            return {"verification_results": {}, "verified_drugs": [], "removed_drugs": []}

        # Deduplicate by normalized name, keeping FIRST original format for each
        # This avoids duplicate API calls for case variants (e.g., "Aspirin" vs "aspirin")
        normalized_to_original: Dict[str, str] = {}
        for drug in drug_list:
            normalized = self._normalize_drug_name(drug)
            if normalized not in normalized_to_original:
                normalized_to_original[normalized] = drug  # Keep first original format
        
        # Check cache and collect drugs to verify
        normalized_results: Dict[str, Dict] = {}  # Results keyed by normalized name
        drugs_to_verify: List[str] = []  # Original format drugs to verify
        
        for normalized, original in normalized_to_original.items():
            cached = self._get_cached(original)
            if cached:
                normalized_results[normalized] = cached
            else:
                drugs_to_verify.append(original)  # Use original format for Tavily
        
        cached_count = len(normalized_to_original) - len(drugs_to_verify)
        if cached_count > 0:
            print(f"  ✓ {cached_count} drug(s) from cache")
        
        # Verify uncached drugs (using ORIGINAL format for Tavily)
        if drugs_to_verify:
            if not self.tavily_client:
                print(f"  ⚠ Tavily not available - skipping {len(drugs_to_verify)} drugs")
                for drug in drugs_to_verify:
                    normalized = self._normalize_drug_name(drug)
                    normalized_results[normalized] = {"is_drug": True, "reason": "Not verified (Tavily unavailable)"}
            else:
                print(f"  🔍 Verifying {len(drugs_to_verify)} drug(s) via Tavily+LLM...")
                
                # Search using ORIGINAL drug format (max 5 concurrent)
                search_results_map = {}
                with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                    future_to_drug = {
                        executor.submit(self._search_drug_term, drug): drug  # Original format
                        for drug in drugs_to_verify
                    }
                    for future in as_completed(future_to_drug):
                        result = future.result()
                        search_results_map[result["drug_term"]] = result["results"]

                # Verify with LLM using ORIGINAL drug format (max 5 concurrent)
                with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                    future_to_drug = {
                        executor.submit(
                            self._verify_single_drug,
                            drug,  # Original format
                            search_results_map.get(drug, [])
                        ): drug
                        for drug in drugs_to_verify
                    }
                    for future in as_completed(future_to_drug):
                        result = future.result()
                        drug_term = result["drug_term"]  # Original format
                        normalized = self._normalize_drug_name(drug_term)
                        
                        entry = {
                            "is_drug": result["is_drug"],
                            "reason": result["reason"],
                            "original_searched": drug_term,  # Preserve original format
                        }
                        normalized_results[normalized] = entry
                        # Cache with normalized key
                        self._set_cached(drug_term, entry)
                        
                        status = "✓" if result["is_drug"] else "✗"
                        print(f"    {status} '{drug_term}': is_drug={result['is_drug']}")

        # Build verification_results keyed by ORIGINAL drug names from input
        # This preserves the exact format from the input for CSV output
        verification_results = {}
        for drug in drug_list:
            normalized = self._normalize_drug_name(drug)
            if normalized in normalized_results:
                verification_results[drug] = normalized_results[normalized]

        # Separate verified and removed drugs (using ORIGINAL names from input)
        verified_drugs = [d for d in drug_list if verification_results.get(d, {}).get("is_drug", True)]
        removed_drugs = [
            {"Drug": d, "Reason": verification_results.get(d, {}).get("reason", "Unknown")}
            for d in drug_list if not verification_results.get(d, {}).get("is_drug", True)
        ]

        return {
            "verification_results": verification_results,
            "verified_drugs": verified_drugs,
            "removed_drugs": removed_drugs,
        }


def load_extraction_results(csv_path: str, max_rows: int = None) -> tuple:
    """Load extraction results from CSV, preserving all columns.
    
    Returns:
        tuple: (list of row dicts with parsed drugs, list of column names)
    """
    results = []
    columns = []

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return results, columns

    try:
        df = pd.read_csv(csv_path)
        columns = list(df.columns)

        # Find abstract_id column (may be input_ID or similar)
        id_col = None
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if col_lower in ['abstract_id', 'input_id', 'input_abstract_id']:
                id_col = col
                break

        for _, row in df.iterrows():
            # Keep all original columns as a dict
            row_dict = row.to_dict()
            
            # Convert NaN to empty string for string columns
            for key, value in row_dict.items():
                if pd.isna(value):
                    row_dict[key] = ''
                else:
                    row_dict[key] = str(value) if not isinstance(value, (int, float, bool)) else value
            
            # Add mapped abstract_id for processing
            if id_col:
                row_dict['abstract_id'] = str(row.get(id_col, ''))
            
            # Parse drug lists from JSON strings (from extraction columns)
            try:
                primary_drugs = json.loads(row.get('extraction_primary_drugs', '[]') or '[]')
            except (json.JSONDecodeError, TypeError):
                primary_drugs = []
            try:
                secondary_drugs = json.loads(row.get('extraction_secondary_drugs', '[]') or '[]')
            except (json.JSONDecodeError, TypeError):
                secondary_drugs = []
            try:
                comparator_drugs = json.loads(row.get('extraction_comparator_drugs', '[]') or '[]')
            except (json.JSONDecodeError, TypeError):
                comparator_drugs = []
            
            # Add parsed lists for processing
            row_dict['_primary_drugs'] = primary_drugs
            row_dict['_secondary_drugs'] = secondary_drugs
            row_dict['_comparator_drugs'] = comparator_drugs

            results.append(row_dict)
            
            if max_rows and len(results) >= max_rows:
                break

        return results, columns

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def process_single_row(row: Dict, processor: DrugVerificationProcessor, index: int) -> Dict:
    """Process a single row, preserving all input columns and adding verification columns."""
    print(f"Processing row {index}: ID {row.get('abstract_id', 'unknown')}")

    # Get parsed drug lists (added by load function)
    primary_drugs = row.get('_primary_drugs', [])
    secondary_drugs = row.get('_secondary_drugs', [])
    comparator_drugs = row.get('_comparator_drugs', [])

    # Combine all drugs for verification
    all_drugs = primary_drugs + secondary_drugs + comparator_drugs
    
    # Verify drugs
    verification_result = processor.verify_drugs(all_drugs, str(row.get('abstract_id', '')))

    # Filter each category based on verification - preserve original categories
    verified_results = verification_result['verification_results']
    
    # Filter each category separately, keeping drugs that are verified as valid
    verified_primary = [d for d in primary_drugs if verified_results.get(d, {}).get("is_drug", True)]
    verified_secondary = [d for d in secondary_drugs if verified_results.get(d, {}).get("is_drug", True)]
    verified_comparator = [d for d in comparator_drugs if verified_results.get(d, {}).get("is_drug", True)]
    
    # Track removed drugs with their original category
    removed_drugs_with_category = []
    for d in primary_drugs:
        if not verified_results.get(d, {}).get("is_drug", True):
            removed_drugs_with_category.append({
                "Drug": d,
                "Category": "Primary",
                "Reason": verified_results.get(d, {}).get("reason", "Unknown")
            })
    for d in secondary_drugs:
        if not verified_results.get(d, {}).get("is_drug", True):
            removed_drugs_with_category.append({
                "Drug": d,
                "Category": "Secondary",
                "Reason": verified_results.get(d, {}).get("reason", "Unknown")
            })
    for d in comparator_drugs:
        if not verified_results.get(d, {}).get("is_drug", True):
            removed_drugs_with_category.append({
                "Drug": d,
                "Category": "Comparator",
                "Reason": verified_results.get(d, {}).get("reason", "Unknown")
            })

    # Start with all original columns (excluding internal parsed fields)
    result = {k: v for k, v in row.items() if not k.startswith('_')}
    
    # Add Step 3 verification columns
    result['verified_primary_drugs'] = json.dumps(verified_primary)
    result['verified_secondary_drugs'] = json.dumps(verified_secondary)
    result['verified_comparator_drugs'] = json.dumps(verified_comparator)
    result['verification_removed_drugs'] = json.dumps(removed_drugs_with_category)
    result['verification_results'] = json.dumps(verified_results)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Step 3: Drug Verification Processor')
    parser.add_argument('--input_file', default='step2.csv', help='Input CSV file from Step 1 (extraction results)')
    parser.add_argument('--output_file', default="step3.csv", help='Output CSV file (default: auto-generated)')
    parser.add_argument('--num_rows', type=int, default=None, help='Number of rows to process')
    parser.add_argument('--model', default='gpt-4.1', help='Model to use for verification LLM calls')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for LLM')
    parser.add_argument('--max_tokens', type=int, default=30000, help='Max tokens for LLM')
    parser.add_argument('--max_parallel', type=int, default=5, help='Maximum parallel Tavily queries')
    parser.add_argument('--cache_file', default='data/drug_verification_cache.json', help='Cache file path')

    args = parser.parse_args()

    # Generate output filename
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = (args.model or settings.llm.LLM_MODEL).replace("/", "_")
        args.output_file = f"step3_verification_{model_name}_{timestamp}.csv"

    print("🔬 Step 3: Drug Verification Processor")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model: {args.model or settings.llm.LLM_MODEL}")
    print(f"Number of rows: {args.num_rows or 'all'}")
    print(f"Max parallel queries: {args.max_parallel}")
    print(f"Cache file: {args.cache_file}")
    print()

    # Load extraction results (preserves all columns from input CSV)
    rows, input_columns = load_extraction_results(args.input_file, args.num_rows)
    if not rows:
        print("No extraction results loaded. Exiting.")
        return

    print(f"Loaded {len(rows)} rows with {len(input_columns)} columns")
    print(f"Input columns: {input_columns}")

    # Initialize processor
    processor = DrugVerificationProcessor(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_parallel=args.max_parallel,
        cache_file=args.cache_file,
    )

    # Process rows (sequential to manage Tavily rate limits) with intermediate saves
    results = []
    save_interval = 5
    last_saved_count = 0
    
    for i, row in enumerate(rows, 1):
        try:
            result = process_single_row(row, processor, i)
            results.append(result)
            
            # Save intermediate results every 5 processed
            if len(results) - last_saved_count >= save_interval:
                df = pd.DataFrame(results)
                df.to_csv(args.output_file, index=False)
                processor.save_cache()
                last_saved_count = len(results)
                print(f"💾 Saved: {len(results)} rows to CSV, {len(processor.verification_cache)} drugs cached")
                
        except Exception as e:
            print(f"Error processing row {i}: {e}")

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    processor.save_cache()

    # Summary
    print()
    print("📊 Summary:")
    print(f"Total rows processed: {len(results)}")
    print(f"Total drugs cached: {len(processor.verification_cache)}")
    print(f"Results saved to: {args.output_file}")
    print(f"Cache saved to: {args.cache_file}")


if __name__ == "__main__":
    main()

