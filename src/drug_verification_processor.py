#!/usr/bin/env python3
"""
Step 3: Drug Verification Processor

This script runs ONLY the verification step (Step 3) using validation results from Step 2.
It uses Tavily search to verify if each drug term is a valid drug/regimen.

Usage:
    python src/drug_verification_processor.py --input_file step2_validation_results.csv --output_file verification_results.csv
"""

import csv
import json
import os
import sys
import argparse
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langfuse.langchain import CallbackHandler

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.langfuse_config import get_langfuse_config
from src.llm_handler import LLMConfig, create_llm
from src.prompts import get_system_prompt


class DrugVerificationProcessor:
    """Processor for Step 3: Drug Verification only."""

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        max_parallel: int = 5,
    ):
        """Initialize the verification processor.

        Args:
            model: Model name (uses default from settings if not specified)
            temperature: Temperature (uses default from settings if not specified)
            max_tokens: Max tokens (uses default from settings if not specified)
            max_parallel: Maximum parallel Tavily queries
        """
        self.langfuse_config = get_langfuse_config()
        self.max_parallel = max_parallel
        
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
        """Verify a list of drugs.

        Args:
            drug_list: List of drug terms to verify
            abstract_id: Optional abstract ID for tracking

        Returns:
            dict: Verification results for all drugs
        """
        if not drug_list:
            return {"verification_results": {}, "verified_drugs": [], "removed_drugs": []}

        if not self.tavily_client:
            print(f"⚠ Tavily not available - skipping verification for {len(drug_list)} drugs")
            return {
                "verification_results": {},
                "verified_drugs": drug_list,
                "removed_drugs": [],
            }

        # Remove duplicates
        unique_drugs = list(set(drug_list))
        print(f"  🔍 Verifying {len(unique_drugs)} drug(s)...")

        # Search for all drugs in parallel
        search_results_map = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_drug = {
                executor.submit(self._search_drug_term, drug): drug
                for drug in unique_drugs
            }
            for future in as_completed(future_to_drug):
                result = future.result()
                search_results_map[result["drug_term"]] = result["results"]

        # Verify each drug with LLM
        verification_results = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_drug = {
                executor.submit(
                    self._verify_single_drug,
                    drug,
                    search_results_map.get(drug, [])
                ): drug
                for drug in unique_drugs
            }
            for future in as_completed(future_to_drug):
                result = future.result()
                verification_results[result["drug_term"]] = {
                    "is_drug": result["is_drug"],
                    "reason": result["reason"],
                }
                status = "✓" if result["is_drug"] else "✗"
                print(f"    {status} '{result['drug_term']}': is_drug={result['is_drug']}")

        # Separate verified and removed drugs
        verified_drugs = [d for d in drug_list if verification_results.get(d, {}).get("is_drug", True)]
        removed_drugs = [
            {"Drug": d, "Reason": verification_results[d].get("reason", "Unknown")}
            for d in drug_list if not verification_results.get(d, {}).get("is_drug", True)
        ]

        return {
            "verification_results": verification_results,
            "verified_drugs": verified_drugs,
            "removed_drugs": removed_drugs,
        }


def load_validation_results(csv_path: str, max_rows: int = None) -> List[Dict]:
    """Load validation results from Step 2 CSV."""
    results = []

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return results

    try:
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            # Parse drug lists from JSON strings
            primary_drugs = json.loads(row.get('validation_primary_drugs', '[]') or '[]')
            secondary_drugs = json.loads(row.get('validation_secondary_drugs', '[]') or '[]')
            comparator_drugs = json.loads(row.get('validation_comparator_drugs', '[]') or '[]')
            removed_drugs = json.loads(row.get('validation_removed_drugs', '[]') or '[]')
            reasoning = json.loads(row.get('validation_reasoning', '[]') or '[]')

            results.append({
                'abstract_id': str(row.get('abstract_id', '')),
                'session_title': str(row.get('session_title', '')),
                'abstract_title': str(row.get('abstract_title', '')),
                'ground_truth': str(row.get('ground_truth', '')),
                # Step 1 columns
                'extraction_response': str(row.get('extraction_response', '{}')),
                'extraction_primary_drugs': str(row.get('extraction_primary_drugs', '[]')),
                'extraction_secondary_drugs': str(row.get('extraction_secondary_drugs', '[]')),
                'extraction_comparator_drugs': str(row.get('extraction_comparator_drugs', '[]')),
                'extraction_reasoning': str(row.get('extraction_reasoning', '[]')),
                # Step 2 columns
                'validation_response': str(row.get('validation_response', '{}')),
                'primary_drugs': primary_drugs,
                'secondary_drugs': secondary_drugs,
                'comparator_drugs': comparator_drugs,
                'validation_removed_drugs': removed_drugs,
                'validation_reasoning': reasoning,
            })
            
            if max_rows and len(results) >= max_rows:
                break

        return results

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_single_row(row: Dict, processor: DrugVerificationProcessor, index: int) -> Dict:
    """Process a single row."""
    print(f"Processing row {index}: ID {row['abstract_id']}")

    # Combine all drugs for verification
    all_drugs = row['primary_drugs'] + row['secondary_drugs'] + row['comparator_drugs']
    
    # Verify drugs
    verification_result = processor.verify_drugs(all_drugs, row['abstract_id'])

    # Filter each category based on verification
    verified_results = verification_result['verification_results']
    
    verified_primary = [d for d in row['primary_drugs'] if verified_results.get(d, {}).get("is_drug", True)]
    verified_secondary = [d for d in row['secondary_drugs'] if verified_results.get(d, {}).get("is_drug", True)]
    verified_comparator = [d for d in row['comparator_drugs'] if verified_results.get(d, {}).get("is_drug", True)]

    return {
        'abstract_id': row['abstract_id'],
        'session_title': row['session_title'],
        'abstract_title': row['abstract_title'],
        'ground_truth': row['ground_truth'],
        # Step 1 columns
        'extraction_response': row['extraction_response'],
        'extraction_primary_drugs': row.get('extraction_primary_drugs', '[]'),
        'extraction_secondary_drugs': row.get('extraction_secondary_drugs', '[]'),
        'extraction_comparator_drugs': row.get('extraction_comparator_drugs', '[]'),
        'extraction_reasoning': row.get('extraction_reasoning', '[]'),
        # Step 2 columns
        'validation_response': row['validation_response'],
        'validation_primary_drugs': json.dumps(row['primary_drugs']),
        'validation_secondary_drugs': json.dumps(row['secondary_drugs']),
        'validation_comparator_drugs': json.dumps(row['comparator_drugs']),
        'validation_removed_drugs': json.dumps(row['validation_removed_drugs']),
        'validation_reasoning': json.dumps(row['validation_reasoning']),
        # Step 3 columns
        'verified_primary_drugs': json.dumps(verified_primary),
        'verified_secondary_drugs': json.dumps(verified_secondary),
        'verified_comparator_drugs': json.dumps(verified_comparator),
        'verification_removed_drugs': json.dumps(verification_result['removed_drugs']),
        'verification_results': json.dumps(verified_results),
    }


def main():
    parser = argparse.ArgumentParser(description='Step 3: Drug Verification Processor')
    parser.add_argument('--input_file', required=True, help='Input CSV file from Step 2 (validation results)')
    parser.add_argument('--output_file', default=None, help='Output CSV file (default: auto-generated)')
    parser.add_argument('--num_rows', type=int, default=None, help='Number of rows to process')
    parser.add_argument('--model', default=None, help='Model to use for verification LLM calls')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature for LLM')
    parser.add_argument('--max_tokens', type=int, default=None, help='Max tokens for LLM')
    parser.add_argument('--max_parallel', type=int, default=5, help='Maximum parallel Tavily queries')

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
    print()

    # Load validation results
    rows = load_validation_results(args.input_file, args.num_rows)
    if not rows:
        print("No validation results loaded. Exiting.")
        return

    print(f"Loaded {len(rows)} validation results")

    # Initialize processor
    processor = DrugVerificationProcessor(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_parallel=args.max_parallel,
    )

    # Process rows (sequential to manage Tavily rate limits)
    results = []
    for i, row in enumerate(rows, 1):
        try:
            result = process_single_row(row, processor, i)
            results.append(result)
        except Exception as e:
            print(f"Error processing row {i}: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)

    # Summary
    print()
    print("📊 Summary:")
    print(f"Total processed: {len(results)}")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

