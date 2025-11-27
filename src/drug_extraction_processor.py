#!/usr/bin/env python3
"""
Step 1: Drug Extraction Processor

This script runs ONLY the extraction step (Step 1) and saves results to CSV.
The output can be used as input for the validation processor (Step 2).

Usage:
    python src/drug_extraction_processor.py --input_file data/input.csv --output_file extraction_results.csv
"""

import csv
import json
import os
import sys
import argparse
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any

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


class DrugExtractionProcessor:
    """Processor for Step 1: Drug Extraction only."""

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        """Initialize the extraction processor.

        Args:
            model: Model name (uses default from settings if not specified)
            temperature: Temperature (uses default from settings if not specified)
            max_tokens: Max tokens (uses default from settings if not specified)
        """
        self.langfuse_config = get_langfuse_config()
        
        # Create LLM config
        self.llm_config = LLMConfig(
            api_key=settings.llm.LLM_API_KEY,
            model=model or settings.llm.LLM_MODEL,
            base_url=settings.llm.LLM_BASE_URL,
            temperature=temperature if temperature is not None else settings.llm.LLM_TEMPERATURE,
            max_tokens=max_tokens or settings.llm.LLM_MAX_TOKENS,
            name="DrugExtractionProcessor",
        )
        self.llm = create_llm(self.llm_config)

        # Load system prompt
        self.system_prompt, self.prompt_version = get_system_prompt(
            langfuse_client=None,
            prompt_name="DRUG_EXTRACTION_SYSTEM_PROMPT",
            fallback_to_file=True,
        )
        print(f"✓ Extraction processor initialized with model: {self.llm_config.model}")

    def extract(self, abstract_title: str, abstract_id: str = None) -> Dict[str, Any]:
        """Run extraction on a single abstract.

        Args:
            abstract_title: The abstract title to extract drugs from
            abstract_id: Optional abstract ID for tracking

        Returns:
            dict: Extraction result with response and metadata
        """
        input_content = f"Extract drugs from the following:\n\nabstract_title: {abstract_title}"

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=input_content),
        ]

        # Setup Langfuse if available
        callbacks = []
        if self.langfuse_config:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_config.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_config.secret_key
            os.environ["LANGFUSE_HOST"] = self.langfuse_config.host
            callbacks = [CallbackHandler()]

        config = RunnableConfig(
            callbacks=callbacks,
            metadata={"langfuse_tags": [f"abstract_id:{abstract_id or 'unknown'}", "step:extraction"]}
        )

        try:
            response: AIMessage = self.llm.invoke(messages, config=config)
            return {
                "success": True,
                "response": response.content,
                "error": None,
            }
        except Exception as e:
            print(f"✗ Error extracting drugs: {e}")
            return {
                "success": False,
                "response": '{"Primary Drugs": [], "Secondary Drugs": [], "Comparator Drugs": []}',
                "error": str(e),
            }


def load_abstracts_from_csv(csv_path: str, max_abstracts: int = None, randomize: bool = False) -> List[Dict]:
    """Load abstracts from CSV file, preserving ALL columns.
    
    Returns:
        List of dicts where each dict contains:
        - All original columns from the CSV (prefixed with 'input_')
        - Mapped columns for processing: abstract_id, abstract_title
    """
    abstracts = []

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return abstracts

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            
            if reader.fieldnames:
                header_map = {h.lower().strip(): h for h in reader.fieldnames}
                original_headers = list(reader.fieldnames)
            else:
                header_map = {}
                original_headers = []
            
            # Map standard column names (case-insensitive)
            id_col = header_map.get('abstract_id') or header_map.get('abstract id') or header_map.get('id')
            title_col = header_map.get('abstract_title') or header_map.get('abstract title') or header_map.get('title')
            
            for row in reader:
                # Start with all original columns (preserve everything)
                abstract_data = {}
                for header in original_headers:
                    # Skip empty column names (from trailing commas in CSV)
                    header_stripped = header.strip()
                    if not header_stripped:
                        continue
                    # Store original column with 'input_' prefix to avoid collision
                    col_key = f"input_{header_stripped}"
                    abstract_data[col_key] = row.get(header, '')
                
                # Add mapped columns for processing (these are used by the processor)
                abstract_data['abstract_id'] = row.get(id_col, '') if id_col else ''
                abstract_data['abstract_title'] = row.get(title_col, '') if title_col else ''
                
                if abstract_data['abstract_id'] or abstract_data['abstract_title']:
                    abstracts.append(abstract_data)

        if randomize and max_abstracts and len(abstracts) > max_abstracts:
            import random
            abstracts = random.sample(abstracts, max_abstracts)
        elif max_abstracts and len(abstracts) > max_abstracts:
            abstracts = abstracts[:max_abstracts]

        return abstracts

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def parse_extraction_response(response: str) -> Dict[str, Any]:
    """Parse the extraction response JSON into separate fields."""
    import re
    
    try:
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx+1]
            else:
                json_str = response

        parsed = json.loads(json_str)
        
        def get_list(data, keys):
            for key in keys:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return []

        return {
            'primary_drugs': get_list(parsed, ['Primary Drugs', 'primary_drugs']),
            'secondary_drugs': get_list(parsed, ['Secondary Drugs', 'secondary_drugs']),
            'comparator_drugs': get_list(parsed, ['Comparator Drugs', 'comparator_drugs']),
            'reasoning': get_list(parsed, ['Reasoning', 'reasoning']),
        }
    except Exception:
        return {
            'primary_drugs': [],
            'secondary_drugs': [],
            'comparator_drugs': [],
            'reasoning': [],
        }


def process_single_abstract(abstract: Dict, processor: DrugExtractionProcessor, index: int) -> Dict:
    """Process a single abstract, preserving all original input columns."""
    print(f"Processing abstract {index}: ID {abstract['abstract_id']}")

    result = processor.extract(
        abstract_title=abstract['abstract_title'],
        abstract_id=abstract['abstract_id']
    )

    # Parse the extraction response into separate columns
    parsed = parse_extraction_response(result['response'])

    # Start with all original input columns (those prefixed with 'input_')
    output = {}
    for key, value in abstract.items():
        if key.startswith('input_'):
            output[key] = value
    
    # Add extraction-specific columns
    output.update({
        'extraction_response': result['response'],
        'extraction_primary_drugs': json.dumps(parsed['primary_drugs']),
        'extraction_secondary_drugs': json.dumps(parsed['secondary_drugs']),
        'extraction_comparator_drugs': json.dumps(parsed['comparator_drugs']),
        'extraction_reasoning': json.dumps(parsed['reasoning']),
        'extraction_success': result['success'],
        'extraction_error': result['error'] or '',
    })
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Step 1: Drug Extraction Processor')
    parser.add_argument('--input_file', required=True, help='Input CSV file with abstracts')
    parser.add_argument('--output_file', default=None, help='Output CSV file (default: auto-generated)')
    parser.add_argument('--num_abstracts', type=int, default=None, help='Number of abstracts to process')
    parser.add_argument('--randomize', action='store_true', help='Randomize abstract selection')
    parser.add_argument('--model', default=None, help='Model to use for extraction')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature for LLM')
    parser.add_argument('--max_tokens', type=int, default=None, help='Max tokens for LLM')
    parser.add_argument('--parallel_workers', type=int, default=3, help='Number of parallel workers')

    args = parser.parse_args()

    # Generate output filename
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = (args.model or settings.llm.LLM_MODEL).replace("/", "_")
        args.output_file = f"step1_extraction_{model_name}_{timestamp}.csv"

    print("🔬 Step 1: Drug Extraction Processor")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model: {args.model or settings.llm.LLM_MODEL}")
    print(f"Number of abstracts: {args.num_abstracts or 'all'}")
    print()

    # Load abstracts
    abstracts = load_abstracts_from_csv(args.input_file, args.num_abstracts, args.randomize)
    if not abstracts:
        print("No abstracts loaded. Exiting.")
        return

    print(f"Loaded {len(abstracts)} abstracts")

    # Initialize processor
    processor = DrugExtractionProcessor(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Process abstracts with intermediate saves every 5 results
    results = []
    save_interval = 5
    last_saved_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        future_to_index = {
            executor.submit(process_single_abstract, abstract, processor, i): i
            for i, abstract in enumerate(abstracts, 1)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results.append((index, result))
                
                # Save intermediate results every 5 processed
                if len(results) - last_saved_count >= save_interval:
                    # Sort by original order before saving
                    sorted_results = sorted(results, key=lambda x: x[0])
                    sorted_data = [r for _, r in sorted_results]
                    df = pd.DataFrame(sorted_data)
                    df.to_csv(args.output_file, index=False)
                    last_saved_count = len(results)
                    print(f"💾 Intermediate save: {len(results)} results saved to {args.output_file}")
                    
            except Exception as e:
                print(f"Error processing abstract {index}: {e}")

    # Sort by original order
    results.sort(key=lambda x: x[0])
    results = [r for _, r in results]

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)

    # Summary
    successful = sum(1 for r in results if r['extraction_success'])
    print()
    print("📊 Summary:")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

