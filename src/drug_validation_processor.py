#!/usr/bin/env python3
"""
Step 2: Drug Validation Processor

This script runs ONLY the validation step (Step 2) using extraction results from Step 1.
The output can be used as input for the verification processor (Step 3).

Usage:
    python src/drug_validation_processor.py --input_file step1_extraction_results.csv --output_file validation_results.csv
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


class DrugValidationProcessor:
    """Processor for Step 2: Drug Validation only."""

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        """Initialize the validation processor.

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
            name="DrugValidationProcessor",
        )
        self.llm = create_llm(self.llm_config)

        # Load system prompt
        self.system_prompt, self.prompt_version = get_system_prompt(
            langfuse_client=None,
            prompt_name="DRUG_VALIDATION_SYSTEM_PROMPT",
            fallback_to_file=True,
        )
        print(f"✓ Validation processor initialized with model: {self.llm_config.model}")

    def validate(self, abstract_title: str, extraction_response: str, abstract_id: str = None) -> Dict[str, Any]:
        """Run validation on extraction results.

        Args:
            abstract_title: The original abstract title
            extraction_response: The JSON response from Step 1 (extraction)
            abstract_id: Optional abstract ID for tracking

        Returns:
            dict: Validation result with response and metadata
        """
        validation_input = f"""Validate the extracted drugs for the following:

**Title:** {abstract_title}

**Extracted JSON:**
{extraction_response}"""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=validation_input),
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
            metadata={"langfuse_tags": [f"abstract_id:{abstract_id or 'unknown'}", "step:validation"]}
        )

        try:
            response: AIMessage = self.llm.invoke(messages, config=config)
            return {
                "success": True,
                "response": response.content,
                "error": None,
            }
        except Exception as e:
            print(f"✗ Error validating drugs: {e}")
            return {
                "success": False,
                "response": '{"Primary Drugs": [], "Secondary Drugs": [], "Comparator Drugs": [], "Flagged Drugs": [], "Potential Valid Drugs": [], "Non-Therapeutic Drugs": [], "Reasoning": []}',
                "error": str(e),
            }


def parse_validation_response(response: str) -> Dict[str, Any]:
    """Parse the validation response JSON with new format.
    
    New format includes:
    - Primary/Secondary/Comparator Drugs (unchanged from input)
    - Flagged Drugs (items flagged for review)
    - Potential Valid Drugs (missed therapeutic drugs)
    - Non-Therapeutic Drugs (drugs judged non-therapeutic)
    - Reasoning
    """
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
            'flagged_drugs': get_list(parsed, ['Flagged Drugs', 'flagged_drugs']),
            'potential_valid_drugs': get_list(parsed, ['Potential Valid Drugs', 'potential_valid_drugs']),
            'non_therapeutic_drugs': get_list(parsed, ['Non-Therapeutic Drugs', 'non_therapeutic_drugs']),
            'reasoning': get_list(parsed, ['Reasoning', 'reasoning']),
        }
    except Exception as e:
        return {
            'primary_drugs': [],
            'secondary_drugs': [],
            'comparator_drugs': [],
            'flagged_drugs': [],
            'potential_valid_drugs': [],
            'non_therapeutic_drugs': [],
            'reasoning': [],
        }


def load_extraction_results(csv_path: str, max_rows: int = None) -> List[Dict]:
    """Load extraction results from Step 1 CSV, preserving ALL columns."""
    results = []

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return results

    try:
        df = pd.read_csv(csv_path)
        
        # Find the extraction response column
        extraction_col = None
        for col in df.columns:
            if 'extraction_response' in col.lower():
                extraction_col = col
                break
        
        if not extraction_col:
            print("Error: Could not find extraction_response column in CSV")
            return results

        # Find abstract_title column (may be input_Abstract Title or similar)
        title_col = None
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if 'abstract_title' in col_lower or col_lower == 'input_abstract_title':
                title_col = col
                break
        
        # Find abstract_id column
        id_col = None
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if col_lower in ['abstract_id', 'input_id', 'input_abstract_id']:
                id_col = col
                break

        for _, row in df.iterrows():
            # Preserve ALL columns from the input
            row_data = {}
            for col in df.columns:
                val = row.get(col)
                # Handle NaN values
                if pd.isna(val):
                    row_data[col] = ''
                else:
                    row_data[col] = str(val)
            
            # Add mapped columns for processing (for backward compatibility)
            row_data['abstract_id'] = str(row.get(id_col, '')) if id_col else ''
            row_data['abstract_title'] = str(row.get(title_col, '')) if title_col else ''
            row_data['extraction_response'] = str(row.get(extraction_col, '{}'))
            
            results.append(row_data)
            
            if max_rows and len(results) >= max_rows:
                break

        return results

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def process_single_row(row: Dict, processor: DrugValidationProcessor, index: int) -> Dict:
    """Process a single row, preserving all original columns."""
    print(f"Processing row {index}: ID {row.get('abstract_id', 'unknown')}")

    result = processor.validate(
        abstract_title=row.get('abstract_title', ''),
        extraction_response=row.get('extraction_response', '{}'),
        abstract_id=row.get('abstract_id', '')
    )

    # Parse the validation response (new format)
    parsed = parse_validation_response(result['response'])

    # Start with all columns from the input row (preserves input_ columns and Step 1 columns)
    output = {}
    for key, value in row.items():
        # Skip the mapped columns we added for processing
        if key not in ['abstract_id', 'abstract_title']:
            output[key] = value
    
    # Add Step 2 validation columns
    output.update({
        'validation_response': result['response'],
        'validation_primary_drugs': json.dumps(parsed['primary_drugs']),
        'validation_secondary_drugs': json.dumps(parsed['secondary_drugs']),
        'validation_comparator_drugs': json.dumps(parsed['comparator_drugs']),
        'validation_flagged_drugs': json.dumps(parsed['flagged_drugs']),
        'validation_potential_valid_drugs': json.dumps(parsed['potential_valid_drugs']),
        'validation_non_therapeutic_drugs': json.dumps(parsed['non_therapeutic_drugs']),
        'validation_reasoning': json.dumps(parsed['reasoning']),
        'validation_success': result['success'],
        'validation_error': result['error'] or '',
    })
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Step 2: Drug Validation Processor')
    parser.add_argument('--input_file', default='step1.csv', help='Input CSV file from Step 1 (extraction results)')
    parser.add_argument('--output_file', default='step2.csv', help='Output CSV file (default: auto-generated)')
    parser.add_argument('--num_rows', type=int, default=None, help='Number of rows to process')
    parser.add_argument('--model', default='gemini/gemini-3-pro-preview', help='Model to use for validation')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for LLM')
    parser.add_argument('--max_tokens', type=int, default=50000, help='Max tokens for LLM')
    parser.add_argument('--parallel_workers', type=int, default=3, help='Number of parallel workers')

    args = parser.parse_args()

    # Generate output filename
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = (args.model or settings.llm.LLM_MODEL).replace("/", "_")
        args.output_file = f"step2_validation_{model_name}_{timestamp}.csv"

    print("✅ Step 2: Drug Validation Processor")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model: {args.model or settings.llm.LLM_MODEL}")
    print(f"Number of rows: {args.num_rows or 'all'}")
    print()

    # Load extraction results
    rows = load_extraction_results(args.input_file, args.num_rows)
    if not rows:
        print("No extraction results loaded. Exiting.")
        return

    print(f"Loaded {len(rows)} extraction results")

    # Initialize processor
    processor = DrugValidationProcessor(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Process rows with intermediate saves every 5 results
    results = []
    save_interval = 5
    last_saved_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
        future_to_index = {
            executor.submit(process_single_row, row, processor, i): i
            for i, row in enumerate(rows, 1)
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
                print(f"Error processing row {index}: {e}")

    # Sort by original order
    results.sort(key=lambda x: x[0])
    results = [r for _, r in results]

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)

    # Summary
    successful = sum(1 for r in results if r['validation_success'])
    print()
    print("📊 Summary:")
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

