#!/usr/bin/env python3
"""
Step 2: Drug Validation Processor

This script runs ONLY the validation step (Step 2) using extraction results from Step 1.
The output can be used as input for the verification processor (Step 3).

Usage:
    python src/drug_validation_processor.py --input_file step1_extraction_results.csv --output_file validation_results.csv
"""

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
        enable_caching: bool = False,
    ):
        """Initialize the validation processor.

        Args:
            model: Model name (uses default from settings if not specified)
            temperature: Temperature (uses default from settings if not specified)
            max_tokens: Max tokens (uses default from settings if not specified)
            enable_caching: Enable Anthropic prompt caching for reduced costs
        """
        self.enable_caching = enable_caching
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

        # Load combined prompt file (contains both validation instructions and extraction rules)
        full_prompt, self.prompt_version = get_system_prompt(
            langfuse_client=None,
            prompt_name="DRUG_VALIDATION_SYSTEM_PROMPT",
            fallback_to_file=True,
        )
        
        # Parse message sections from the combined prompt
        self.system_prompt, self.extraction_rules = self._parse_message_sections(full_prompt)
        print(f"✓ Validation processor initialized with model: {self.llm_config.model}")
        if self.enable_caching:
            print("✓ Prompt caching enabled for DrugValidationProcessor")

    def _parse_message_sections(self, full_prompt: str) -> tuple:
        """Parse MESSAGE_1 (validation instructions) and MESSAGE_2 (extraction rules) from combined prompt.
        
        Args:
            full_prompt: The combined prompt content with message markers
            
        Returns:
            tuple: (validation_instructions, extraction_rules)
        """
        import re
        
        # Extract MESSAGE_1: Validation Instructions
        msg1_pattern = r'<!-- MESSAGE_1_START: VALIDATION_INSTRUCTIONS -->(.*?)<!-- MESSAGE_1_END: VALIDATION_INSTRUCTIONS -->'
        msg1_match = re.search(msg1_pattern, full_prompt, re.DOTALL)
        validation_instructions = msg1_match.group(1).strip() if msg1_match else full_prompt
        
        # Extract MESSAGE_2: Extraction Rules
        msg2_pattern = r'<!-- MESSAGE_2_START: EXTRACTION_RULES -->(.*?)<!-- MESSAGE_2_END: EXTRACTION_RULES -->'
        msg2_match = re.search(msg2_pattern, full_prompt, re.DOTALL)
        extraction_rules = msg2_match.group(1).strip() if msg2_match else ""
        
        if not msg1_match:
            print("⚠ Warning: MESSAGE_1 markers not found, using full prompt as validation instructions")
        if not msg2_match:
            print("⚠ Warning: MESSAGE_2 markers not found, extraction rules may be empty")
            
        return validation_instructions, extraction_rules

    def validate(self, abstract_title: str, extraction_response: str, abstract_id: str = None) -> Dict[str, Any]:
        """Run validation on extraction results using 3-message pattern.

        Args:
            abstract_title: The original abstract title
            extraction_response: The JSON response from Step 1 (extraction)
            abstract_id: Optional abstract ID for tracking

        Returns:
            dict: Validation result with response and metadata
        """
        # 3-message pattern (both extracted from DRUG_VALIDATION_SYSTEM_PROMPT.md):
        # Message 1: System Instruction (MESSAGE_1 section)
        # Message 2: Reference Rules - flat-numbered extraction rules (MESSAGE_2 section)
        # Message 3: Extraction Result to Validate
        
        validation_input = f"""Validate the extracted drugs for the following:

**Abstract Title:** {abstract_title}

**Extraction Result:**
{extraction_response}"""

        # Build system message with optional caching
        if self.enable_caching:
            system_msg = SystemMessage(content=[
                {"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}
            ])
        else:
            system_msg = SystemMessage(content=self.system_prompt)

        # Build reference rules message with optional caching
        reference_rules_content = f"# REFERENCE RULES DOCUMENT\n\nThe following are the extraction rules that the extractor was instructed to follow:\n\n{self.extraction_rules}"
        if self.enable_caching:
            reference_rules_msg = HumanMessage(content=[
                {"type": "text", "text": reference_rules_content, "cache_control": {"type": "ephemeral"}}
            ])
        else:
            reference_rules_msg = HumanMessage(content=reference_rules_content)

        messages = [
            system_msg,
            reference_rules_msg,
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
            
            # Log cache performance metrics if caching is enabled
            if self.enable_caching and hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                if usage:
                    input_token_details = usage.get("input_token_details", {})
                    cache_creation = input_token_details.get("cache_creation", 0)
                    cache_read = input_token_details.get("cache_read", 0)
                    if cache_creation > 0 or cache_read > 0:
                        print(f"  📦 Cache stats - creation: {cache_creation}, read: {cache_read}")
            
            return {
                "success": True,
                "response": response.content,
                "error": None,
            }
        except Exception as e:
            print(f"✗ Error validating drugs: {e}")
            return {
                "success": False,
                "response": '{"validation_status": "FAIL", "validation_confidence": 0.0, "missed_drugs": [], "grounded_search_performed": false, "search_results": [], "issues_found": [], "checks_performed": {}, "validation_reasoning": "Error during validation"}',
                "error": str(e),
            }


def parse_validation_response(response: str) -> Dict[str, Any]:
    """Parse the validation response JSON with new format.
    
    New format includes:
    - validation_status: PASS | REVIEW | FAIL
    - validation_confidence: 0.0 to 1.0
    - missed_drugs: Array of drugs that should have been extracted
    - grounded_search_performed: Boolean
    - search_results: Array of search results
    - issues_found: Array of issues with check_type, severity, etc.
    - checks_performed: Status of each validation check
    - validation_reasoning: Step-by-step reasoning
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
        
        def get_value(data, keys, default=None):
            for key in keys:
                if key in data:
                    return data[key]
            return default
        
        def get_list(data, keys):
            for key in keys:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return []

        return {
            'validation_status': get_value(parsed, ['validation_status'], 'REVIEW'),
            'validation_confidence': get_value(parsed, ['validation_confidence'], 0.0),
            'missed_drugs': get_list(parsed, ['missed_drugs']),
            'grounded_search_performed': get_value(parsed, ['grounded_search_performed'], False),
            'search_results': get_list(parsed, ['search_results']),
            'issues_found': get_list(parsed, ['issues_found']),
            'checks_performed': get_value(parsed, ['checks_performed'], {}),
            'validation_reasoning': get_value(parsed, ['validation_reasoning'], ''),
        }
    except Exception as e:
        return {
            'validation_status': 'FAIL',
            'validation_confidence': 0.0,
            'missed_drugs': [],
            'grounded_search_performed': False,
            'search_results': [],
            'issues_found': [],
            'checks_performed': {},
            'validation_reasoning': f'Error parsing response: {str(e)}',
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
    
    # Add Step 2 validation columns (new format) with pretty-printed JSON
    output.update({
        'validation_response': result['response'],
        'validation_status': parsed['validation_status'],
        'validation_confidence': parsed['validation_confidence'],
        'validation_missed_drugs': json.dumps(parsed['missed_drugs'], indent=2, ensure_ascii=False),
        'validation_grounded_search_performed': parsed['grounded_search_performed'],
        'validation_search_results': json.dumps(parsed['search_results'], indent=2, ensure_ascii=False),
        'validation_issues_found': json.dumps(parsed['issues_found'], indent=2, ensure_ascii=False),
        'validation_checks_performed': json.dumps(parsed['checks_performed'], indent=2, ensure_ascii=False),
        'validation_reasoning': parsed['validation_reasoning'],
        'validation_success': result['success'],
        'validation_error': result['error'] or '',
    })
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Step 2: Drug Validation Processor')
    parser.add_argument('--input_file', default='step1_filtered.csv', help='Input CSV file from Step 1 (extraction results)')
    parser.add_argument('--output_file', default='step2_filtered.csv', help='Output CSV file (default: auto-generated)')
    parser.add_argument('--num_rows', type=int, default=None, help='Number of rows to process')
    parser.add_argument('--model', default='gemini/gemini-3-flash-preview', help='Model to use for validation')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for LLM')
    parser.add_argument('--max_tokens', type=int, default=50000, help='Max tokens for LLM')
    parser.add_argument('--parallel_workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--enable_caching', action='store_true', help='Enable Anthropic prompt caching for reduced costs')

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
        enable_caching=args.enable_caching,
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

