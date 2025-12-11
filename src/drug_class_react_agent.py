"""Drug Class Extraction Agent using 3-Message Structure for Reasoning Models.

This module implements a drug class extraction agent optimized for Gemini reasoning models.
It uses a 3-message structure:
1. System Prompt - Role, task, workflow, output format
2. Rules Message - All 40 extraction rules organized by application sequence
3. Input Message - Drug info, abstract, and search results

Key changes from the previous ReAct pattern:
- No tool calling - all rules provided upfront in message 2
- Simplified graph with single LLM call
- Optimized for reasoning models that perform internal reasoning
"""

import json
import os
import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.config import settings
from src.langfuse_config import get_langfuse_config
from src.llm_handler import LLMConfig, create_llm
from src.prompts import get_system_prompt


class DrugClassReActAgent:
    """Drug Class Extraction Agent using 3-Message Structure.

    This agent uses a simplified architecture optimized for reasoning models:
    - Parses prompt file into 3 separate message sections
    - Sends system prompt, rules, and input as 3 messages
    - No tool calling - LLM has all rules upfront for reasoning
    """

    def __init__(
        self,
        agent_name: str = "DrugClassReActAgent",
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        """Initialize the Drug Class Extraction Agent.

        Args:
            agent_name: Name of the agent for identification and logging
            model: LLM model to use (default: from settings)
            temperature: LLM temperature (default: from settings)
            max_tokens: LLM max tokens (default: from settings)
        """
        self.agent_name = agent_name
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Initialize Langfuse
        self.langfuse_config = get_langfuse_config()
        self.langfuse = self._initialize_langfuse() if self.langfuse_config else None

        # Initialize LLM (no tool binding for reasoning models)
        self.llm_config = self._get_llm_config()
        self.llm = create_llm(self.llm_config)

        # Load and parse the 3-message prompt structure
        self.system_prompt, self.rules_message, self.input_template = self._load_prompt_sections()

    def _initialize_langfuse(self) -> Langfuse | None:
        """Initialize Langfuse client for tracing and observability.

        Returns:
            Langfuse client instance or None if initialization fails
        """
        if not self.langfuse_config:
            print(f"ℹ Langfuse not configured for {self.agent_name}")
            return None

        try:
            langfuse = Langfuse(
                public_key=self.langfuse_config.public_key,
                secret_key=self.langfuse_config.secret_key,
                host=self.langfuse_config.host,
            )
            if langfuse.auth_check():
                print(f"✓ Langfuse initialized successfully for {self.agent_name}")
                return langfuse
            else:
                print(f"✗ Langfuse authentication failed for {self.agent_name}")
                return None
        except Exception as e:
            print(f"✗ Error initializing Langfuse for {self.agent_name}: {e}")
            return None

    def _get_llm_config(self) -> LLMConfig:
        """Get LLM configuration from settings or overrides.

        Returns:
            LLMConfig: Configuration for the language model
        """
        return LLMConfig(
            api_key=settings.llm.LLM_API_KEY,
            model=self._model or settings.llm.LLM_MODEL,
            base_url=settings.llm.LLM_BASE_URL,
            temperature=self._temperature if self._temperature is not None else settings.llm.LLM_TEMPERATURE,
            max_tokens=self._max_tokens or settings.llm.LLM_MAX_TOKENS,
            name=self.agent_name,
        )

    def _load_prompt_sections(self) -> tuple[str, str, str]:
        """Load and parse the 3-message prompt structure from file.

        The prompt file contains sections marked with HTML-style comments:
        - <!-- MESSAGE_1_START: SYSTEM_PROMPT --> ... <!-- MESSAGE_1_END: SYSTEM_PROMPT -->
        - <!-- MESSAGE_2_START: RULES_MESSAGE --> ... <!-- MESSAGE_2_END: RULES_MESSAGE -->
        - <!-- MESSAGE_3_START: INPUT_TEMPLATE --> ... <!-- MESSAGE_3_END: INPUT_TEMPLATE -->

        Returns:
            Tuple of (system_prompt, rules_message, input_template)
        """
        # Fetch the full prompt content
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name="DRUG_CLASS_EXTRACTION_FROM_SEARCH_REACT_PATTERN",
            fallback_to_file=True,
        )
        self.prompt_version = prompt_version

        # Extract sections using regex
        def extract_section(content: str, section_name: str) -> str:
            """Extract content between MESSAGE_X_START and MESSAGE_X_END markers."""
            pattern = rf'<!-- MESSAGE_\d+_START: {section_name} -->\s*(.*?)\s*<!-- MESSAGE_\d+_END: {section_name} -->'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                # Clean up the extracted content
                section = match.group(1).strip()
                # Remove the markdown header if present (e.g., "## SYSTEM_PROMPT")
                section = re.sub(rf'^##\s*{section_name}\s*\n+', '', section)
                return section
            return ""

        system_prompt = extract_section(prompt_content, "SYSTEM_PROMPT")
        rules_message = extract_section(prompt_content, "RULES_MESSAGE")
        input_template = extract_section(prompt_content, "INPUT_TEMPLATE")

        if not system_prompt:
            print("⚠ Warning: Could not extract SYSTEM_PROMPT section from prompt file")
        if not rules_message:
            print("⚠ Warning: Could not extract RULES_MESSAGE section from prompt file")
        if not input_template:
            print("⚠ Warning: Could not extract INPUT_TEMPLATE section from prompt file")

        return system_prompt, rules_message, input_template

    def _format_search_results(self, drug_class_results: List[Dict], firm_results: List[Dict]) -> str:
        """Format search results for the input message.

        Args:
            drug_class_results: Results from drug class search (cached)
            firm_results: Results from firm search (cached)

        Returns:
            str: Formatted search results string
        """
        all_results = (drug_class_results or []) + (firm_results or [])

        if not all_results:
            return "No search results available."

        formatted_parts = []
        for i, result in enumerate(all_results, 1):
            content = result.get("raw_content") or result.get("content", "No content available")
            url = result.get("url", "Unknown URL")

            if len(content) > 5000:
                content = content[:5000] + "... [truncated]"

            formatted_parts.append(f"### Result {i}")
            formatted_parts.append(f"**URL**: {url}")
            formatted_parts.append(f"**Content**: {content}")
            formatted_parts.append("")

        return "\n".join(formatted_parts)

    def _build_input_message(
        self,
        drug: str,
        abstract_title: str,
        full_abstract: str,
        drug_class_results: List[Dict],
        firm_results: List[Dict],
    ) -> str:
        """Build the input message (Message 3) from template.

        Args:
            drug: The drug name
            abstract_title: Abstract title
            full_abstract: Full abstract text
            drug_class_results: Drug class search results
            firm_results: Firm search results

        Returns:
            str: Formatted input message
        """
        # Truncate abstract if too long
        abstract_text = full_abstract or ""
        if len(abstract_text) > 10000:
            abstract_text = abstract_text[:10000] + "... [truncated]"

        # Format search results
        search_results = self._format_search_results(drug_class_results, firm_results)

        # Build the input message using template placeholders
        input_message = f"""# EXTRACTION INPUT

## Drug
{drug}

## Abstract Title
{abstract_title or "Not provided"}

## Full Abstract Text
{abstract_text or "Not provided"}

## Search Results

{search_results}"""

        return input_message

    def invoke(
        self,
        drug: str,
        abstract_title: str = "",
        full_abstract: str = "",
        drug_class_results: List[Dict] = None,
        firm_results: List[Dict] = None,
        abstract_id: str = None
    ) -> dict:
        """Invoke the drug class extraction agent with 3-message structure.

        Args:
            drug: The drug name to extract class for
            abstract_title: The abstract title for context (optional)
            full_abstract: The full abstract text for context (optional)
            drug_class_results: Pre-fetched drug class search results (optional)
            firm_results: Pre-fetched firm search results (optional)
            abstract_id: The abstract ID for tracking in Langfuse (optional)

        Returns:
            dict: Result containing messages and metadata
        """
        # Build tags for Langfuse tracing
        tags = [
            drug,
            f"prompt_version:{getattr(self, 'prompt_version', 'unknown')}",
            f"model:{self.llm_config.model}",
        ]
        if abstract_id:
            tags.append(f"abstract_id:{abstract_id}")

        # Build the 3 messages
        input_message = self._build_input_message(
            drug=drug,
            abstract_title=abstract_title,
            full_abstract=full_abstract,
            drug_class_results=drug_class_results or [],
            firm_results=firm_results or [],
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.rules_message),
            HumanMessage(content=input_message),
        ]

        # Set environment variables for Langfuse callback handler
        if self.langfuse:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_config.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_config.secret_key
            os.environ["LANGFUSE_HOST"] = self.langfuse_config.host

        # Configure with Langfuse tracing
        config = RunnableConfig(
            callbacks=(
                [CallbackHandler()]
                if self.langfuse
                else []
            ),
            metadata={"langfuse_tags": tags} if self.langfuse else {},
        )

        # Invoke the LLM
        try:
            response: AIMessage = self.llm.invoke(messages, config)
            return {
                "messages": messages + [response],
                "llm_calls": 1,
            }
        except Exception as e:
            print(f"✗ Error during LLM call: {e}")
            return {
                "messages": messages + [
                    AIMessage(content=f"I encountered an error during drug class extraction: {str(e)}. Please try again.")
                ],
                "llm_calls": 1,
            }

    def parse_response(self, result: dict) -> Dict[str, Any]:
        """Parse the agent's response to extract structured drug class data.

        Args:
            result: Agent invocation result containing messages

        Returns:
            Dictionary with extracted fields matching the new output format
        """
        try:
            messages = result.get('messages', [])
            if not messages:
                return self._default_response()

            final_message = messages[-1]
            content = getattr(final_message, 'content', '')

            if not content or content.startswith("I encountered an error"):
                return self._default_response(error=content)

            # Try to parse JSON response
            # Look for JSON in code block first
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    return self._extract_fields(parsed)
                except json.JSONDecodeError:
                    pass

            # Try to find raw JSON object
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return self._extract_fields(parsed)
                except json.JSONDecodeError:
                    pass

            # Failed to parse
            return self._default_response(error="Failed to parse JSON response")

        except Exception as e:
            print(f"Error parsing response: {e}")
            return self._default_response(error=str(e))

    def _extract_fields(self, parsed: dict) -> Dict[str, Any]:
        """Extract fields from parsed JSON response.

        Updated to match the new output format without tool-related fields.

        Args:
            parsed: Parsed JSON dictionary

        Returns:
            Dictionary with normalized fields
        """
        quality_metrics = parsed.get('quality_metrics', {})

        return {
            'drug_name': parsed.get('drug_name', ''),
            'drug_classes': parsed.get('drug_classes', ['NA']),
            'selected_sources': parsed.get('selected_sources', []),
            'confidence_score': parsed.get('confidence_score'),
            'reasoning': parsed.get('reasoning', ''),
            'extraction_details': parsed.get('extraction_details', []),
            'exclusions_applied': parsed.get('exclusions_applied', []),
            'quality_metrics_completeness': quality_metrics.get('completeness'),
            'quality_metrics_rule_adherence': quality_metrics.get('rule_adherence'),
            'quality_metrics_clinical_accuracy': quality_metrics.get('clinical_accuracy'),
            'quality_metrics_formatting_compliance': quality_metrics.get('formatting_compliance'),
            'success': True,
        }

    def _default_response(self, error: str = None) -> Dict[str, Any]:
        """Return a default response structure for error cases.

        Args:
            error: Optional error message

        Returns:
            Dictionary with default values
        """
        response = {
            'drug_name': '',
            'drug_classes': ['NA'],
            'selected_sources': [],
            'confidence_score': None,
            'reasoning': '',
            'extraction_details': [],
            'exclusions_applied': [],
            'quality_metrics_completeness': None,
            'quality_metrics_rule_adherence': None,
            'quality_metrics_clinical_accuracy': None,
            'quality_metrics_formatting_compliance': None,
            'success': False,
        }
        if error:
            response['error'] = error
        return response
