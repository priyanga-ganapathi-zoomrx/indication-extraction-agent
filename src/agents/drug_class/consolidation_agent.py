"""Drug Class Consolidation Only Agent.

This module implements an agent that consolidates and deduplicates
explicit drug classes against drug-specific selections.

NOTE: This agent expects PRE-SELECTED drug classes as input.
Selection should be performed upstream before calling this agent.

Uses a 3-message prompt structure optimized for reasoning models:
- SYSTEM_PROMPT: Role, workflow, consolidation rules, output format
- RULES_MESSAGE: 36 extraction rules (for parent-child identification)
- INPUT_TEMPLATE: Consolidation input data
"""

import json
import os
import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.agents.core import settings, get_langfuse_config, LLMConfig, create_llm
from src.agents.drug_class.prompts import get_system_prompt


# Prompt names
CONSOLIDATION_PROMPT_NAME = "DRUG_CLASS_CONSOLIDATION_PROMPT"
EXTRACTION_RULES_PROMPT_NAME = "DRUG_CLASS_EXTRACTION_FROM_SEARCH_REACT_PATTERN"


class DrugClassConsolidationOnlyAgent:
    """Drug Class Consolidation Only Agent.

    This agent performs consolidation of drug classes:
    - Deduplicates explicit drug classes vs drug-specific selections
    - Handles parent-child relationships between explicit and drug-specific classes
    - Removes redundant classes from the explicit list
    
    NOTE: This agent expects pre-selected drug classes. The selection step
    should be performed upstream using DrugClassSelectionAgent.
    """

    def __init__(
        self,
        agent_name: str = "DrugClassConsolidationOnlyAgent",
        llm_model: str | None = None,
        temperature: float = None,
        max_tokens: int = None,
        enable_caching: bool = False,
    ):
        """Initialize the Drug Class Consolidation Only Agent.

        Args:
            agent_name: Name of the agent for identification and logging
            llm_model: Optional override for the LLM model name
            temperature: Optional override for LLM temperature
            max_tokens: Optional override for LLM max tokens
            enable_caching: Enable Anthropic prompt caching for reduced costs
        """
        self.agent_name = agent_name
        self._llm_model = llm_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.enable_caching = enable_caching

        # Initialize Langfuse
        self.langfuse_config = get_langfuse_config()
        self.langfuse = self._initialize_langfuse() if self.langfuse_config else None

        # Initialize LLM
        self.llm_config = self._get_llm_config()
        self.llm = create_llm(self.llm_config)

        # Load and parse prompt sections
        self.system_prompt, self.rules_message, self.input_template = self._load_prompt_sections()

        # Log caching status
        if self.enable_caching:
            print(f"✓ Prompt caching enabled for {self.agent_name}")

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
            model=self._llm_model or settings.llm.LLM_MODEL,
            base_url=settings.llm.LLM_BASE_URL,
            temperature=self._temperature if self._temperature is not None else settings.llm.LLM_TEMPERATURE,
            max_tokens=self._max_tokens or settings.llm.LLM_MAX_TOKENS,
            name=self.agent_name,
        )

    def _load_prompt_sections(self) -> tuple[str, str, str]:
        """Load and parse the 3-message prompt sections from prompt files.

        Loads from two separate files:
        - SYSTEM_PROMPT and INPUT_TEMPLATE from DRUG_CLASS_CONSOLIDATION_PROMPT.md
        - RULES_MESSAGE from DRUG_CLASS_EXTRACTION_FROM_SEARCH_REACT_PATTERN.md (shared rules)

        Returns:
            tuple: (system_prompt, rules_message, input_template)
        """
        # Load the main prompt content (SYSTEM_PROMPT and INPUT_TEMPLATE)
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name=CONSOLIDATION_PROMPT_NAME,
            fallback_to_file=True,
        )
        self.prompt_version = prompt_version

        # Extract SYSTEM_PROMPT section
        system_start = prompt_content.find("<!-- MESSAGE_1_START: SYSTEM_PROMPT -->")
        system_end = prompt_content.find("<!-- MESSAGE_1_END: SYSTEM_PROMPT -->")
        if system_start != -1 and system_end != -1:
            system_prompt = prompt_content[system_start + len("<!-- MESSAGE_1_START: SYSTEM_PROMPT -->"):system_end].strip()
        else:
            raise ValueError("Could not find SYSTEM_PROMPT section markers in prompt file")

        # Extract INPUT_TEMPLATE section (MESSAGE_3 in consolidation prompt file)
        input_start = prompt_content.find("<!-- MESSAGE_3_START: INPUT_TEMPLATE -->")
        input_end = prompt_content.find("<!-- MESSAGE_3_END: INPUT_TEMPLATE -->")
        if input_start != -1 and input_end != -1:
            input_template = prompt_content[input_start + len("<!-- MESSAGE_3_START: INPUT_TEMPLATE -->"):input_end].strip()
        else:
            raise ValueError("Could not find INPUT_TEMPLATE section markers in prompt file")

        # Load RULES_MESSAGE from shared rules file
        rules_content, rules_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name=EXTRACTION_RULES_PROMPT_NAME,
            fallback_to_file=True,
        )
        self.rules_version = rules_version

        # Extract RULES_MESSAGE section from shared rules file
        rules_start = rules_content.find("<!-- MESSAGE_2_START: RULES_MESSAGE -->")
        rules_end = rules_content.find("<!-- MESSAGE_2_END: RULES_MESSAGE -->")
        if rules_start != -1 and rules_end != -1:
            rules_message = rules_content[rules_start + len("<!-- MESSAGE_2_START: RULES_MESSAGE -->"):rules_end].strip()
        else:
            raise ValueError("Could not find RULES_MESSAGE section markers in rules prompt file")

        print(f"✓ Loaded prompt sections (version: {prompt_version}, rules: {rules_version})")
        return system_prompt, rules_message, input_template

    def _format_input_message(
        self,
        abstract_title: str,
        explicit_drug_classes: Dict[str, Any],
        drug_selections: List[Dict[str, Any]],
    ) -> str:
        """Format the input message with consolidation data.

        Args:
            abstract_title: The abstract title
            explicit_drug_classes: Explicit drug classes from title extraction
            drug_selections: Pre-selected drug classes for each drug (with evidence)

        Returns:
            str: Formatted input message with data substituted
        """
        # Format explicit drug classes as JSON
        explicit_json = json.dumps(explicit_drug_classes, indent=2)

        # Format drug selections as JSON
        selections_json = json.dumps(drug_selections, indent=2)

        # Substitute placeholders in template
        formatted = self.input_template.replace("{abstract_title}", abstract_title)
        formatted = formatted.replace("{explicit_drug_classes_json}", explicit_json)
        formatted = formatted.replace("{drug_selections_json}", selections_json)

        return formatted

    def invoke(
        self,
        abstract_title: str,
        explicit_drug_classes: Dict[str, Any],
        drug_selections: List[Dict[str, Any]],
        abstract_id: str = None,
    ) -> Dict[str, Any]:
        """Invoke the consolidation agent to deduplicate drug classes.

        Args:
            abstract_title: The abstract title
            explicit_drug_classes: Dict with "drug_classes" and optional "extraction_details"
            drug_selections: List of dicts with "drug_name", "selected_drug_classes", 
                           "selection_reasoning", and optional "extraction_details"
            abstract_id: Optional abstract ID for tracking in Langfuse

        Returns:
            dict: Consolidation result containing drug_class_mappings,
                  refined_explicit_drug_classes, consolidation_summary, and metadata
        """
        # Handle edge case: empty abstract title
        if not abstract_title or not abstract_title.strip():
            return self._default_response(
                abstract_title, "Empty abstract title provided", llm_calls=0
            )

        # Build tags for Langfuse tracing
        tags = [
            f"abstract_id:{abstract_id or 'unknown'}",
            f"prompt_version:{getattr(self, 'prompt_version', 'unknown')}",
            f"model:{self.llm_config.model}",
            "drug_class_consolidation_only",
            f"drugs_count:{len(drug_selections)}",
        ]

        # Format the input message
        input_message = self._format_input_message(
            abstract_title, explicit_drug_classes, drug_selections
        )

        # Build messages for LLM with optional caching
        if self.enable_caching:
            # Use content blocks with cache_control for Anthropic prompt caching
            system_msg = SystemMessage(content=[
                {"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}
            ])
            rules_msg = HumanMessage(content=[
                {"type": "text", "text": self.rules_message, "cache_control": {"type": "ephemeral"}}
            ])
        else:
            system_msg = SystemMessage(content=self.system_prompt)
            rules_msg = HumanMessage(content=self.rules_message)

        messages = [
            system_msg,
            rules_msg,
            HumanMessage(content=input_message),
        ]

        # Set environment variables for Langfuse callback handler
        if self.langfuse_config:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_config.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_config.secret_key
            os.environ["LANGFUSE_HOST"] = self.langfuse_config.host

        # Configure with Langfuse tracing and tags
        config = RunnableConfig(
            callbacks=[CallbackHandler()] if self.langfuse else [],
            metadata={"langfuse_tags": tags} if self.langfuse else {},
        )

        # Invoke the LLM
        try:
            response: AIMessage = self.llm.invoke(messages, config)

            # Log cache performance metrics if caching is enabled
            if self.enable_caching and hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                if usage:
                    input_token_details = usage.get("input_token_details", {})
                    cache_creation = input_token_details.get("cache_creation", 0)
                    cache_read = input_token_details.get("cache_read", 0)
                    if cache_creation > 0 or cache_read > 0:
                        print(f"  📦 Cache stats - creation: {cache_creation}, read: {cache_read}")

            return self._parse_response(response, abstract_title, llm_calls=1)
        except Exception as e:
            print(f"✗ Error during consolidation LLM call: {e}")
            return self._default_response(
                abstract_title, f"Consolidation failed due to system error: {str(e)}", llm_calls=1
            )

    def _parse_response(
        self, response: AIMessage, abstract_title: str, llm_calls: int = 1
    ) -> Dict[str, Any]:
        """Parse the consolidation response from the LLM.

        Args:
            response: LLM response message
            abstract_title: The abstract title (for reference in output)
            llm_calls: Number of LLM calls made

        Returns:
            dict: Parsed consolidation result with raw_json_response
        """
        try:
            content = getattr(response, "content", "")

            if not content:
                return self._default_response(
                    abstract_title, "Empty response from LLM", llm_calls, content
                )

            # Try to parse JSON response from code block
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    parsed = json.loads(json_str)
                    raw_json_pretty = json.dumps(parsed, indent=2)
                    return self._format_result(parsed, llm_calls, raw_json_pretty)
                except json.JSONDecodeError as e:
                    return self._default_response(
                        abstract_title, f"Failed to parse JSON response: {e}", llm_calls, content
                    )

            # Try to find JSON object without code blocks
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    parsed = json.loads(json_str)
                    if "refined_explicit_drug_classes" in parsed:
                        raw_json_pretty = json.dumps(parsed, indent=2)
                        return self._format_result(parsed, llm_calls, raw_json_pretty)
            except (json.JSONDecodeError, AttributeError):
                pass

            # Fallback: couldn't parse response
            return self._default_response(
                abstract_title, f"Unable to parse consolidation response: {content[:500]}", llm_calls, content
            )

        except Exception as e:
            print(f"Error parsing consolidation response: {e}")
            content = getattr(response, "content", "") if hasattr(response, "content") else ""
            return self._default_response(abstract_title, f"Parse error: {e}", llm_calls, content)

    def _format_result(self, parsed: Dict, llm_calls: int, raw_json_pretty: str) -> Dict[str, Any]:
        """Format the parsed consolidation result into a consistent structure.

        Args:
            parsed: Parsed JSON response from LLM
            llm_calls: Number of LLM calls made
            raw_json_pretty: Pretty-printed JSON string from LLM response

        Returns:
            dict: Formatted consolidation result with raw_json_response
        """
        # Extract refined explicit drug classes (the main output)
        refined_explicit = parsed.get("refined_explicit_drug_classes", {})

        # Extract reasoning
        reasoning = parsed.get("reasoning", "")

        # Count duplicates removed
        removed_classes = refined_explicit.get("removed_classes", [])
        duplicates_removed = len(removed_classes)

        return {
            "refined_explicit_drug_classes": refined_explicit,
            "reasoning": reasoning,
            "duplicates_removed": duplicates_removed,
            "raw_json_response": raw_json_pretty,
            "llm_calls": llm_calls,
            "consolidation_success": True,
        }

    def _default_response(
        self, abstract_title: str, reason: str, llm_calls: int = 0, raw_content: str = ""
    ) -> Dict[str, Any]:
        """Create a default consolidation response for error cases.

        Args:
            abstract_title: The abstract title
            reason: Reason for the default response
            llm_calls: Number of LLM calls made
            raw_content: Raw content from LLM response (if available)

        Returns:
            dict: Default consolidation result
        """
        # Create a default JSON structure for error cases
        default_json = {
            "refined_explicit_drug_classes": {
                "drug_classes": ["NA"],
                "removed_classes": []
            },
            "reasoning": reason
        }
        raw_json_pretty = json.dumps(default_json, indent=2) if not raw_content else raw_content

        return {
            "refined_explicit_drug_classes": {
                "drug_classes": ["NA"],
                "removed_classes": []
            },
            "reasoning": reason,
            "duplicates_removed": 0,
            "raw_json_response": raw_json_pretty,
            "llm_calls": llm_calls,
            "consolidation_success": False,
        }

