"""Drug Class Selection Agent.

This module implements a post-processing agent that selects the most appropriate
drug class(es) from extracted candidates using priority and specificity rules.

Key features:
- Applies class type priority (MoA > Chemical > Mode > Therapeutic)
- Applies specificity rules (child over parent)
- Handles multiple distinct biological targets
- Uses single LLM call (no tool calling needed)
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


# Prompt names for drug class selection
DRUG_CLASS_SELECTION_PROMPT_NAME = "DRUG_CLASS_SELECTION_SYSTEM_PROMPT"
DRUG_CLASS_EXTRACTION_PROMPT_NAME = "DRUG_CLASS_EXTRACTION_FROM_SEARCH_REACT_PATTERN"


class DrugClassSelectionAgent:
    """Drug Class Selection Agent that selects optimal drug classes from candidates.

    This agent applies:
    - Class type priority: MoA > Chemical > Mode > Therapeutic
    - Specificity selection: Prefer child (specific) over parent (broad)
    - Multiple target handling: Return multiple specific classes for distinct targets
    """

    def __init__(
        self,
        agent_name: str = "DrugClassSelectionAgent",
        llm_model: str | None = None,
        temperature: float = None,
        max_tokens: int = None,
        enable_caching: bool = False,
    ):
        """Initialize the Drug Class Selection Agent.

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

        # Initialize LLM (no tool binding for selection)
        self.llm_config = self._get_llm_config()
        self.llm = create_llm(self.llm_config)

        # Load prompt
        self.selection_prompt = self._get_selection_prompt()

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

    def _get_selection_prompt(self) -> str:
        """Get the selection system prompt.

        Fetches the prompt from Langfuse if configured, otherwise falls back to local file.

        Returns:
            str: Selection system prompt content
        """
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name=DRUG_CLASS_SELECTION_PROMPT_NAME,
            fallback_to_file=True,
        )
        self.selection_prompt_version = prompt_version
        return prompt_content

    def _get_extraction_rules_prompt(self) -> str:
        """Get the extraction rules prompt to use as reference.

        Loads the DRUG_CLASS_EXTRACTION_FROM_SEARCH_REACT_PATTERN prompt to provide
        the selector with the rules the extractor was instructed to follow.

        Returns:
            str: Extraction rules prompt content (just the rules section)
        """
        try:
            prompt_content, prompt_version = get_system_prompt(
                langfuse_client=self.langfuse,
                prompt_name=DRUG_CLASS_EXTRACTION_PROMPT_NAME,
                fallback_to_file=True,
            )
            self.extraction_rules_version = prompt_version
            
            # Extract just the RULES_MESSAGE section
            rules_start = prompt_content.find("<!-- MESSAGE_2_START: RULES_MESSAGE -->")
            rules_end = prompt_content.find("<!-- MESSAGE_2_END: RULES_MESSAGE -->")
            
            if rules_start != -1 and rules_end != -1:
                rules_section = prompt_content[rules_start:rules_end + len("<!-- MESSAGE_2_END: RULES_MESSAGE -->")]
                return rules_section
            else:
                # Fallback: return full prompt if markers not found
                return prompt_content
        except Exception as e:
            print(f"✗ Error loading extraction rules prompt: {e}")
            self.extraction_rules_version = "error"
            return "Error loading extraction rules. Please proceed with selection rules only."

    def _format_selection_input(
        self,
        drug_name: str,
        extraction_details: List[Dict],
    ) -> str:
        """Format the selection input JSON for the LLM.

        Args:
            drug_name: The drug name
            extraction_details: List of extraction detail dicts with full information

        Returns:
            str: JSON-formatted input string
        """
        # Transform extraction_details to use 'drug_class' instead of 'normalized_form'
        extracted_classes = []
        for detail in extraction_details:
            extracted_class = {
                "extracted_text": detail.get("extracted_text", ""),
                "class_type": detail.get("class_type", "Therapeutic"),
                "drug_class": detail.get("normalized_form", detail.get("extracted_text", "")),
                "evidence": detail.get("evidence", ""),
                "source": detail.get("source", ""),
                "rules_applied": detail.get("rules_applied", []),
            }
            extracted_classes.append(extracted_class)
        
        input_data = {
            "drug_name": drug_name,
            "extracted_classes": extracted_classes,
        }
        return json.dumps(input_data, indent=2)

    def invoke(
        self,
        drug_name: str,
        extraction_details: List[Dict],
        abstract_id: str = None,
    ) -> Dict[str, Any]:
        """Invoke the selection agent to choose the best drug class(es).

        Args:
            drug_name: The drug name
            extraction_details: List of extraction detail dicts with full information
            abstract_id: The abstract ID for tracking in Langfuse (optional)

        Returns:
            dict: Selection result containing selected classes and reasoning
        """
        # Handle edge case: no classes to select from
        if not extraction_details:
            return {
                "drug_name": drug_name,
                "selected_drug_classes": ["NA"],
                "reasoning": "No extracted classes provided for selection.",
                "llm_calls": 0,
                "selection_success": True,
            }

        # Handle edge case: only one unique class - no LLM call needed
        unique_classes = list(set(
            detail.get('normalized_form', detail.get('extracted_text', ''))
            for detail in extraction_details
            if detail.get('normalized_form') or detail.get('extracted_text')
        ))
        
        if len(unique_classes) == 1:
            return {
                "drug_name": drug_name,
                "selected_drug_classes": [unique_classes[0]],
                "reasoning": "Only one class was extracted. No selection logic needed.",
                "llm_calls": 0,
                "selection_success": True,
            }

        # Load extraction rules
        extraction_rules = self._get_extraction_rules_prompt()

        # Build tags for Langfuse tracing
        tags = [
            drug_name,
            f"abstract_id:{abstract_id or 'unknown'}",
            f"selection_prompt_version:{getattr(self, 'selection_prompt_version', 'unknown')}",
            f"extraction_rules_version:{getattr(self, 'extraction_rules_version', 'unknown')}",
            f"model:{self.llm_config.model}",
            "drug_class_selection",
        ]

        # Format the selection input
        input_json = self._format_selection_input(drug_name, extraction_details)

        # Build the prompt with input substituted
        prompt_with_input = self.selection_prompt.replace("{input_json}", input_json)

        # Build messages for LLM with optional caching
        if self.enable_caching:
            # Use content blocks with cache_control for Anthropic prompt caching
            system_msg = SystemMessage(content=[
                {"type": "text", "text": prompt_with_input, "cache_control": {"type": "ephemeral"}}
            ])
            extraction_rules_msg = HumanMessage(content=[
                {"type": "text", "text": extraction_rules, "cache_control": {"type": "ephemeral"}}
            ])
        else:
            system_msg = SystemMessage(content=prompt_with_input)
            extraction_rules_msg = HumanMessage(content=extraction_rules)

        messages = [
            system_msg,
            extraction_rules_msg,
            HumanMessage(content="Understand the 36 extraction rules, analyze the evidence for each extracted class, and select the most appropriate drug class(es) based on the selection rules."),
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

            return self._parse_selection_response(response, drug_name, llm_calls=1)
        except Exception as e:
            print(f"✗ Error during selection LLM call: {e}")
            return {
                "drug_name": drug_name,
                "selected_drug_classes": ["NA"],
                "reasoning": f"Selection failed due to system error: {str(e)}",
                "llm_calls": 1,
                "selection_success": False,
            }

    def _parse_selection_response(
        self, response: AIMessage, drug_name: str, llm_calls: int = 1
    ) -> Dict[str, Any]:
        """Parse the selection response from the LLM.

        Args:
            response: LLM response message
            drug_name: The drug name
            llm_calls: Number of LLM calls made

        Returns:
            dict: Parsed selection result
        """
        try:
            content = getattr(response, "content", "")

            if not content:
                return self._default_selection_response(
                    drug_name, "Empty response from LLM", llm_calls
                )

            # Try to parse JSON response from code block
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    return {
                        "drug_name": parsed.get("drug_name", drug_name),
                        "selected_drug_classes": parsed.get("selected_drug_classes", ["NA"]),
                        "reasoning": parsed.get("reasoning", ""),
                        "llm_calls": llm_calls,
                        "selection_success": True,
                    }
                except json.JSONDecodeError as e:
                    return self._default_selection_response(
                        drug_name, f"Failed to parse JSON response: {e}", llm_calls
                    )

            # Try to find JSON object without code blocks
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    parsed = json.loads(json_str)
                    if "selected_drug_classes" in parsed:
                        return {
                            "drug_name": parsed.get("drug_name", drug_name),
                            "selected_drug_classes": parsed.get("selected_drug_classes", ["NA"]),
                            "reasoning": parsed.get("reasoning", ""),
                            "llm_calls": llm_calls,
                            "selection_success": True,
                        }
            except (json.JSONDecodeError, AttributeError):
                pass

            # Fallback: couldn't parse response
            return self._default_selection_response(
                drug_name, f"Unable to parse selection response: {content[:500]}", llm_calls
            )

        except Exception as e:
            print(f"Error parsing selection response: {e}")
            return self._default_selection_response(drug_name, f"Parse error: {e}", llm_calls)

    def _default_selection_response(
        self, drug_name: str, reason: str, llm_calls: int = 0
    ) -> Dict[str, Any]:
        """Create a default selection response for error cases.

        Args:
            drug_name: The drug name
            reason: Reason for the default response
            llm_calls: Number of LLM calls made

        Returns:
            dict: Default selection result
        """
        return {
            "drug_name": drug_name,
            "selected_drug_classes": ["NA"],
            "reasoning": reason,
            "llm_calls": llm_calls,
            "selection_success": False,
        }

