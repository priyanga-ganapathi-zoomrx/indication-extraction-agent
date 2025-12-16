"""Drug Class Validation Agent.

This module implements a validation agent that verifies drug class extractions
against established rules, flagging potential errors for manual QC review.

Key features:
- Loads validation prompt and extraction rules as reference
- Uses single LLM call (no tool calling needed)
- Validates hallucination, omission, and rule compliance
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


# Prompt names for drug class validation
DRUG_CLASS_VALIDATION_PROMPT_NAME = "DRUG_CLASS_VALIDATION_SYSTEM_PROMPT"
DRUG_CLASS_EXTRACTION_PROMPT_NAME = "DRUG_CLASS_EXTRACTION_FROM_SEARCH_REACT_PATTERN"


class DrugClassValidationAgent:
    """Drug Class Validation Agent that validates extraction results against rules.

    This agent validates:
    - Hallucination detection: Are extracted drug classes grounded in sources?
    - Omission detection: Are there valid drug classes that weren't extracted?
    - Rule compliance: Were the extraction rules applied correctly?
    """

    def __init__(
        self,
        agent_name: str = "DrugClassValidationAgent",
        llm_model: str | None = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        """Initialize the Drug Class Validation Agent.

        Args:
            agent_name: Name of the agent for identification and logging
            llm_model: Optional override for the LLM model name
            temperature: Optional override for LLM temperature
            max_tokens: Optional override for LLM max tokens
        """
        self.agent_name = agent_name
        self._llm_model = llm_model
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Initialize Langfuse
        self.langfuse_config = get_langfuse_config()
        self.langfuse = self._initialize_langfuse() if self.langfuse_config else None

        # Initialize LLM (no tool binding for validation)
        self.llm_config = self._get_llm_config()
        self.llm = create_llm(self.llm_config)

        # Load prompts
        self.validation_prompt = self._get_validation_prompt()
        self.extraction_rules_prompt = self._get_extraction_rules_prompt()

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

    def _get_validation_prompt(self) -> str:
        """Get the validation system prompt.

        Fetches the prompt from Langfuse if configured, otherwise falls back to local file.

        Returns:
            str: Validation system prompt content
        """
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name=DRUG_CLASS_VALIDATION_PROMPT_NAME,
            fallback_to_file=True,
        )
        self.validation_prompt_version = prompt_version
        return prompt_content

    def _get_extraction_rules_prompt(self) -> str:
        """Get the extraction rules prompt to use as reference for validation.

        Loads the DRUG_CLASS_EXTRACTION_FROM_SEARCH_REACT_PATTERN prompt to provide
        the validator with the rules the extractor was instructed to follow.

        Returns:
            str: Extraction rules prompt content
        """
        try:
            prompt_content, prompt_version = get_system_prompt(
                langfuse_client=self.langfuse,
                prompt_name=DRUG_CLASS_EXTRACTION_PROMPT_NAME,
                fallback_to_file=True,
            )
            self.extraction_rules_version = prompt_version
            print(f"✓ Loaded drug class extraction rules (version: {prompt_version})")
            return prompt_content
        except Exception as e:
            print(f"✗ Error loading extraction rules prompt: {e}")
            self.extraction_rules_version = "error"
            return "Error loading extraction rules. Please verify rule application manually."

    def _format_search_results(self, search_results: List[Dict]) -> str:
        """Format search results for the validation input.

        Args:
            search_results: List of search result dictionaries

        Returns:
            str: Formatted search results string
        """
        if not search_results:
            return "No search results available."

        formatted_parts = []
        for i, result in enumerate(search_results, 1):
            content = result.get("raw_content") or result.get("content", "No content available")
            url = result.get("url", "Unknown URL")

            # Truncate long content
            if len(content) > 3000:
                content = content[:3000] + "... [truncated]"

            formatted_parts.append(f"### Search Result {i}")
            formatted_parts.append(f"**URL**: {url}")
            formatted_parts.append(f"**Content**: {content}")
            formatted_parts.append("")

        return "\n".join(formatted_parts)

    def _format_validation_input(
        self,
        drug_name: str,
        abstract_title: str,
        full_abstract: str,
        search_results: List[Dict],
        extraction_result: Dict[str, Any],
    ) -> str:
        """Format the validation input for the LLM.

        Args:
            drug_name: The drug name being validated
            abstract_title: The abstract title
            full_abstract: The full abstract text
            search_results: List of search result dictionaries
            extraction_result: The extraction result to validate

        Returns:
            str: Formatted input message
        """
        # Format drug classes for display
        drug_classes = extraction_result.get("drug_classes", ["NA"])
        if drug_classes == ["NA"] or not drug_classes:
            drug_classes_display = '["NA"] (extractor returned no drug class)'
        else:
            drug_classes_display = json.dumps(drug_classes)

        # Handle empty extraction case
        empty_notice = ""
        if drug_classes == ["NA"] or not drug_classes:
            empty_notice = """

IMPORTANT:
- The extractor returned NA (no drug class).
- Your job is to determine if a drug class SHOULD exist based on the sources.
- If the sources clearly contain a valid drug class per rules, treat this as a high-severity omission/FAIL.
- If no drug class exists per rules, you may mark PASS but must explain why."""

        # Format search results
        search_results_str = self._format_search_results(search_results)

        # Format extraction details
        extraction_details = extraction_result.get("extraction_details", [])
        extraction_details_str = json.dumps(extraction_details, indent=2) if extraction_details else "[]"

        input_content = f"""## Validation Input

### Drug Information
- **drug_name**: {drug_name}

### Original Sources

**Abstract Title:**
{abstract_title or "Not provided"}

**Full Abstract:**
{full_abstract or "Not provided"}

**Search Results:**
{search_results_str}

### Extraction Result to Validate
- **drug_classes**: {drug_classes_display}
- **selected_sources**: {json.dumps(extraction_result.get("selected_sources", []))}
- **confidence_score**: {extraction_result.get("confidence_score", "N/A")}
- **reasoning**: {extraction_result.get("reasoning", "")}
- **extraction_details**: 
{extraction_details_str}

Please perform all 3 validation checks (Hallucination Detection, Omission Detection, Rule Compliance) and return your validation result in the specified JSON format.{empty_notice}"""

        return input_content

    def invoke(
        self,
        drug_name: str,
        abstract_title: str,
        full_abstract: str,
        search_results: List[Dict],
        extraction_result: Dict[str, Any],
        abstract_id: str = None,
    ) -> Dict[str, Any]:
        """Invoke the validation agent with extraction result to validate.

        Args:
            drug_name: The drug name being validated
            abstract_title: The abstract title
            full_abstract: The full abstract text
            search_results: List of search result dictionaries
            extraction_result: The extraction result to validate
            abstract_id: The abstract ID for tracking in Langfuse (optional)

        Returns:
            dict: Validation result containing status, issues, and reasoning
        """
        # Build tags for Langfuse tracing
        tags = [
            drug_name,
            f"abstract_id:{abstract_id or 'unknown'}",
            f"validation_prompt_version:{getattr(self, 'validation_prompt_version', 'unknown')}",
            f"extraction_rules_version:{getattr(self, 'extraction_rules_version', 'unknown')}",
            f"model:{self.llm_config.model}",
            "drug_class_validation",
        ]

        # Format the reference rules message
        reference_rules_content = f"""## REFERENCE RULES DOCUMENT

The following is the complete extraction rules document that the extractor was instructed to follow. Use this as your authoritative reference to verify compliance.

---

{self.extraction_rules_prompt}

---

END OF REFERENCE RULES DOCUMENT"""

        # Format the validation input message
        input_content = self._format_validation_input(
            drug_name=drug_name,
            abstract_title=abstract_title,
            full_abstract=full_abstract,
            search_results=search_results,
            extraction_result=extraction_result,
        )

        # Build messages for LLM
        messages = [
            SystemMessage(content=self.validation_prompt),
            HumanMessage(content=reference_rules_content),
            HumanMessage(content=input_content),
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
            return self._parse_validation_response(response, llm_calls=1)
        except Exception as e:
            print(f"✗ Error during validation LLM call: {e}")
            return {
                "validation_status": "REVIEW",
                "validation_confidence": 0.0,
                "issues_found": [
                    {
                        "check_type": "system_error",
                        "severity": "high",
                        "description": f"Validation failed due to system error: {str(e)}",
                        "evidence": "",
                        "drug_class": "",
                        "rule_reference": "",
                    }
                ],
                "checks_performed": {},
                "validation_reasoning": f"Validation could not be completed due to error: {str(e)}",
                "llm_calls": 1,
            }

    def _parse_validation_response(self, response: AIMessage, llm_calls: int = 1) -> Dict[str, Any]:
        """Parse the validation response from the LLM.

        Args:
            response: LLM response message
            llm_calls: Number of LLM calls made

        Returns:
            dict: Parsed validation result
        """
        try:
            content = getattr(response, "content", "")

            if not content or content.startswith("I encountered an error"):
                return self._default_validation_response(
                    f"Validation error: {content}", llm_calls
                )

            # Try to parse JSON response from code block
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    return {
                        "validation_status": parsed.get("validation_status", "REVIEW"),
                        "validation_confidence": parsed.get("validation_confidence", 0.5),
                        "issues_found": parsed.get("issues_found", []),
                        "checks_performed": parsed.get("checks_performed", {}),
                        "validation_reasoning": parsed.get("validation_reasoning", ""),
                        "llm_calls": llm_calls,
                    }
                except json.JSONDecodeError as e:
                    return self._default_validation_response(
                        f"Failed to parse JSON response: {e}", llm_calls
                    )

            # Try to find JSON object without code blocks
            try:
                # Look for JSON object with validation_status
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    parsed = json.loads(json_str)
                    if "validation_status" in parsed:
                        return {
                            "validation_status": parsed.get("validation_status", "REVIEW"),
                            "validation_confidence": parsed.get("validation_confidence", 0.5),
                            "issues_found": parsed.get("issues_found", []),
                            "checks_performed": parsed.get("checks_performed", {}),
                            "validation_reasoning": parsed.get("validation_reasoning", ""),
                            "llm_calls": llm_calls,
                        }
            except (json.JSONDecodeError, AttributeError):
                pass

            # Fallback: extract status from text
            status = "REVIEW"
            if "PASS" in content.upper() and "FAIL" not in content.upper():
                status = "PASS"
            elif "FAIL" in content.upper():
                status = "FAIL"

            return {
                "validation_status": status,
                "validation_confidence": 0.5,
                "issues_found": [],
                "checks_performed": {},
                "validation_reasoning": content[:2000] if content else "Unable to parse validation response",
                "llm_calls": llm_calls,
            }

        except Exception as e:
            print(f"Error parsing validation response: {e}")
            return self._default_validation_response(f"Parse error: {e}", llm_calls)

    def _default_validation_response(self, reason: str, llm_calls: int = 0) -> Dict[str, Any]:
        """Create a default validation response for error cases.

        Args:
            reason: Reason for the default response
            llm_calls: Number of LLM calls made

        Returns:
            dict: Default validation result
        """
        return {
            "validation_status": "REVIEW",
            "validation_confidence": 0.0,
            "issues_found": [
                {
                    "check_type": "system_error",
                    "severity": "medium",
                    "description": reason,
                    "evidence": "",
                    "drug_class": "",
                    "rule_reference": "",
                }
            ],
            "checks_performed": {},
            "validation_reasoning": reason,
            "llm_calls": llm_calls,
        }

