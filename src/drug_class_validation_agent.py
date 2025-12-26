"""Drug Class Validation Agent.

This module implements a validation agent that verifies drug class extractions
against established rules, flagging potential errors for manual QC review.

Key features:
- Loads validation prompt and extraction rules as reference
- Uses LiteLLM SDK directly with native web search enabled
- Enforces structured JSON output via Pydantic response_format
- Validates hallucination, omission, and rule compliance
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional

import litellm
from litellm.exceptions import APIError, RateLimitError, ServiceUnavailableError, Timeout
from pydantic import BaseModel, Field, model_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.prompts import get_system_prompt


# Prompt names for drug class validation
DRUG_CLASS_VALIDATION_PROMPT_NAME = "DRUG_CLASS_VALIDATION_SYSTEM_PROMPT"
DRUG_CLASS_EXTRACTION_PROMPT_NAME = "DRUG_CLASS_EXTRACTION_FROM_SEARCH_REACT_PATTERN"


# =============================================================================
# Pydantic Response Schema Models (matching system prompt exactly)
# =============================================================================

class ValidationIssue(BaseModel):
    """Issue found during validation."""
    check_type: Literal["hallucination", "omission", "rule_compliance"]
    severity: Literal["high", "medium", "low"]
    description: str = Field(description="Clear description of the issue found")
    evidence: str = Field(description="Specific evidence from sources supporting this finding")
    drug_class: str = Field(default="", description="The specific drug class involved")
    transformed_drug_class: Optional[str] = Field(
        default=None,
        description="The correctly transformed drug class after applying the rule (required for rule_compliance)"
    )
    rule_reference: str = Field(default="", description="Rule X (if applicable)")

    @model_validator(mode='after')
    def validate_transformed_drug_class(self) -> 'ValidationIssue':
        """Ensure transformed_drug_class is provided for rule_compliance issues."""
        if self.check_type == "rule_compliance" and not self.transformed_drug_class:
            raise ValueError("transformed_drug_class is required for rule_compliance issues")
        return self


class CheckResult(BaseModel):
    """Result of a single validation check."""
    passed: bool
    note: str


class ChecksPerformed(BaseModel):
    """All validation checks performed."""
    hallucination_detection: Optional[CheckResult] = None
    omission_detection: Optional[CheckResult] = None
    rule_compliance: Optional[CheckResult] = None


class ExtractedDrugClass(BaseModel):
    """Drug class extracted via grounded search."""
    class_name: str = Field(description="The drug class formatted per rules")
    class_type: Literal["MoA", "Chemical", "Mode", "Therapeutic"]
    source_url: str = Field(description="Actual URL where the drug class was found")
    source_title: str = Field(description="Title of the source page")
    evidence: str = Field(description="Exact text snippet from source")
    confidence: Literal["high", "medium", "low"]


class DrugClassValidationResponse(BaseModel):
    """Complete validation response schema."""
    validation_status: Literal["PASS", "REVIEW", "FAIL"]
    validation_confidence: float = Field(ge=0.0, le=1.0)
    extraction_performed: bool = Field(
        default=False,
        description="true ONLY when input drug_classes was ['NA'] or []. Must be false when validating existing drug classes."
    )
    extracted_drug_classes: List[ExtractedDrugClass] = Field(
        default_factory=list,
        description="ONLY for grounded search results. Empty [] when extraction_performed is false. Never copy from input."
    )
    missed_drug_classes: List[str] = Field(default_factory=list)
    issues_found: List[ValidationIssue] = Field(default_factory=list)
    checks_performed: ChecksPerformed
    validation_reasoning: str


# =============================================================================
# Drug Class Validation Agent
# =============================================================================

class DrugClassValidationAgent:
    """Drug Class Validation Agent that validates extraction results against rules.

    This agent validates:
    - Hallucination detection: Are extracted drug classes grounded in sources?
    - Omission detection: Are there valid drug classes that weren't extracted?
    - Rule compliance: Were the extraction rules applied correctly?

    Uses LiteLLM SDK directly with native web search enabled for grounded validation.
    Enforces structured JSON output via Pydantic response_format.
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

        # Initialize Langfuse OTEL integration for LiteLLM
        self._initialize_langfuse()

        # Load prompts
        self.validation_prompt = self._get_validation_prompt()
        self.extraction_rules_prompt = self._get_extraction_rules_prompt()

        print(f"✓ {self.agent_name} initialized with model: {self._llm_model or settings.llm.LLM_MODEL}")

    def _initialize_langfuse(self):
        """Initialize Langfuse configuration for LiteLLM OTEL integration."""
        if settings.langfuse.LANGFUSE_PUBLIC_KEY and settings.langfuse.LANGFUSE_SECRET_KEY:
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse.LANGFUSE_PUBLIC_KEY
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse.LANGFUSE_SECRET_KEY
            os.environ["LANGFUSE_HOST"] = settings.langfuse.LANGFUSE_HOST
            litellm.callbacks = ["langfuse_otel"]
            print(f"✓ Langfuse OTEL integration configured for {self.agent_name}")
        else:
            print("ℹ Langfuse not configured (missing keys)")

    def _get_validation_prompt(self) -> str:
        """Get the validation system prompt.

        Fetches the prompt from local file.

        Returns:
            str: Validation system prompt content
        """
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=None,
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
                langfuse_client=None,
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((Timeout, APIError, ServiceUnavailableError, RateLimitError)),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    def _call_llm_with_retry(self, completion_params: dict):
        """Call LiteLLM with exponential backoff retry.

        Args:
            completion_params: Parameters to pass to litellm.completion()

        Returns:
            LiteLLM completion response

        Raises:
            Exception: Re-raises the exception after all retries are exhausted
        """
        return litellm.completion(**completion_params)

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

        # Handle empty extraction case - triggers EXTRACTION MODE
        empty_notice = ""
        if drug_classes == ["NA"] or not drug_classes:
            empty_notice = """

IMPORTANT - EXTRACTION MODE TRIGGERED:
- The extractor returned NA (no drug class).
- First, run omission detection on original sources. If missed classes are found, add them to "missed_drug_classes" array.
- Then, ALWAYS perform GROUNDED SEARCH EXTRACTION to find the drug class.
- Use your search grounding capability to query authoritative sources (FDA, NIH, NCI, etc.).
- Apply ALL rules from the reference document to format the extracted drug class.
- Set "extraction_performed": true and populate "extracted_drug_classes" in your output.
- Both "missed_drug_classes" (from original sources) and "extracted_drug_classes" (from grounded search) can be populated.
- If HIGH severity omission found in original sources, set validation_status to FAIL.
- If you successfully extract a drug class via grounded search with no omissions, set validation_status to PASS.
- If no drug class found even with grounded search, set validation_status to REVIEW and explain why."""

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
        model_name = self._llm_model or settings.llm.LLM_MODEL
        tags = [
            drug_name,
            f"abstract_id:{abstract_id or 'unknown'}",
            f"validation_prompt_version:{getattr(self, 'validation_prompt_version', 'unknown')}",
            f"extraction_rules_version:{getattr(self, 'extraction_rules_version', 'unknown')}",
            f"model:{model_name}",
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

        # Build messages for LiteLLM (OpenAI-compatible format)
        messages = [
            {"role": "system", "content": self.validation_prompt},
            {"role": "user", "content": reference_rules_content},
            {"role": "user", "content": input_content},
        ]

        # Metadata for Langfuse tracing
        metadata = {
            "agent_name": self.agent_name,
            "abstract_id": abstract_id,
            "drug_name": drug_name,
            "tags": tags,
        }

        # Invoke LiteLLM with web search and structured output enabled
        try:
            # Build completion parameters
            completion_params = {
                "model": model_name,
                "messages": messages,
                "temperature": self._temperature if self._temperature is not None else settings.llm.LLM_TEMPERATURE,
                "max_tokens": self._max_tokens or settings.llm.LLM_MAX_TOKENS,
                "response_format": DrugClassValidationResponse,  # Pydantic model for structured output
                "web_search_options": {"search_context_size": "medium"},
                "metadata": metadata,
                "timeout": 90,  # Client-side timeout (seconds) to prevent nginx 504 gateway timeouts
            }

            # Add base_url and api_key if configured
            if settings.llm.LLM_BASE_URL:
                completion_params["base_url"] = settings.llm.LLM_BASE_URL
            if settings.llm.LLM_API_KEY:
                completion_params["api_key"] = settings.llm.LLM_API_KEY

            response = self._call_llm_with_retry(completion_params)

            # Extract content from response
            content = response.choices[0].message.content

            return self._parse_validation_response(content, llm_calls=1)
        except Exception as e:
            # Determine error type for tracking
            error_type = None
            if isinstance(e, Timeout):
                error_type = "timeout"
            elif isinstance(e, RateLimitError):
                error_type = "rate_limit"
            elif isinstance(e, (APIError, ServiceUnavailableError)):
                error_type = "api_error"
            else:
                error_type = "unknown_error"
            
            print(f"✗ Error during validation LLM call: {e}")
            return {
                "validation_status": "REVIEW",
                "validation_confidence": 0.0,
                "extraction_performed": False,
                "extracted_drug_classes": [],
                "missed_drug_classes": [],
                "issues_found": [
                    {
                        "check_type": "system_error",
                        "severity": "high",
                        "description": f"Validation failed due to system error: {str(e)}",
                        "evidence": "",
                        "drug_class": "",
                        "transformed_drug_class": None,
                        "rule_reference": "",
                    }
                ],
                "checks_performed": {},
                "validation_reasoning": f"Validation could not be completed due to error: {str(e)}",
                "llm_calls": 1,
                "error_type": error_type,
            }

    def _parse_validation_response(self, content: str, llm_calls: int = 1) -> Dict[str, Any]:
        """Parse the validation response from the LLM.

        Uses Pydantic model_validate_json for structured parsing when possible,
        with fallback to regex-based JSON extraction.

        Args:
            content: Raw response content from the LLM
            llm_calls: Number of LLM calls made

        Returns:
            dict: Parsed validation result
        """
        if not content:
            return self._default_validation_response("Empty response from LLM", llm_calls)

        if content.startswith("I encountered an error"):
            return self._default_validation_response(f"Validation error: {content}", llm_calls)

        # Try to parse with Pydantic model first (structured output)
        try:
            # If response_format worked, content should be valid JSON
            parsed = DrugClassValidationResponse.model_validate_json(content)
            result = parsed.model_dump()
            result["llm_calls"] = llm_calls
            result["error_type"] = None
            return result
        except Exception:
            # Pydantic parsing failed, try fallback methods
            pass

        # Fallback: Try to extract JSON from code block
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group(1))
                return self._extract_validation_fields(parsed_json, llm_calls)
            except json.JSONDecodeError:
                pass

        # Fallback: Try to find raw JSON object
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                parsed_json = json.loads(json_str)
                if "validation_status" in parsed_json:
                    return self._extract_validation_fields(parsed_json, llm_calls)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Final fallback: extract status from text
        status = "REVIEW"
        if "PASS" in content.upper() and "FAIL" not in content.upper():
            status = "PASS"
        elif "FAIL" in content.upper():
            status = "FAIL"

        return {
            "validation_status": status,
            "validation_confidence": 0.5,
            "extraction_performed": False,
            "extracted_drug_classes": [],
            "missed_drug_classes": [],
            "issues_found": [],
            "checks_performed": {},
            "validation_reasoning": content[:2000] if content else "Unable to parse validation response",
            "llm_calls": llm_calls,
            "error_type": None,
        }

    def _extract_validation_fields(self, parsed_json: Dict[str, Any], llm_calls: int) -> Dict[str, Any]:
        """Extract validation fields from parsed JSON with defaults.

        Args:
            parsed_json: Parsed JSON dictionary
            llm_calls: Number of LLM calls made

        Returns:
            dict: Validated fields with defaults
        """
        return {
            "validation_status": parsed_json.get("validation_status", "REVIEW"),
            "validation_confidence": parsed_json.get("validation_confidence", 0.5),
            "extraction_performed": parsed_json.get("extraction_performed", False),
            "extracted_drug_classes": parsed_json.get("extracted_drug_classes", []),
            "missed_drug_classes": parsed_json.get("missed_drug_classes", []),
            "issues_found": parsed_json.get("issues_found", []),
            "checks_performed": parsed_json.get("checks_performed", {}),
            "validation_reasoning": parsed_json.get("validation_reasoning", ""),
            "llm_calls": llm_calls,
            "error_type": None,
        }

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
            "extraction_performed": False,
            "extracted_drug_classes": [],
            "missed_drug_classes": [],
            "issues_found": [
                {
                    "check_type": "system_error",
                    "severity": "medium",
                    "description": reason,
                    "evidence": "",
                    "drug_class": "",
                    "transformed_drug_class": None,
                    "rule_reference": "",
                }
            ],
            "checks_performed": {},
            "validation_reasoning": reason,
            "llm_calls": llm_calls,
            "error_type": None,
        }
