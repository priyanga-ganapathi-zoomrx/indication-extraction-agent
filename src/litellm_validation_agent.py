"""LiteLLM-based Indication Validation Agent.

This module implements an indication validation agent using the LiteLLM SDK directly,
bypassing LangChain to avoid Gemini-specific property stripping issues (e.g., thought_signature).
Designed specifically for Gemini 3 Pro compatibility.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional

import litellm
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool

from src.config import settings
from src.prompts import get_system_prompt, EXTRACTION_PROMPT_NAME, VALIDATION_PROMPT_NAME
from src.tools import get_tools


# Define response schema for validation
class ValidationIssue(BaseModel):
    check_type: str = Field(description="Type of check: hallucination, omission, source_selection, rule_application, exclusion_compliance, formatting")
    severity: str = Field(description="Severity level: high, medium, low")
    description: str = Field(description="Clear description of the issue found")
    evidence: str = Field(description="Specific evidence supporting this finding")
    component: str = Field(description="The specific component involved (if applicable)")


class CheckResult(BaseModel):
    passed: bool = Field(description="Whether the check passed")
    note: str = Field(description="Explanation of the check result")


class ChecksPerformed(BaseModel):
    source_selection: Optional[CheckResult] = None
    hallucination_check: Optional[CheckResult] = None
    omission_check: Optional[CheckResult] = None
    rule_application: Optional[CheckResult] = None
    exclusion_compliance: Optional[CheckResult] = None
    formatting_compliance: Optional[CheckResult] = None


class ValidationResponse(BaseModel):
    validation_status: str = Field(description="PASS, REVIEW, or FAIL")
    validation_confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    issues_found: List[ValidationIssue] = Field(default_factory=list)
    checks_performed: Dict[str, Any] = Field(default_factory=dict)
    validation_reasoning: str = Field(description="Step-by-step explanation of the validation process")


class LiteLLMValidationAgent:
    """Indication Validation Agent using LiteLLM directly.
    
    This agent validates indication extractions against established rules,
    flagging potential errors for manual QC review. Uses LiteLLM SDK directly
    to ensure full Gemini 3 Pro compatibility.
    """

    def __init__(
        self,
        agent_name: str = "LiteLLMValidationAgent",
        llm_model: str = None
    ):
        """Initialize the LiteLLM Validation Agent.

        Args:
            agent_name: Name of the agent for identification and logging
            llm_model: Optional override for the LLM model name (defaults to settings)
        """
        self.agent_name = agent_name
        self.llm_model = llm_model or settings.llm.LLM_MODEL
        
        # Initialize tools
        self.tools = get_tools()
        self.tools_map = {tool.name: tool for tool in self.tools}
        self.tools_schema = [convert_to_openai_tool(tool) for tool in self.tools]
        
        # Initialize Langfuse
        self._initialize_langfuse()
        
        # Load prompts
        self._load_prompts()
        
        print(f"✓ {self.agent_name} initialized with model: {self.llm_model}")

    def _initialize_langfuse(self):
        """Initialize Langfuse configuration for LiteLLM."""
        if settings.langfuse.LANGFUSE_PUBLIC_KEY and settings.langfuse.LANGFUSE_SECRET_KEY:
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse.LANGFUSE_PUBLIC_KEY
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse.LANGFUSE_SECRET_KEY
            os.environ["LANGFUSE_HOST"] = settings.langfuse.LANGFUSE_HOST
            
            # Set LiteLLM callbacks to use OTEL integration
            litellm.callbacks = ["langfuse_otel"]
            print(f"✓ Langfuse OTEL integration configured for {self.agent_name}")
        else:
            print(f"ℹ Langfuse not configured (missing keys)")

    def _load_prompts(self):
        """Load system prompts for validation."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load validation system prompt
        try:
            validation_prompt_path = os.path.join(current_dir, "prompts", f"{VALIDATION_PROMPT_NAME}.md")
            with open(validation_prompt_path, 'r', encoding='utf-8') as f:
                self.validation_prompt = f.read().strip()
            print(f"✓ Loaded validation prompt from local file")
            self.validation_prompt_version = "local_file"
        except Exception as e:
            print(f"✗ Error loading validation prompt: {e}")
            self.validation_prompt = "You are a medical indication validation assistant."
            self.validation_prompt_version = "fallback"
        
        # Load extraction rules prompt (reference for validation)
        try:
            extraction_prompt_path = os.path.join(current_dir, "prompts", f"{EXTRACTION_PROMPT_NAME}.md")
            with open(extraction_prompt_path, 'r', encoding='utf-8') as f:
                self.extraction_rules_prompt = f.read().strip()
            print(f"✓ Loaded extraction rules prompt from local file")
            self.extraction_rules_version = "local_file"
        except Exception as e:
            print(f"✗ Error loading extraction rules prompt: {e}")
            self.extraction_rules_prompt = "Error loading extraction rules. Please verify rule application manually."
            self.extraction_rules_version = "error"

    def _format_validation_input(
        self,
        session_title: str,
        abstract_title: str,
        extraction_result: Dict[str, Any],
    ) -> str:
        """Format the validation input for the LLM.

        Args:
            session_title: The session/conference title
            abstract_title: The research abstract title
            extraction_result: The extraction result to validate

        Returns:
            str: Formatted input message
        """
        generated_indication = extraction_result.get('indication', '')
        indication_display = (
            generated_indication
            if str(generated_indication).strip()
            else "(EMPTY - extractor returned nothing; validate whether an indication should exist)"
        )

        empty_notice = ""
        if not str(generated_indication).strip():
            empty_notice = """

IMPORTANT:
- The extractor returned an empty indication.
- Your job is to determine if an indication SHOULD exist based on the titles.
- If the titles clearly contain a valid indication, treat this as a high-severity omission/FAIL.
- If no indication exists in the titles, you may mark PASS/REVIEW but must explain why no indication is expected."""

        input_content = f"""## Validation Input

### Original Titles
- **session_title**: {session_title}
- **abstract_title**: {abstract_title}

### Extraction Result to Validate
- **generated_indication**: {indication_display}
- **selected_source**: {extraction_result.get('selected_source', '')}
- **confidence_score**: {extraction_result.get('confidence_score', 'N/A')}
- **reasoning**: {extraction_result.get('reasoning', '')}
- **rules_retrieved**: {json.dumps(extraction_result.get('rules_retrieved', []), indent=2)}
- **components_identified**: {json.dumps(extraction_result.get('components_identified', []), indent=2)}

Please perform all 6 validation checks and return your validation result in the specified JSON format.{empty_notice}"""

        return input_content

    def invoke(
        self,
        session_title: str,
        abstract_title: str,
        extraction_result: Dict[str, Any],
        abstract_id: str = None,
    ) -> Dict[str, Any]:
        """Invoke the validation agent with extraction result to validate.

        Args:
            session_title: The session/conference title
            abstract_title: The research abstract title
            extraction_result: The extraction result to validate
            abstract_id: The abstract ID for tracking in Langfuse (optional)

        Returns:
            dict: Validation result containing status, issues, and reasoning
        """
        # Format the reference rules message
        reference_rules_content = f"""## REFERENCE RULES DOCUMENT

The following is the complete extraction rules document that the extractor was instructed to follow. Use this as your reference to verify compliance.

---

{self.extraction_rules_prompt}

---

END OF REFERENCE RULES DOCUMENT"""

        # Format the validation input message
        validation_input = self._format_validation_input(
            session_title, abstract_title, extraction_result
        )

        # Build messages - system prompt + reference rules + validation input
        messages = [
            {"role": "system", "content": self.validation_prompt},
            {"role": "user", "content": reference_rules_content},
            {"role": "user", "content": validation_input}
        ]

        # Metadata for Langfuse tracing
        metadata = {
            "agent_name": self.agent_name,
            "abstract_id": abstract_id,
            "session_title": session_title[:100] if session_title else "",
            "abstract_title": abstract_title[:100] if abstract_title else "",
            "tags": [
                f"abstract_id:{abstract_id or 'unknown'}",
                f"model:{self.llm_model}",
                f"validation_prompt_version:{self.validation_prompt_version}",
                f"extraction_rules_version:{self.extraction_rules_version}",
                "validation"
            ]
        }

        llm_calls = 0
        max_iterations = 10  # Safety limit for tool call loops

        while llm_calls < max_iterations:
            try:
                # Build LiteLLM completion parameters
                completion_params = {
                    "model": self.llm_model,
                    "messages": messages,
                    "tools": self.tools_schema,
                    "tool_choice": "auto",
                    "temperature": 1,
                    "max_tokens": settings.llm.LLM_MAX_TOKENS,
                    "metadata": metadata,
                }
                
                # Add base_url and api_key if configured
                if settings.llm.LLM_BASE_URL:
                    completion_params["base_url"] = settings.llm.LLM_BASE_URL
                if settings.llm.LLM_API_KEY:
                    completion_params["api_key"] = settings.llm.LLM_API_KEY
                
                # Add Gemini-specific parameters for thinking models
                if "gemini" in self.llm_model.lower() and ("3" in self.llm_model or "thinking" in self.llm_model.lower()):
                    completion_params["reasoning_effort"] = "high"
                
                response = litellm.completion(**completion_params)
                llm_calls += 1

            except Exception as e:
                print(f"✗ Error during LLM call: {e}")
                return self._default_validation_response(f"LLM call error: {str(e)}", llm_calls)

            response_message = response.choices[0].message
            
            # Append assistant message to history
            messages.append(response_message.model_dump())

            tool_calls = response_message.tool_calls

            if tool_calls:
                # Process tool calls
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        result = f"Error parsing tool arguments: {str(e)}"
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": result
                        })
                        continue

                    if function_name in self.tools_map:
                        tool_function = self.tools_map[function_name]
                        try:
                            result = tool_function.invoke(function_args)
                        except Exception as e:
                            result = f"Error executing tool {function_name}: {str(e)}"
                    else:
                        result = f"Error: Tool {function_name} not found"

                    # Append tool result to messages
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(result)
                    })
            else:
                # No tool calls, this is the final response
                return self._parse_validation_response(response_message.content, llm_calls)

        # Max iterations reached
        return self._default_validation_response(
            f"Maximum iterations ({max_iterations}) reached without final response",
            llm_calls
        )

    def _parse_validation_response(self, content: str, llm_calls: int) -> Dict[str, Any]:
        """Parse the validation response from the LLM.

        Args:
            content: Raw response content from the LLM
            llm_calls: Number of LLM calls made

        Returns:
            dict: Parsed validation result including raw_llm_response
        """
        # Always include the raw LLM response
        raw_response = content if content else ""
        
        if not content:
            result = self._default_validation_response("Empty response from LLM", llm_calls)
            result["raw_llm_response"] = raw_response
            return result

        if content.startswith("I encountered an error"):
            result = self._default_validation_response(f"LLM error: {content}", llm_calls)
            result["raw_llm_response"] = raw_response
            return result

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
                    "raw_llm_response": raw_response,
                }
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON from code block: {e}")

        # Try to find JSON object directly (without code blocks)
        try:
            # Look for a complete JSON object with validation_status
            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*"validation_status"(?:[^{}]|(?:\{[^{}]*\}))*\}'
            json_match = re.search(json_pattern, content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return {
                    "validation_status": parsed.get("validation_status", "REVIEW"),
                    "validation_confidence": parsed.get("validation_confidence", 0.5),
                    "issues_found": parsed.get("issues_found", []),
                    "checks_performed": parsed.get("checks_performed", {}),
                    "validation_reasoning": parsed.get("validation_reasoning", ""),
                    "llm_calls": llm_calls,
                    "raw_llm_response": raw_response,
                }
        except (json.JSONDecodeError, AttributeError):
            pass

        # Try to parse as direct JSON
        try:
            # Remove any leading/trailing text and try to find JSON
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx + 1]
                parsed = json.loads(json_str)
                if "validation_status" in parsed:
                    return {
                        "validation_status": parsed.get("validation_status", "REVIEW"),
                        "validation_confidence": parsed.get("validation_confidence", 0.5),
                        "issues_found": parsed.get("issues_found", []),
                        "checks_performed": parsed.get("checks_performed", {}),
                        "validation_reasoning": parsed.get("validation_reasoning", ""),
                        "llm_calls": llm_calls,
                        "raw_llm_response": raw_response,
                    }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract status from text
        status = "REVIEW"
        content_upper = content.upper()
        if "PASS" in content_upper and "FAIL" not in content_upper:
            status = "PASS"
        elif "FAIL" in content_upper:
            status = "FAIL"

        return {
            "validation_status": status,
            "validation_confidence": 0.5,
            "issues_found": [],
            "checks_performed": {},
            "validation_reasoning": content[:2000] if content else "Unable to parse validation response",
            "llm_calls": llm_calls,
            "raw_llm_response": raw_response,
        }

    def _default_validation_response(self, reason: str, llm_calls: int = 0, raw_response: str = "") -> Dict[str, Any]:
        """Create a default validation response for error cases.

        Args:
            reason: Reason for the default response
            llm_calls: Number of LLM calls made
            raw_response: Raw LLM response content (if any)

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
                    "component": "",
                }
            ],
            "checks_performed": {},
            "validation_reasoning": reason,
            "llm_calls": llm_calls,
            "raw_llm_response": raw_response,
        }


# Convenience function for running the validation
def run_validation(
    session_title: str,
    abstract_title: str,
    extraction_result: Dict[str, Any],
    abstract_id: str = None,
    llm_model: str = None,
) -> Dict[str, Any]:
    """Convenience function to run indication validation.

    Args:
        session_title: The session/conference title
        abstract_title: The research abstract title
        extraction_result: The extraction result to validate
        abstract_id: The abstract ID for tracking
        llm_model: Optional LLM model override

    Returns:
        dict: Validation result
    """
    agent = LiteLLMValidationAgent(llm_model=llm_model)
    return agent.invoke(
        session_title=session_title,
        abstract_title=abstract_title,
        extraction_result=extraction_result,
        abstract_id=abstract_id
    )

