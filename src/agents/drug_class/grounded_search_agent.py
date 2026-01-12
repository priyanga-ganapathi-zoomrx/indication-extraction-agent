"""Drug Class Grounded Search Agent.

This module implements an agent that identifies drug classes using OpenAI's
web_search_preview tool for grounded search capabilities.

Key features:
- Uses DRUG_CLASS_GROUNDED_SEARCH_PROMPT with 3-message structure
- Enables OpenAI web_search_preview for real-time web search
- Returns drug classes with source URLs, evidence, and annotations
- Captures source annotations from web search results
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.agents.core import settings, get_langfuse_config, LLMConfig, create_llm
from src.agents.drug_class.prompts import get_system_prompt


# Prompt name for grounded search
DRUG_CLASS_GROUNDED_SEARCH_PROMPT_NAME = "DRUG_CLASS_GROUNDED_SEARCH_PROMPT"


class DrugClassGroundedSearchAgent:
    """Drug Class Grounded Search Agent that extracts drug classes with source attribution.

    This agent uses OpenAI's web_search_preview tool to:
    - Identify drug classes from authoritative medical sources
    - Provide actual source URLs for each drug class
    - Include evidence snippets from sources
    - Return confidence levels for each extraction
    - Capture web search annotations with source citations
    """

    def __init__(
        self,
        agent_name: str = "DrugClassGroundedSearchAgent",
        llm_model: str | None = None,
        temperature: float = None,
        max_tokens: int = None,
        enable_caching: bool = False,
        enable_web_search: bool = True,
    ):
        """Initialize the Drug Class Grounded Search Agent.

        Args:
            agent_name: Name of the agent for identification and logging
            llm_model: Optional override for the LLM model name
            temperature: Optional override for LLM temperature
            max_tokens: Optional override for LLM max tokens
            enable_caching: Enable Anthropic prompt caching for reduced costs
            enable_web_search: Enable OpenAI web_search_preview tool (default: True)
        """
        self.agent_name = agent_name
        self._llm_model = llm_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.enable_caching = enable_caching
        self.enable_web_search = enable_web_search

        # Initialize Langfuse
        self.langfuse_config = get_langfuse_config()
        self.langfuse = self._initialize_langfuse() if self.langfuse_config else None

        # Initialize LLM with web search tool
        self.llm_config = self._get_llm_config()
        self.model_kwargs = self._get_model_kwargs()
        self.llm = create_llm(self.llm_config, model_kwargs=self.model_kwargs)

        # Load and parse the 3-message prompt structure
        self.system_prompt, self.rules_message, self.input_template = self._load_prompt_sections()

        # Log configuration
        if self.enable_caching:
            print(f"✓ Prompt caching enabled for {self.agent_name}")
        if self.enable_web_search:
            print(f"✓ Web search (web_search_preview) enabled for {self.agent_name}")

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
        )

    def _get_model_kwargs(self) -> dict:
        """Get model kwargs for LLM configuration.

        Returns:
            dict: Model kwargs including web_search_preview tool if enabled
        """
        if self.enable_web_search:
            return {"tools": [{"type": "web_search_preview"}]}
        return {}

    def _load_prompt_sections(self) -> Tuple[str, str, str]:
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
            prompt_name=DRUG_CLASS_GROUNDED_SEARCH_PROMPT_NAME,
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

        # Validate that all sections were found
        if not system_prompt:
            print("⚠ Warning: SYSTEM_PROMPT section not found in prompt file")
        if not rules_message:
            print("⚠ Warning: RULES_MESSAGE section not found in prompt file")
        if not input_template:
            print("⚠ Warning: INPUT_TEMPLATE section not found in prompt file")

        print(f"✓ Loaded 3-message prompt structure (version: {prompt_version})")
        return system_prompt, rules_message, input_template

    def _format_input_message(
        self,
        drug_name: str,
        abstract_title: str,
    ) -> str:
        """Format the input message for the LLM.

        Args:
            drug_name: The drug name to identify class for
            abstract_title: The abstract title for context

        Returns:
            str: Formatted input message
        """
        # Use the input template structure
        input_content = f"""# EXTRACTION INPUT

## Drug Name
{drug_name}

## Abstract Title
{abstract_title or "Not provided"}"""
        return input_content

    def invoke(
        self,
        drug_name: str,
        abstract_title: str = "",
        abstract_id: str = None,
    ) -> Dict[str, Any]:
        """Invoke the grounded search agent to identify drug class.

        Args:
            drug_name: The drug name to identify class for
            abstract_title: The abstract title for context (optional)
            abstract_id: The abstract ID for tracking in Langfuse (optional)

        Returns:
            dict: Result containing drug classes with sources, evidence, and annotations
        """
        # Build tags for Langfuse tracing
        tags = [
            drug_name,
            f"abstract_id:{abstract_id or 'unknown'}",
            f"prompt_version:{getattr(self, 'prompt_version', 'unknown')}",
            f"model:{self.llm_config.model}",
            "drug_class_grounded_search",
        ]

        # Format the input message
        input_content = self._format_input_message(
            drug_name=drug_name,
            abstract_title=abstract_title,
        )

        # Build 3-message structure
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

        input_msg = HumanMessage(content=input_content)

        # Build messages: System prompt -> Rules message -> Input message
        messages = [
            system_msg,
            rules_msg,
            input_msg,
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

            # Parse response and extract annotations
            return self._parse_response(response, llm_calls=1)
        except Exception as e:
            print(f"✗ Error during grounded search LLM call: {e}")
            import traceback
            traceback.print_exc()
            return self._default_response(
                drug_name=drug_name,
                error=f"LLM call failed: {str(e)}",
                llm_calls=1,
            )

    def _extract_annotations(self, response: AIMessage) -> List[Dict[str, Any]]:
        """Extract annotations from the LLM response.

        OpenAI's web_search_preview returns structured content blocks with
        source citations in the annotations.

        Args:
            response: LLM response message

        Returns:
            List of annotation dictionaries with source information
        """
        annotations = []

        # Check if response has additional_kwargs with annotations
        additional_kwargs = getattr(response, "additional_kwargs", {})
        
        # OpenAI responses may include annotations in different places
        # Check for annotations in additional_kwargs
        if "annotations" in additional_kwargs:
            raw_annotations = additional_kwargs["annotations"]
            if isinstance(raw_annotations, list):
                annotations.extend(raw_annotations)

        # Check if content is a list (structured content blocks)
        content = getattr(response, "content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    # Extract annotations from content blocks
                    if "annotations" in block:
                        block_annotations = block["annotations"]
                        if isinstance(block_annotations, list):
                            annotations.extend(block_annotations)
                    # Also check for url_citation type blocks
                    if block.get("type") == "url_citation":
                        annotations.append({
                            "type": "url_citation",
                            "url": block.get("url", ""),
                            "title": block.get("title", ""),
                            "start_index": block.get("start_index"),
                            "end_index": block.get("end_index"),
                        })

        # Check response_metadata for citations
        response_metadata = getattr(response, "response_metadata", {})
        if "citations" in response_metadata:
            citations = response_metadata["citations"]
            if isinstance(citations, list):
                for citation in citations:
                    annotations.append({
                        "type": "citation",
                        "source": citation,
                    })

        return annotations

    def _get_text_content(self, response: AIMessage) -> str:
        """Extract text content from response, handling both string and structured content.

        Args:
            response: LLM response message

        Returns:
            str: Text content from the response
        """
        content = getattr(response, "content", "")
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            # Concatenate text from all content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif "text" in block:
                        text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            return "\n".join(text_parts)
        
        return str(content)

    def _parse_response(
        self,
        response: AIMessage,
        llm_calls: int = 1,
    ) -> Dict[str, Any]:
        """Parse the response from the LLM.

        Args:
            response: LLM response message
            llm_calls: Number of LLM calls made

        Returns:
            dict: Parsed result with drug classes, metadata, and annotations
        """
        try:
            # Extract text content (handles both string and structured content)
            content = self._get_text_content(response)
            
            # Extract annotations from web search
            annotations = self._extract_annotations(response)
            
            # Store raw response for debugging
            raw_content = getattr(response, "content", "")
            raw_llm_response = str(raw_content) if not isinstance(raw_content, str) else raw_content

            if not content or content.startswith("I encountered an error"):
                return self._default_response(
                    drug_name="",
                    error=f"Empty or error response: {content}",
                    llm_calls=llm_calls,
                    raw_llm_response=raw_llm_response,
                    annotations=annotations,
                )

            # Try to parse JSON response from code block
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    return self._extract_fields(parsed, llm_calls, raw_llm_response, annotations)
                except json.JSONDecodeError:
                    pass  # Try other parsing methods

            # Try to find JSON object without code blocks
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    parsed = json.loads(json_str)
                    if "drug_name" in parsed or "drug_classes" in parsed:
                        return self._extract_fields(parsed, llm_calls, raw_llm_response, annotations)
            except (json.JSONDecodeError, AttributeError):
                pass

            # Failed to parse - return default with raw content
            return self._default_response(
                drug_name="",
                error="Failed to parse JSON response",
                llm_calls=llm_calls,
                raw_llm_response=raw_llm_response,
                annotations=annotations,
            )

        except Exception as e:
            print(f"Error parsing response: {e}")
            import traceback
            traceback.print_exc()
            return self._default_response(
                drug_name="",
                error=f"Parse error: {e}",
                llm_calls=llm_calls,
                raw_llm_response=str(getattr(response, "content", "")),
                annotations=[],
            )

    def _extract_fields(
        self,
        parsed: dict,
        llm_calls: int = 1,
        raw_llm_response: str = None,
        annotations: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract fields from parsed JSON response.

        Args:
            parsed: Parsed JSON dictionary
            llm_calls: Number of LLM calls made
            raw_llm_response: Raw LLM response for debugging
            annotations: List of annotations from web search

        Returns:
            Dictionary with normalized fields
        """
        # Extract drug classes - handle both list of dicts and list of strings
        drug_classes_raw = parsed.get("drug_classes", [])
        drug_classes = []
        drug_class_details = []

        for dc in drug_classes_raw:
            if isinstance(dc, dict):
                class_name = dc.get("class_name", "")
                if class_name:
                    drug_classes.append(class_name)
                    drug_class_details.append({
                        "class_name": class_name,
                        "class_type": dc.get("class_type", ""),
                        "source_url": dc.get("source_url", ""),
                        "source_title": dc.get("source_title", ""),
                        "evidence": dc.get("evidence", ""),
                        "confidence": dc.get("confidence", ""),
                        "rules_applied": dc.get("rules_applied", []),
                    })
            elif isinstance(dc, str) and dc:
                drug_classes.append(dc)
                drug_class_details.append({
                    "class_name": dc,
                    "class_type": "",
                    "source_url": "",
                    "source_title": "",
                    "evidence": "",
                    "confidence": "",
                    "rules_applied": [],
                })

        # If no classes found, set to ["NA"]
        if not drug_classes:
            drug_classes = ["NA"]

        return {
            "drug_name": parsed.get("drug_name", ""),
            "drug_classes": drug_classes,
            "drug_class_details": drug_class_details,
            "reasoning": parsed.get("reasoning", ""),
            "no_class_found": parsed.get("no_class_found", len(drug_classes) == 0 or drug_classes == ["NA"]),
            "llm_calls": llm_calls,
            "success": True,
            "raw_llm_response": raw_llm_response,
            "annotations": annotations or [],
        }

    def _default_response(
        self,
        drug_name: str = "",
        error: str = None,
        llm_calls: int = 0,
        raw_llm_response: str = None,
        annotations: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a default response for error cases.

        Args:
            drug_name: The drug name that was queried
            error: Optional error message
            llm_calls: Number of LLM calls made
            raw_llm_response: Raw LLM response for debugging
            annotations: List of annotations from web search

        Returns:
            dict: Default response structure
        """
        response = {
            "drug_name": drug_name,
            "drug_classes": ["NA"],
            "drug_class_details": [],
            "reasoning": error or "Unable to determine drug class",
            "no_class_found": True,
            "llm_calls": llm_calls,
            "success": False,
            "raw_llm_response": raw_llm_response,
            "annotations": annotations or [],
        }
        if error:
            response["error"] = error
        return response
