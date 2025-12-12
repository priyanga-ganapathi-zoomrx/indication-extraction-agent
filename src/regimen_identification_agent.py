"""Regimen Identification Agent.

This module implements a regimen identification agent that identifies
if a drug is a clinical regimen and extracts its component drugs.

It sends:
1. System Prompt - The regimen identification instructions
2. Input Message - Abstract title and single drug name

Returns JSON with components array.
"""

import json
import os
import re
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.config import settings
from src.langfuse_config import get_langfuse_config
from src.llm_handler import LLMConfig, create_llm
from src.prompts import get_system_prompt


class RegimenIdentificationAgent:
    """Regimen Identification Agent.

    This agent identifies if a drug is a clinical regimen (e.g., CHOP, FOLFIRI)
    and extracts its component drugs. If the drug is not a regimen, it returns
    the drug itself in the components array.
    """

    def __init__(
        self,
        agent_name: str = "RegimenIdentificationAgent",
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        """Initialize the Regimen Identification Agent.

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

        # Initialize LLM
        self.llm_config = self._get_llm_config()
        self.llm = create_llm(self.llm_config)

        # Load the prompt
        self.system_prompt = self._load_prompt()

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

    def _load_prompt(self) -> str:
        """Load the regimen identification prompt from file.

        Returns:
            str: The prompt content
        """
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name="REGIMEN_IDENTIFICATION_PROMPT",
            fallback_to_file=True,
        )
        self.prompt_version = prompt_version
        return prompt_content

    def _build_input_message(self, drug: str, abstract_title: str) -> str:
        """Build the input message with abstract title and drug.

        Args:
            drug: The drug name
            abstract_title: Abstract title for context

        Returns:
            str: Formatted input message
        """
        return f"""Abstract Title: {abstract_title or "Not provided"}
Drug: {drug}"""

    def invoke(
        self,
        drug: str,
        abstract_title: str = "",
        abstract_id: str = None,
    ) -> dict:
        """Invoke the regimen identification agent.

        Args:
            drug: The drug name to identify components for
            abstract_title: The abstract title for context (optional)
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

        # Build the input message
        input_message = self._build_input_message(drug=drug, abstract_title=abstract_title)

        messages = [
            SystemMessage(content=self.system_prompt),
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
                    AIMessage(content=f"Error: {str(e)}")
                ],
                "llm_calls": 1,
            }

    def parse_response(self, result: dict) -> Dict[str, Any]:
        """Parse the agent's response to extract components.

        Args:
            result: Agent invocation result containing messages

        Returns:
            Dictionary with extracted components
        """
        try:
            messages = result.get('messages', [])
            if not messages:
                return self._default_response()

            final_message = messages[-1]
            content = getattr(final_message, 'content', '')

            if not content or content.startswith("Error:"):
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

        Args:
            parsed: Parsed JSON dictionary

        Returns:
            Dictionary with components
        """
        components = parsed.get('components', [])
        
        # Ensure components is a list
        if not isinstance(components, list):
            components = [components] if components else []

        return {
            'components': components,
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
            'components': [],
            'success': False,
        }
        if error:
            response['error'] = error
        return response

