"""Indication Extraction Validation Agent using LangGraph.

This module implements a validation agent that verifies indication extractions
against established rules, flagging potential errors for manual QC review.
"""

import json
import operator
import os
import re
from typing import Annotated, Any, Dict, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.config import settings
from src.langfuse_config import get_langfuse_config
from src.llm_handler import LLMConfig, create_llm
from src.prompts import get_system_prompt, EXTRACTION_PROMPT_NAME
from src.tools import get_tools


class ValidationState(TypedDict):
    """State schema for the indication validation agent.

    Attributes:
        messages: List of messages in the conversation, using operator.add to append
        llm_calls: Counter for the number of LLM calls made
    """

    messages: Annotated[list[BaseMessage], operator.add]
    llm_calls: int


class IndicationValidationAgent:
    """Indication Validation Agent that validates extraction results against rules.

    This agent uses LangGraph to verify:
    - Source selection correctness
    - Component grounding (hallucination check)
    - Component completeness (omission check)
    - Rule application verification
    - Exclusion compliance
    - Formatting compliance
    """

    def __init__(self, agent_name: str = "IndicationValidationAgent", llm_model: str | None = None):
        """Initialize the Indication Validation Agent.

        Args:
            agent_name: Name of the agent for identification and logging
            llm_model: Optional override for the LLM model name
        """
        self.agent_name = agent_name
        self._llm_model = llm_model
        self.tools = get_tools()  # Same tools as extraction agent
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        # Initialize Langfuse
        self.langfuse_config = get_langfuse_config()
        self.langfuse = self._initialize_langfuse() if self.langfuse_config else None
        self.langfuse_callback = CallbackHandler() if self.langfuse else None

        # Initialize LLM
        self.llm_config = self._get_llm_config()
        self.llm = create_llm(self.llm_config)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # System prompt (validation instructions)
        self.system_prompt = self._get_system_prompt()
        
        # Reference prompt (extraction rules - loaded dynamically)
        self.extraction_rules_prompt = self._get_extraction_rules_prompt()

        # Build the graph
        self.graph = self._build_graph()

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
        """Get LLM configuration from settings.

        Returns:
            LLMConfig: Configuration for the language model
        """
        return LLMConfig(
            api_key=settings.llm.LLM_API_KEY,
            model=self._llm_model or settings.llm.LLM_MODEL,
            base_url=settings.llm.LLM_BASE_URL,
            temperature=settings.llm.LLM_TEMPERATURE,
            max_tokens=settings.llm.LLM_MAX_TOKENS,
            name=self.agent_name,
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the indication validation agent.

        Fetches the prompt from Langfuse if configured, otherwise falls back to local file.

        Returns:
            str: System prompt content
        """
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name="INDICATION_VALIDATION_SYSTEM_PROMPT",
            fallback_to_file=True,
        )
        # Store the prompt version for tagging
        self.prompt_version = prompt_version
        return prompt_content

    def _get_extraction_rules_prompt(self) -> str:
        """Get the extraction rules prompt to use as reference for validation.

        Dynamically loads the MEDICAL_INDICATION_EXTRACTION_SYSTEM_PROMPT to provide
        the validator with the rules the extractor was instructed to follow.

        Returns:
            str: Extraction rules prompt content
        """
        try:
            prompt_content, prompt_version = get_system_prompt(
                langfuse_client=self.langfuse,
                prompt_name=EXTRACTION_PROMPT_NAME,
                fallback_to_file=True,
            )
            # Store version for reference
            self.extraction_rules_version = prompt_version
            print(f"✓ Loaded extraction rules prompt (version: {prompt_version})")
            return prompt_content
        except Exception as e:
            print(f"✗ Error loading extraction rules prompt: {e}")
            self.extraction_rules_version = "error"
            return "Error loading extraction rules. Please verify rule application manually."

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the validation agent.

        Returns:
            StateGraph: Compiled state graph ready for execution
        """
        # Create the state graph
        graph = StateGraph(ValidationState)

        # Add nodes
        graph.add_node("llm_call", self._llm_call_node)
        graph.add_node("tool_node", ToolNode(self.tools))

        # Add edges
        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges(
            "llm_call", self._should_continue, ["tool_node", END]
        )
        graph.add_edge("tool_node", "llm_call")

        # Compile the graph
        return graph.compile()

    def _llm_call_node(self, state: ValidationState) -> dict:
        """LLM node that decides whether to call a tool or respond.

        Args:
            state: Current state containing messages and llm_calls counter

        Returns:
            dict: Updated state with new message and incremented llm_calls
        """
        messages_for_llm = [SystemMessage(content=self.system_prompt)] + state.get(
            "messages", []
        )

        try:
            response: AIMessage = self.llm_with_tools.invoke(messages_for_llm)

            # Ensure the response has content to prevent parsing errors
            if not response.content and not response.tool_calls:
                response.content = "[Thinking...]"

            return {
                "messages": [response],
                "llm_calls": state.get("llm_calls", 0) + 1,
            }
        except Exception as e:
            print(f"✗ Error during LLM call: {e}")
            # Return an error message instead of crashing
            error_message = AIMessage(
                content=f"I encountered an error while validating: {str(e)}"
            )
            return {
                "messages": [error_message],
                "llm_calls": state.get("llm_calls", 0) + 1,
            }

    def _should_continue(self, state: ValidationState) -> Literal["tool_node", END]:
        """Determine whether to continue to tool execution or end.

        Args:
            state: Current state containing messages

        Returns:
            str: Next node to execute ("tool_node" or END)
        """
        messages = state.get("messages", [])
        if not messages:
            return END

        last_message = messages[-1]

        # If the LLM makes a tool call, route to the tool node
        if (
            isinstance(last_message, AIMessage)
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        ):
            return "tool_node"

        # Otherwise, we're done
        return END

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
        input_content = f"""## Validation Input

### Original Titles
- **session_title**: {session_title}
- **abstract_title**: {abstract_title}

### Extraction Result to Validate
- **generated_indication**: {extraction_result.get('indication', '')}
- **selected_source**: {extraction_result.get('selected_source', '')}
- **confidence_score**: {extraction_result.get('confidence_score', 'N/A')}
- **reasoning**: {extraction_result.get('reasoning', '')}
- **rules_retrieved**: {json.dumps(extraction_result.get('rules_retrieved', []), indent=2)}
- **components_identified**: {json.dumps(extraction_result.get('components_identified', []), indent=2)}

Please perform all 6 validation checks and return your validation result in the specified JSON format."""

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
        import os

        # Build tags for Langfuse tracing
        tags = [
            f"abstract_id:{abstract_id or 'unknown'}",
            f"validation_prompt_version:{getattr(self, 'prompt_version', 'unknown')}",
            f"extraction_rules_version:{getattr(self, 'extraction_rules_version', 'unknown')}",
            f"model:{self.llm_config.model}",
            "validation",
        ]

        # Format the reference rules message (second message with extraction prompt)
        reference_rules_content = f"""## REFERENCE RULES DOCUMENT

The following is the complete extraction rules document that the extractor was instructed to follow. Use this as your reference to verify compliance.

---

{self.extraction_rules_prompt}

---

END OF REFERENCE RULES DOCUMENT"""

        # Format the validation input message
        input_content = self._format_validation_input(
            session_title, abstract_title, extraction_result
        )

        # Create initial state with two messages:
        # 1. Reference rules document
        # 2. Validation input to check
        initial_state = {
            "messages": [
                HumanMessage(content=reference_rules_content),
                HumanMessage(content=input_content),
            ],
            "llm_calls": 0,
        }

        # Set environment variables for Langfuse callback handler
        if self.langfuse_config:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_config.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_config.secret_key
            os.environ["LANGFUSE_HOST"] = self.langfuse_config.host

        # Configure with Langfuse tracing and tags
        config = RunnableConfig(
            recursion_limit=100,
            callbacks=(
                [self.langfuse_callback]
                if self.langfuse_callback
                else []
            ),
            metadata={"langfuse_tags": tags} if self.langfuse else {},
        )

        # Invoke the graph
        try:
            result = self.graph.invoke(initial_state, config)
            return self._parse_validation_response(result)
        except Exception as e:
            print(f"✗ Error during validation invocation: {e}")
            return {
                "validation_status": "REVIEW",
                "validation_confidence": 0.0,
                "issues_found": [
                    {
                        "check_type": "system_error",
                        "severity": "high",
                        "description": f"Validation failed due to system error: {str(e)}",
                        "evidence": "",
                        "component": "",
                    }
                ],
                "checks_performed": {},
                "validation_reasoning": f"Validation could not be completed due to error: {str(e)}",
                "llm_calls": initial_state.get("llm_calls", 0),
            }

    def _parse_validation_response(self, result: Dict) -> Dict[str, Any]:
        """Parse the validation response from the agent.

        Args:
            result: Agent invocation result

        Returns:
            dict: Parsed validation result
        """
        try:
            # Get the final message from the agent
            messages = result.get("messages", [])
            if not messages:
                return self._default_validation_response("No response from validator")

            final_message = messages[-1]
            content = getattr(final_message, "content", "")

            if not content or content.startswith("I encountered an error"):
                return self._default_validation_response(
                    f"Validation error: {content}"
                )

            # Try to parse JSON response
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    # Ensure required fields exist
                    return {
                        "validation_status": parsed.get("validation_status", "REVIEW"),
                        "validation_confidence": parsed.get("validation_confidence", 0.5),
                        "issues_found": parsed.get("issues_found", []),
                        "checks_performed": parsed.get("checks_performed", {}),
                        "validation_reasoning": parsed.get("validation_reasoning", ""),
                        "llm_calls": result.get("llm_calls", 0),
                    }
                except json.JSONDecodeError as e:
                    return self._default_validation_response(
                        f"Failed to parse JSON response: {e}"
                    )

            # Try to find JSON without code blocks
            try:
                # Look for JSON object pattern
                json_pattern = r'\{[^{}]*"validation_status"[^{}]*\}'
                simple_match = re.search(json_pattern, content, re.DOTALL)
                if simple_match:
                    parsed = json.loads(simple_match.group(0))
                    return {
                        "validation_status": parsed.get("validation_status", "REVIEW"),
                        "validation_confidence": parsed.get("validation_confidence", 0.5),
                        "issues_found": parsed.get("issues_found", []),
                        "checks_performed": parsed.get("checks_performed", {}),
                        "validation_reasoning": parsed.get("validation_reasoning", ""),
                        "llm_calls": result.get("llm_calls", 0),
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
                "validation_reasoning": content[:1000] if content else "Unable to parse validation response",
                "llm_calls": result.get("llm_calls", 0),
            }

        except Exception as e:
            print(f"Error parsing validation response: {e}")
            return self._default_validation_response(f"Parse error: {e}")

    def _default_validation_response(self, reason: str) -> Dict[str, Any]:
        """Create a default validation response for error cases.

        Args:
            reason: Reason for the default response

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
            "llm_calls": 0,
        }

    def visualize(self, output_path: str = "indication_validation_agent_graph.png"):
        """Visualize the agent's graph structure.

        Args:
            output_path: Path to save the graph visualization
        """
        try:
            from IPython.display import Image

            graph_image = self.graph.get_graph(xray=True).draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(graph_image)
            print(f"✓ Graph visualization saved to {output_path}")
            return Image(graph_image)
        except Exception as e:
            print(f"✗ Error visualizing graph: {e}")
            print("Note: Graph visualization requires graphviz to be installed.")
            return None

