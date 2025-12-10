"""Drug Class Extraction Agent using LangGraph ReAct Pattern.

This module implements a drug class extraction agent using the LangGraph framework with
ReAct pattern (Reasoning + Acting). The agent uses tool calling for rule retrieval,
following the same pattern as the indication extraction agent.

Key differences from the sequential drug_class_agent.py:
- Uses ReAct loop with conditional edges for tool calling
- LLM decides when to call the get_drug_class_rules tool
- Accepts pre-fetched/cached search results as input
"""

import operator
import os
import json
import re
from typing import Annotated, List, Literal, Dict, Any

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
from src.prompts import get_system_prompt
from src.rule_tool import get_drug_class_tools


class DrugClassMessagesState(TypedDict):
    """State schema for the drug class extraction agent.

    Attributes:
        messages: List of messages in the conversation, using operator.add to append
        llm_calls: Counter for the number of LLM calls made
    """

    messages: Annotated[list[BaseMessage], operator.add]
    llm_calls: int


class DrugClassReActAgent:
    """Drug Class Extraction Agent using ReAct pattern with tool calling.

    This agent uses LangGraph to create a stateful conversation flow that can:
    - Extract drug classes from search results, abstract title, and abstract text
    - Call the get_drug_class_rules tool to retrieve category-specific extraction rules
    - Handle errors gracefully
    - Trace all operations with Langfuse
    """

    def __init__(
        self,
        agent_name: str = "DrugClassReActAgent",
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        """Initialize the Drug Class ReAct Agent.

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

        # Get drug class tools
        self.tools = get_drug_class_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        # Initialize Langfuse
        self.langfuse_config = get_langfuse_config()
        self.langfuse = self._initialize_langfuse() if self.langfuse_config else None

        # Initialize LLM
        self.llm_config = self._get_llm_config()
        self.llm = create_llm(self.llm_config)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # System prompt
        self.system_prompt = self._get_system_prompt()

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

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the drug class extraction agent.

        Fetches the prompt from Langfuse if configured, otherwise falls back to local file.

        Returns:
            str: System prompt content
        """
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name="DRUG_CLASS_EXTRACTION_FROM_SEARCH_REACT_PATTERN",
            fallback_to_file=True,
        )
        # Store the prompt version for tagging
        self.prompt_version = prompt_version
        return prompt_content

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the drug class extraction agent.

        Returns:
            StateGraph: Compiled state graph ready for execution
        """
        # Create the state graph
        graph = StateGraph(DrugClassMessagesState)

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

    def _llm_call_node(self, state: DrugClassMessagesState) -> dict:
        """LLM node that decides whether to call a tool or respond.

        This node handles:
        - Adding the system prompt
        - Invoking the LLM with tools
        - Error handling for LLM calls
        - Ensuring messages have content to prevent parsing errors

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
                content=f"I encountered an error while processing your request: {str(e)}"
            )
            return {
                "messages": [error_message],
                "llm_calls": state.get("llm_calls", 0) + 1,
            }

    def _should_continue(self, state: DrugClassMessagesState) -> Literal["tool_node", END]:
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

    def _format_search_results_for_prompt(
        self,
        drug: str,
        drug_class_results: List[Dict],
        firm_results: List[Dict],
        abstract_title: str = "",
        full_abstract: str = ""
    ) -> str:
        """Format search results according to the prompt's INPUT specification.

        Args:
            drug: The drug name
            drug_class_results: Results from drug class search (cached)
            firm_results: Results from firm search (cached)
            abstract_title: Abstract title for context
            full_abstract: Full abstract text for context

        Returns:
            str: Formatted input string for the extraction prompt
        """
        all_results = (drug_class_results or []) + (firm_results or [])

        formatted_parts = [f"Drug: {drug}"]

        # Add abstract title if provided
        if abstract_title:
            formatted_parts.append(f"\nAbstract title: {abstract_title}")

        # Add full abstract if provided
        if full_abstract:
            abstract_text = full_abstract
            if len(abstract_text) > 10000:
                abstract_text = abstract_text[:10000] + "... [truncated]"
            formatted_parts.append(f"\nAbstract Text: {abstract_text}")

        # Add search results
        if not all_results:
            formatted_parts.append("\nNo search results available.")
        else:
            for i, result in enumerate(all_results, 1):
                content = result.get("raw_content") or result.get("content", "No content available")
                url = result.get("url", "Unknown URL")

                if len(content) > 5000:
                    content = content[:5000] + "... [truncated]"

                formatted_parts.append(f"\nExtracted Content {i}: {content}")
                formatted_parts.append(f"Content {i} URL: {url}")

        return "\n".join(formatted_parts)

    def invoke(
        self,
        drug: str,
        abstract_title: str = "",
        full_abstract: str = "",
        drug_class_results: List[Dict] = None,
        firm_results: List[Dict] = None,
        abstract_id: str = None
    ) -> dict:
        """Invoke the drug class extraction agent with drug info and search results.

        Args:
            drug: The drug name to extract class for
            abstract_title: The abstract title for context (optional)
            full_abstract: The full abstract text for context (optional)
            drug_class_results: Pre-fetched drug class search results (optional)
            firm_results: Pre-fetched firm search results (optional)
            abstract_id: The abstract ID for tracking in Langfuse (optional)

        Returns:
            dict: Final state containing all messages and metadata
        """
        # Build tags for Langfuse tracing
        tags = [
            drug,  # Drug name as tag
            f"prompt_version:{getattr(self, 'prompt_version', 'unknown')}",
            f"model:{self.llm_config.model}",
        ]
        if abstract_id:
            tags.append(f"abstract_id:{abstract_id}")

        # Format the input message with the drug info and search results
        input_content = self._format_search_results_for_prompt(
            drug=drug,
            drug_class_results=drug_class_results or [],
            firm_results=firm_results or [],
            abstract_title=abstract_title,
            full_abstract=full_abstract,
        )

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=input_content)],
            "llm_calls": 0,
        }

        # Set environment variables for Langfuse callback handler
        if self.langfuse:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_config.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_config.secret_key
            os.environ["LANGFUSE_HOST"] = self.langfuse_config.host

        # Configure with Langfuse tracing and tags
        config = RunnableConfig(
            recursion_limit=100,
            callbacks=(
                [CallbackHandler()]
                if self.langfuse
                else []
            ),
            metadata={"langfuse_tags": tags} if self.langfuse else {},
        )

        # Invoke the graph
        try:
            result = self.graph.invoke(initial_state, config)
            return result
        except Exception as e:
            print(f"✗ Error during agent invocation: {e}")
            return {
                "messages": [
                    HumanMessage(content=input_content),
                    AIMessage(
                        content=f"I encountered an error during drug class extraction: {str(e)}. Please try again."
                    ),
                ],
                "llm_calls": initial_state.get("llm_calls", 0),
            }

    def parse_response(self, result: dict) -> Dict[str, Any]:
        """Parse the agent's response to extract structured drug class data.

        Args:
            result: Agent invocation result containing messages

        Returns:
            Dictionary with extracted fields matching the prompt's output format
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
            'rules_retrieved': parsed.get('rules_retrieved', []),
            'components_identified': parsed.get('components_identified', []),
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
            'rules_retrieved': [],
            'components_identified': [],
            'quality_metrics_completeness': None,
            'quality_metrics_rule_adherence': None,
            'quality_metrics_clinical_accuracy': None,
            'quality_metrics_formatting_compliance': None,
            'success': False,
        }
        if error:
            response['error'] = error
        return response

    def visualize(self, output_path: str = "drug_class_react_agent_graph.png"):
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

