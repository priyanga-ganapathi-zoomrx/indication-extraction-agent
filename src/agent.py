"""Indication Extraction Agent using LangGraph.

This module implements an indication extraction agent using the LangGraph framework with proper
error handling for LLM output, parsing, and tool calling, following patterns from
the galen-fastapi-server repository.
"""

import operator
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
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
from src.tools import get_tools


class MessagesState(TypedDict):
    """State schema for the indication extraction agent.

    Attributes:
        messages: List of messages in the conversation, using operator.add to append
        llm_calls: Counter for the number of LLM calls made
    """

    messages: Annotated[list[BaseMessage], operator.add]
    llm_calls: int


class IndicationExtractionAgent:
    """Indication Extraction Agent that extracts medical indications using LLM and tools.

    This agent uses LangGraph to create a stateful conversation flow that can:
    - Extract medical indications from research abstract and session titles
    - Call appropriate tools to retrieve clinical rules
    - Handle errors gracefully
    - Trace all operations with Langfuse
    """

    def __init__(self, agent_name: str = "IndicationExtractionAgent", enable_caching: bool = False):
        """Initialize the Indication Extraction Agent.

        Args:
            agent_name: Name of the agent for identification and logging
            enable_caching: Enable prompt caching for Gemini models (reduces costs)
        """
        self.agent_name = agent_name
        self.enable_caching = enable_caching
        self.tools = get_tools()
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
        """Get LLM configuration from settings.

        Returns:
            LLMConfig: Configuration for the language model
        """
        return LLMConfig(
            api_key=settings.llm.LLM_API_KEY,
            model=settings.llm.LLM_MODEL,
            base_url=settings.llm.LLM_BASE_URL,
            temperature=settings.llm.LLM_TEMPERATURE,
            max_tokens=settings.llm.LLM_MAX_TOKENS,
            name=self.agent_name,
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the indication extraction agent.

        Fetches the prompt from Langfuse if configured, otherwise falls back to local file.

        Returns:
            str: System prompt content
        """
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name="MEDICAL_INDICATION_EXTRACTION_SYSTEM_PROMPT",
            fallback_to_file=True,
        )
        # Store the prompt version for tagging
        self.prompt_version = prompt_version
        return prompt_content

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the indication extraction agent.

        Returns:
            StateGraph: Compiled state graph ready for execution
        """
        # Create the state graph
        graph = StateGraph(MessagesState)

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

    def _llm_call_node(self, state: MessagesState) -> dict:
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
        # Format system message with cache_control if caching is enabled
        if self.enable_caching:
            system_message = SystemMessage(content=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ])
        else:
            system_message = SystemMessage(content=self.system_prompt)

        messages_for_llm = [system_message] + state.get("messages", [])

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

    def _should_continue(self, state: MessagesState) -> Literal["tool_node", END]:
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

    def invoke(self, abstract_title: str, session_title: str = "", abstract_id: str = None) -> dict:
        """Invoke the indication extraction agent with abstract and session titles.

        Args:
            abstract_title: The abstract title to extract indication from
            session_title: The session title (optional)
            abstract_id: The abstract ID for tracking in Langfuse (optional)

        Returns:
            dict: Final state containing all messages and metadata
        """
        from langchain_core.messages import HumanMessage
        import os

        # Build tags for Langfuse tracing
        tags = [
            f"abstract_id:{abstract_id or 'unknown'}",
            f"prompt_version:{getattr(self, 'prompt_version', 'unknown')}",
            f"model:{self.llm_config.model}",
        ]

        # Format the input message with the titles
        input_content = f"Extract the medical indication from the following:\n\nsession_title: {session_title}\nabstract_title: {abstract_title}"

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
                        content=f"I encountered an error during indication extraction: {str(e)}. Please try again."
                    ),
                ],
                "llm_calls": initial_state.get("llm_calls", 0),
            }

    def visualize(self, output_path: str = "indication_extraction_agent_graph.png"):
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

