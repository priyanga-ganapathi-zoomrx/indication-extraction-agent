"""Calculator Agent using LangGraph.

This module implements a calculator agent using the LangGraph framework with proper
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
from src.tools import get_calculator_tools


class MessagesState(TypedDict):
    """State schema for the calculator agent.

    Attributes:
        messages: List of messages in the conversation, using operator.add to append
        llm_calls: Counter for the number of LLM calls made
    """

    messages: Annotated[list[BaseMessage], operator.add]
    llm_calls: int


class CalculatorAgent:
    """Calculator Agent that performs arithmetic operations using LLM and tools.

    This agent uses LangGraph to create a stateful conversation flow that can:
    - Understand user requests for arithmetic operations
    - Call appropriate tools (add, multiply, divide)
    - Handle errors gracefully
    - Trace all operations with Langfuse
    """

    def __init__(self, agent_name: str = "CalculatorAgent"):
        """Initialize the Calculator Agent.

        Args:
            agent_name: Name of the agent for identification and logging
        """
        self.agent_name = agent_name
        self.tools = get_calculator_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        # Initialize Langfuse
        self.langfuse_config = get_langfuse_config()
        self.langfuse = self._initialize_langfuse()

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
        """Get the system prompt for the calculator agent.

        Returns:
            str: System prompt content
        """
        return """You are a helpful assistant tasked with performing arithmetic operations.

When the user asks you to perform calculations:
1. Use the available tools (add, multiply, divide) to compute the result
2. Show your work step by step if multiple operations are needed
3. Always provide clear, accurate answers

Available tools:
- add(a, b): Adds two numbers
- multiply(a, b): Multiplies two numbers
- divide(a, b): Divides the first number by the second

Important: Always use tools to perform calculations. Do not compute answers directly."""

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the calculator agent.

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

    def invoke(self, user_message: str) -> dict:
        """Invoke the calculator agent with a user message.

        Args:
            user_message: The user's input message

        Returns:
            dict: Final state containing all messages and metadata
        """
        from langchain_core.messages import HumanMessage
        import os

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "llm_calls": 0,
        }

        # Set environment variables for Langfuse callback handler
        if self.langfuse:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_config.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_config.secret_key
            os.environ["LANGFUSE_HOST"] = self.langfuse_config.host

        # Configure with Langfuse tracing
        config = RunnableConfig(
            recursion_limit=100,
            callbacks=(
                [CallbackHandler()]
                if self.langfuse
                else []
            ),
        )

        # Invoke the graph
        try:
            result = self.graph.invoke(initial_state, config)
            return result
        except Exception as e:
            print(f"✗ Error during agent invocation: {e}")
            return {
                "messages": [
                    HumanMessage(content=user_message),
                    AIMessage(
                        content=f"I encountered an error: {str(e)}. Please try again."
                    ),
                ],
                "llm_calls": initial_state.get("llm_calls", 0),
            }

    def visualize(self, output_path: str = "calculator_agent_graph.png"):
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

