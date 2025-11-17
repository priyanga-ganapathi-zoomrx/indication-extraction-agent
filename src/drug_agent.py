"""Drug Extraction Agent using LangGraph.

This module implements a drug extraction agent using the LangGraph framework.
Unlike the indication extraction agent, this agent does not use tools and simply
processes the abstract title with the LLM to extract drugs in a structured JSON format.
"""

import operator
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.config import settings
from src.langfuse_config import get_langfuse_config
from src.llm_handler import LLMConfig, create_llm
from src.prompts import get_system_prompt


class MessagesState(TypedDict):
    """State schema for the drug extraction agent.

    Attributes:
        messages: List of messages in the conversation, using operator.add to append
        llm_calls: Counter for the number of LLM calls made
    """

    messages: Annotated[list[BaseMessage], operator.add]
    llm_calls: int


class DrugExtractionAgent:
    """Drug Extraction Agent that extracts drugs using LLM without tools.

    This agent uses LangGraph to create a simplified stateful conversation flow that:
    - Extracts drugs from research abstract titles
    - Classifies drugs as Primary, Secondary, or Comparator
    - Returns structured JSON output
    - Traces all operations with Langfuse
    - Does not require any tools (unlike the indication extraction agent)
    """

    def __init__(self, agent_name: str = "DrugExtractionAgent"):
        """Initialize the Drug Extraction Agent.

        Args:
            agent_name: Name of the agent for identification and logging
        """
        self.agent_name = agent_name

        # Initialize Langfuse
        self.langfuse_config = get_langfuse_config()
        self.langfuse = self._initialize_langfuse() if self.langfuse_config else None

        # Initialize LLM (without tools)
        self.llm_config = self._get_llm_config()
        self.llm = create_llm(self.llm_config)

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
        """Get the system prompt for the drug extraction agent.

        Fetches the prompt from Langfuse if configured, otherwise falls back to local file.

        Returns:
            str: System prompt content
        """
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name="DRUG_EXTRACTION_SYSTEM_PROMPT",
            fallback_to_file=True,
        )
        # Store the prompt version for tagging
        self.prompt_version = prompt_version
        return prompt_content

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the drug extraction agent.

        This is a simplified graph compared to the indication extraction agent:
        - No tool node (drugs are extracted directly by LLM)
        - Direct path: START -> llm_call -> END

        Returns:
            StateGraph: Compiled state graph ready for execution
        """
        # Create the state graph
        graph = StateGraph(MessagesState)

        # Add single node for LLM call
        graph.add_node("llm_call", self._llm_call_node)

        # Add direct edges (no conditional routing)
        graph.add_edge(START, "llm_call")
        graph.add_edge("llm_call", END)

        # Compile the graph
        return graph.compile()

    def _llm_call_node(self, state: MessagesState) -> dict:
        """LLM node that extracts drugs and returns structured JSON.

        This node handles:
        - Adding the system prompt
        - Invoking the LLM (without tools)
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
            response: AIMessage = self.llm.invoke(messages_for_llm)

            # Ensure the response has content to prevent parsing errors
            if not response.content:
                response.content = '{"Primary Drugs": [], "Secondary Drugs": [], "Comparator Drugs": []}'

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

    def invoke(self, abstract_title: str, session_title: str = "", abstract_id: str = None) -> dict:
        """Invoke the drug extraction agent with abstract title.

        Args:
            abstract_title: The abstract title to extract drugs from
            session_title: Kept for backward compatibility but not used
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

        # Format the input message with the abstract title
        input_content = f"Extract drugs from the following:\n\nabstract_title: {abstract_title}"

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
                        content=f"I encountered an error during drug extraction: {str(e)}. Please try again."
                    ),
                ],
                "llm_calls": initial_state.get("llm_calls", 0),
            }

    def visualize(self, output_path: str = "drug_extraction_agent_graph.png"):
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

