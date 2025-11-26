"""Drug Extraction Agent using LangGraph.

This module implements a drug extraction agent using the LangGraph framework.
The agent processes abstract titles through two LLM calls:
1. Extraction: Extracts drugs from the abstract title
2. Validation: Validates extracted drugs for therapeutic relevance
"""

import operator
from typing import Annotated, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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
        extracted_drugs_json: JSON string containing extracted drugs from first LLM call
        abstract_title: The original abstract title for reference in validation
    """

    messages: Annotated[list[BaseMessage], operator.add]
    llm_calls: int
    extracted_drugs_json: str
    abstract_title: str


class DrugExtractionAgent:
    """Drug Extraction Agent that extracts and validates drugs using LLM.

    This agent uses LangGraph to create a stateful conversation flow that:
    - Extracts drugs from research abstract titles (first LLM call)
    - Validates extracted drugs for therapeutic relevance (second LLM call)
    - Classifies drugs as Primary, Secondary, or Comparator
    - Returns structured JSON output with validated drugs
    - Traces all operations with Langfuse (single trace for both calls)
    """

    def __init__(
        self,
        agent_name: str = "DrugExtractionAgent",
        validation_model: Optional[str] = None,
        validation_temperature: Optional[float] = None,
        validation_max_tokens: Optional[int] = None,
    ):
        """Initialize the Drug Extraction Agent.

        Args:
            agent_name: Name of the agent for identification and logging
            validation_model: Optional model name for validation LLM call (uses default if not specified)
            validation_temperature: Optional temperature for validation LLM call (uses default if not specified)
            validation_max_tokens: Optional max_tokens for validation LLM call (uses default if not specified)
        """
        self.agent_name = agent_name
        self._validation_model = validation_model
        self._validation_temperature = validation_temperature
        self._validation_max_tokens = validation_max_tokens

        # Initialize Langfuse
        self.langfuse_config = get_langfuse_config()
        self.langfuse = self._initialize_langfuse() if self.langfuse_config else None

        # Initialize LLM for extraction (uses default settings)
        self.extraction_llm_config = self._get_extraction_llm_config()
        self.extraction_llm = create_llm(self.extraction_llm_config)

        # Initialize LLM for validation (can use different model)
        self.validation_llm_config = self._get_validation_llm_config()
        self.validation_llm = create_llm(self.validation_llm_config)

        # System prompts for extraction and validation
        self.extraction_system_prompt = self._get_extraction_system_prompt()
        self.validation_system_prompt = self._get_validation_system_prompt()

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

    def _get_extraction_llm_config(self) -> LLMConfig:
        """Get LLM configuration for extraction from default settings.

        Returns:
            LLMConfig: Configuration for the extraction language model
        """
        return LLMConfig(
            api_key=settings.llm.LLM_API_KEY,
            model=settings.llm.LLM_MODEL,
            base_url=settings.llm.LLM_BASE_URL,
            temperature=settings.llm.LLM_TEMPERATURE,
            max_tokens=settings.llm.LLM_MAX_TOKENS,
            name=f"{self.agent_name}_extraction",
        )

    def _get_validation_llm_config(self) -> LLMConfig:
        """Get LLM configuration for validation with optional overrides.

        If validation_model is specified, uses that model for validation.
        Otherwise, falls back to default settings.

        Returns:
            LLMConfig: Configuration for the validation language model
        """
        return LLMConfig(
            api_key=settings.llm.LLM_API_KEY,
            model=self._validation_model or settings.llm.LLM_MODEL,
            base_url=settings.llm.LLM_BASE_URL,
            temperature=self._validation_temperature if self._validation_temperature is not None else settings.llm.LLM_TEMPERATURE,
            max_tokens=self._validation_max_tokens or settings.llm.LLM_MAX_TOKENS,
            name=f"{self.agent_name}_validation",
        )

    def _get_extraction_system_prompt(self) -> str:
        """Get the system prompt for drug extraction.

        Fetches the prompt from Langfuse if configured, otherwise falls back to local file.

        Returns:
            str: System prompt content for extraction
        """
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name="DRUG_EXTRACTION_SYSTEM_PROMPT",
            fallback_to_file=True,
        )
        # Store the prompt version for tagging
        self.extraction_prompt_version = prompt_version
        return prompt_content

    def _get_validation_system_prompt(self) -> str:
        """Get the system prompt for drug validation.

        Fetches the prompt from Langfuse if configured, otherwise falls back to local file.

        Returns:
            str: System prompt content for validation
        """
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name="DRUG_VALIDATION_SYSTEM_PROMPT",
            fallback_to_file=True,
        )
        # Store the prompt version for tagging
        self.validation_prompt_version = prompt_version
        return prompt_content

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the drug extraction agent.

        The graph has two sequential LLM calls:
        - START -> extraction_llm_call -> validation_llm_call -> END

        Returns:
            StateGraph: Compiled state graph ready for execution
        """
        # Create the state graph
        graph = StateGraph(MessagesState)

        # Add nodes for extraction and validation
        graph.add_node("extraction_llm_call", self._extraction_llm_call_node)
        graph.add_node("validation_llm_call", self._validation_llm_call_node)

        # Add edges for sequential flow
        graph.add_edge(START, "extraction_llm_call")
        graph.add_edge("extraction_llm_call", "validation_llm_call")
        graph.add_edge("validation_llm_call", END)

        # Compile the graph
        return graph.compile()

    def _extraction_llm_call_node(self, state: MessagesState) -> dict:
        """Extraction LLM node that extracts drugs from abstract title.

        This node handles:
        - Adding the extraction system prompt
        - Invoking the LLM to extract drugs
        - Storing extracted JSON in state for validation step

        Args:
            state: Current state containing messages and llm_calls counter

        Returns:
            dict: Updated state with extraction response and extracted_drugs_json
        """
        messages_for_llm = [SystemMessage(content=self.extraction_system_prompt)] + state.get(
            "messages", []
        )

        try:
            response: AIMessage = self.extraction_llm.invoke(messages_for_llm)

            # Ensure the response has content to prevent parsing errors
            if not response.content:
                response.content = '{"Primary Drugs": [], "Secondary Drugs": [], "Comparator Drugs": []}'

            return {
                "messages": [response],
                "llm_calls": state.get("llm_calls", 0) + 1,
                "extracted_drugs_json": response.content,
            }
        except Exception as e:
            print(f"✗ Error during extraction LLM call: {e}")
            # Return an error message with empty extraction
            error_content = '{"Primary Drugs": [], "Secondary Drugs": [], "Comparator Drugs": []}'
            error_message = AIMessage(content=error_content)
            return {
                "messages": [error_message],
                "llm_calls": state.get("llm_calls", 0) + 1,
                "extracted_drugs_json": error_content,
            }

    def _validation_llm_call_node(self, state: MessagesState) -> dict:
        """Validation LLM node that validates extracted drugs.

        This node handles:
        - Formatting input with abstract title and extracted JSON
        - Adding the validation system prompt
        - Invoking the LLM to validate drugs
        - Returning the validated drug list

        Args:
            state: Current state containing extracted_drugs_json and abstract_title

        Returns:
            dict: Updated state with validation response
        """
        # Get the abstract title and extracted JSON from state
        abstract_title = state.get("abstract_title", "")
        extracted_json = state.get("extracted_drugs_json", "{}")

        # Format the validation input
        validation_input = f"""Validate the extracted drugs for the following:

**Title:** {abstract_title}

**Extracted JSON:**
{extracted_json}"""

        # Create messages for validation LLM call
        messages_for_llm = [
            SystemMessage(content=self.validation_system_prompt),
            HumanMessage(content=validation_input),
        ]

        try:
            response: AIMessage = self.validation_llm.invoke(messages_for_llm)

            # Ensure the response has content to prevent parsing errors
            if not response.content:
                response.content = '{"Primary Drugs": [], "Secondary Drugs": [], "Comparator Drugs": [], "Removed Drugs": [], "Reasoning": []}'

            return {
                "messages": [response],
                "llm_calls": state.get("llm_calls", 0) + 1,
            }
        except Exception as e:
            print(f"✗ Error during validation LLM call: {e}")
            # Return an error message
            error_message = AIMessage(
                content=f'{{"Primary Drugs": [], "Secondary Drugs": [], "Comparator Drugs": [], "Removed Drugs": [], "Reasoning": ["Error during validation: {str(e)}"]}}'
            )
            return {
                "messages": [error_message],
                "llm_calls": state.get("llm_calls", 0) + 1,
            }

    def invoke(self, abstract_title: str, session_title: str = "", abstract_id: str = None) -> dict:
        """Invoke the drug extraction agent with abstract title.

        This method runs both extraction and validation steps, returning only
        the final validated response.

        Args:
            abstract_title: The abstract title to extract drugs from
            session_title: Kept for backward compatibility but not used
            abstract_id: The abstract ID for tracking in Langfuse (optional)

        Returns:
            dict: Final state containing validated drugs response
        """
        import os

        # Build tags for Langfuse tracing
        tags = [
            f"abstract_id:{abstract_id or 'unknown'}",
            f"extraction_prompt_version:{getattr(self, 'extraction_prompt_version', 'unknown')}",
            f"validation_prompt_version:{getattr(self, 'validation_prompt_version', 'unknown')}",
            f"extraction_model:{self.extraction_llm_config.model}",
            f"validation_model:{self.validation_llm_config.model}",
        ]

        # Format the input message with the abstract title
        input_content = f"Extract drugs from the following:\n\nabstract_title: {abstract_title}"

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=input_content)],
            "llm_calls": 0,
            "extracted_drugs_json": "",
            "abstract_title": abstract_title,
        }

        # Set environment variables for Langfuse callback handler
        if self.langfuse:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_config.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_config.secret_key
            os.environ["LANGFUSE_HOST"] = self.langfuse_config.host

        # Configure with Langfuse tracing and tags (single trace for both LLM calls)
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
                "extracted_drugs_json": "",
                "abstract_title": abstract_title,
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
