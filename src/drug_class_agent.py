"""Drug Class Extraction Agent using LangGraph.

This module implements a drug class extraction agent using the LangGraph framework.
The agent processes drug names through the following steps:
1. Search for drug class information using Tavily (mechanism of action, chemical class, etc.)
2. Search for additional information using drug + firm name
3. Extract drug classes using GPT-4.1 with the search results
"""

import json
import os
import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.config import settings
from src.langfuse_config import get_langfuse_config
from src.llm_handler import LLMConfig, create_llm
from src.prompts import get_system_prompt


class DrugClassState(TypedDict):
    """State schema for the drug class extraction agent.

    Attributes:
        drug: The drug name to extract class for
        firm: List of pharmaceutical firm/company names
        abstract_title: The abstract title for context
        full_abstract: The full abstract text for context
        drug_class_search_results: Results from the first Tavily search (drug class info)
        firm_search_results: Results from the second Tavily search (drug + firm)
        drug_class_result: Final extraction result with drug classes
        llm_calls: Counter for the number of LLM calls made
    """

    drug: str
    firm: list  # List of firm names
    abstract_title: str
    full_abstract: str
    drug_class_search_results: list
    firm_search_results: list
    drug_class_result: dict
    llm_calls: int


class DrugClassAgent:
    """Drug Class Extraction Agent that extracts drug classes using Tavily search and LLM.

    This agent uses LangGraph to create a stateful flow that:
    - Searches for drug class information via Tavily (MoA, chemical class, etc.)
    - Searches for additional info using drug + firm name
    - Extracts drug classes using GPT-4.1 with the DRUG_CLASS_EXTRACTION_FROM_SEARCH prompt
    - Returns structured JSON output with drug classes and source URLs
    - Traces all operations with Langfuse (drug name as tag)
    """

    def __init__(
        self,
        agent_name: str = "DrugClassAgent",
        extraction_model: str = "gpt-4.1",
        extraction_temperature: float = 0.0,
        extraction_max_tokens: int = 4096,
    ):
        """Initialize the Drug Class Agent.

        Args:
            agent_name: Name of the agent for identification and logging
            extraction_model: Model to use for drug class extraction (default: gpt-4.1)
            extraction_temperature: Temperature for extraction LLM (default: 0.0)
            extraction_max_tokens: Max tokens for extraction LLM (default: 4096)
        """
        self.agent_name = agent_name
        self._extraction_model = extraction_model
        self._extraction_temperature = extraction_temperature
        self._extraction_max_tokens = extraction_max_tokens

        # Initialize Langfuse
        self.langfuse_config = get_langfuse_config()
        self.langfuse = self._initialize_langfuse() if self.langfuse_config else None

        # Initialize Tavily client
        self._tavily_client = self._initialize_tavily()

        # Initialize LLM for extraction
        self.extraction_llm_config = self._get_extraction_llm_config()
        self.extraction_llm = create_llm(self.extraction_llm_config)

        # System prompt for extraction
        self.extraction_system_prompt = self._get_extraction_system_prompt()

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

    def _initialize_tavily(self):
        """Initialize Tavily client for drug class searches.

        Returns:
            TavilyClient instance or None if initialization fails
        """
        try:
            from tavily import TavilyClient

            api_key = settings.tavily.TAVILY_API_KEY
            if not api_key:
                print(f"⚠ Tavily API key not configured. Searches will be skipped.")
                return None

            client = TavilyClient(api_key=api_key)
            print(f"✓ Tavily client initialized for {self.agent_name}")
            return client
        except ImportError:
            print(f"✗ tavily-python not installed. Run: pip install tavily-python")
            return None
        except Exception as e:
            print(f"✗ Error initializing Tavily client: {e}")
            return None

    def _get_extraction_llm_config(self) -> LLMConfig:
        """Get LLM configuration for extraction using GPT-4.1.

        Returns:
            LLMConfig: Configuration for the extraction language model
        """
        return LLMConfig(
            api_key=settings.llm.LLM_API_KEY,
            model=self._extraction_model,
            base_url=settings.llm.LLM_BASE_URL,
            temperature=self._extraction_temperature,
            max_tokens=self._extraction_max_tokens,
            name=f"{self.agent_name}_extraction",
        )

    def _get_extraction_system_prompt(self) -> str:
        """Get the system prompt for drug class extraction.

        Fetches the prompt from Langfuse if configured, otherwise falls back to local file.

        Returns:
            str: System prompt content for extraction
        """
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name="DRUG_CLASS_EXTRACTION_FROM_SEARCH",
            fallback_to_file=True,
        )
        # Store the prompt version for tagging
        self.extraction_prompt_version = prompt_version
        return prompt_content

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the drug class extraction agent.

        The graph structure:
        START -> drug_class_search -> firm_search -> extract_drug_class -> END

        Returns:
            StateGraph: Compiled state graph ready for execution
        """
        # Create the state graph
        graph = StateGraph(DrugClassState)

        # Add nodes
        graph.add_node("drug_class_search", self._drug_class_search_node)
        graph.add_node("firm_search", self._firm_search_node)
        graph.add_node("extract_drug_class", self._extract_drug_class_node)

        # Add edges for sequential flow
        graph.add_edge(START, "drug_class_search")
        graph.add_edge("drug_class_search", "firm_search")
        graph.add_edge("firm_search", "extract_drug_class")
        graph.add_edge("extract_drug_class", END)

        # Compile the graph
        return graph.compile()

    def _drug_class_search_node(self, state: DrugClassState) -> dict:
        """First Tavily search node for drug class information.

        Searches for mechanism of action, chemical class, pharmacologic class, etc.

        Query format: ("{{DRUG}}" AND ("Mechanism of Action" OR "12.1 Mechanism of Action" 
                       OR "MoA" OR "mode of action" OR "Pharmacologic Class" OR 
                       "Pharmacological Class" OR "Chemical Class" OR "Therapeutic Class" 
                       OR "Drug Class"))

        Config: search_depth="advanced", include_domains=["nih.gov", "fda.gov", "clinicaltrials.gov"],
                include_raw_content=True, max_results=3

        Args:
            state: Current state containing drug name

        Returns:
            dict: Updated state with drug_class_search_results
        """
        drug = state.get("drug", "")
        
        if not self._tavily_client or not drug:
            print(f"⚠ Skipping drug class search: {'No Tavily client' if not self._tavily_client else 'No drug name'}")
            return {"drug_class_search_results": []}

        # Build the search query
        query = (
            f'("{drug}" AND '
            f'("Mechanism of Action" OR "12.1 Mechanism of Action" OR "MoA" OR "mode of action" OR '
            f'"Pharmacologic Class" OR "Pharmacological Class" OR "Chemical Class" OR '
            f'"Therapeutic Class" OR "Drug Class"))'
        )

        try:
            print(f"🔍 Searching for drug class info: {drug}")
            response = self._tavily_client.search(
                query=query,
                search_depth="advanced",
                include_domains=["nih.gov", "fda.gov", "clinicaltrials.gov"],
                include_raw_content=True,
                max_results=3,
            )

            # Extract relevant information from results
            results = []
            for result in response.get("results", [])[:3]:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "raw_content": result.get("raw_content", ""),
                })

            print(f"✓ Found {len(results)} results for drug class search")
            return {"drug_class_search_results": results}

        except Exception as e:
            print(f"✗ Error during drug class search: {e}")
            return {"drug_class_search_results": []}

    def _firm_search_node(self, state: DrugClassState) -> dict:
        """Second Tavily search node for drug + firm information.

        Query format:
        - Single firm: ({drug} AND {firm})
        - Multiple firms: ({drug} AND ({firm1} OR {firm2} OR ...))

        Config: search_depth="advanced", include_raw_content=True, max_results=3

        Args:
            state: Current state containing drug and firm names (list)

        Returns:
            dict: Updated state with firm_search_results
        """
        drug = state.get("drug", "")
        firms = state.get("firm", [])

        if not self._tavily_client or not drug:
            print(f"⚠ Skipping firm search: {'No Tavily client' if not self._tavily_client else 'No drug name'}")
            return {"firm_search_results": []}

        # Build the search query based on number of firms
        if not firms or len(firms) == 0:
            query = drug
        elif len(firms) == 1:
            query = f'({drug} AND {firms[0]})'
        else:
            # Multiple firms: ({drug} AND ({firm1} OR {firm2} OR ...))
            firms_or = " OR ".join(firms)
            query = f'({drug} AND ({firms_or}))'

        try:
            print(f"🔍 Searching for drug + firm info: {query}")
            response = self._tavily_client.search(
                query=query,
                search_depth="advanced",
                include_raw_content=True,
                max_results=3,
            )

            # Extract relevant information from results
            results = []
            for result in response.get("results", [])[:3]:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "raw_content": result.get("raw_content", ""),
                })

            print(f"✓ Found {len(results)} results for firm search")
            return {"firm_search_results": results}

        except Exception as e:
            print(f"✗ Error during firm search: {e}")
            return {"firm_search_results": []}

    def _format_search_results_for_prompt(
        self,
        drug: str,
        drug_class_results: list,
        firm_results: list,
        abstract_title: str = "",
        full_abstract: str = ""
    ) -> str:
        """Format search results according to the prompt's INPUT specification.

        Args:
            drug: The drug name
            drug_class_results: Results from drug class search
            firm_results: Results from firm search
            abstract_title: The abstract title for context
            full_abstract: The full abstract text for context

        Returns:
            str: Formatted input string for the extraction prompt
        """
        # Combine all results
        all_results = drug_class_results + firm_results

        # Format the input
        formatted_parts = [f"Drug: {drug}"]

        # Add abstract title if provided
        if abstract_title:
            formatted_parts.append(f"\nAbstract title: {abstract_title}")

        # Add full abstract if provided
        if full_abstract:
            # Truncate very long abstracts
            abstract_text = full_abstract
            if len(abstract_text) > 10000:
                abstract_text = abstract_text[:10000] + "... [truncated]"
            formatted_parts.append(f"\nFull Abstract Text: {abstract_text}")

        # Add search results
        if not all_results:
            formatted_parts.append("\nNo search results available.")
        else:
            for i, result in enumerate(all_results, 1):
                # Use raw_content if available, otherwise fall back to content
                content = result.get("raw_content") or result.get("content", "No content available")
                url = result.get("url", "Unknown URL")

                # Truncate very long content
                if len(content) > 5000:
                    content = content[:5000] + "... [truncated]"

                formatted_parts.append(f"\nExtracted Content {i}: {content}")
                formatted_parts.append(f"Content {i} URL: {url}")

        return "\n".join(formatted_parts)

    def _extract_drug_class_node(self, state: DrugClassState, config: RunnableConfig = None) -> dict:
        """LLM extraction node using GPT-4.1 to extract drug classes.

        Uses the DRUG_CLASS_EXTRACTION_FROM_SEARCH prompt with the search results.

        Args:
            state: Current state containing search results
            config: RunnableConfig with callbacks for Langfuse tracing

        Returns:
            dict: Updated state with drug_class_result
        """
        drug = state.get("drug", "")
        drug_class_results = state.get("drug_class_search_results", [])
        firm_results = state.get("firm_search_results", [])
        abstract_title = state.get("abstract_title", "")
        full_abstract = state.get("full_abstract", "")

        # Format search results for the prompt (including abstract info)
        formatted_input = self._format_search_results_for_prompt(
            drug, drug_class_results, firm_results, abstract_title, full_abstract
        )

        # Create messages for LLM call
        messages_for_llm = [
            SystemMessage(content=self.extraction_system_prompt),
            HumanMessage(content=formatted_input),
        ]

        try:
            print(f"🤖 Extracting drug classes using {self._extraction_model}...")
            response: AIMessage = self.extraction_llm.invoke(messages_for_llm, config=config)
            content = response.content

            # Parse JSON response
            try:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    print(f"✓ Extracted drug classes: {parsed.get('drug_classes', [])}")
                    return {
                        "drug_class_result": parsed,
                        "llm_calls": state.get("llm_calls", 0) + 1,
                    }
            except json.JSONDecodeError as e:
                print(f"⚠ JSON parsing error: {e}")

            # Default response if parsing fails
            default_result = {
                "drug_name": drug,
                "drug_classes": ["NA"],
                "content_urls": ["NA"],
                "steps_taken": [{"step": 1, "operation": "Failed to parse LLM response", "evidence": "None", "source_url": "NA"}],
                "raw_response": content,
            }
            return {
                "drug_class_result": default_result,
                "llm_calls": state.get("llm_calls", 0) + 1,
            }

        except Exception as e:
            print(f"✗ Error during drug class extraction: {e}")
            error_result = {
                "drug_name": drug,
                "drug_classes": ["NA"],
                "content_urls": ["NA"],
                "steps_taken": [{"step": 1, "operation": f"Error: {str(e)}", "evidence": "None", "source_url": "NA"}],
                "error": str(e),
            }
            return {
                "drug_class_result": error_result,
                "llm_calls": state.get("llm_calls", 0) + 1,
            }

    def invoke(
        self,
        drug: str,
        firm: list = None,
        abstract_title: str = "",
        full_abstract: str = "",
        abstract_id: str = ""
    ) -> dict:
        """Invoke the drug class extraction agent with drug and firm names.

        This method runs the search and extraction steps, returning the final response.

        Args:
            drug: The drug name to extract class for
            firm: List of pharmaceutical firm/company names (optional)
            abstract_title: The abstract title for context (optional)
            full_abstract: The full abstract text for context (optional)
            abstract_id: The abstract ID for tagging in Langfuse (optional)

        Returns:
            dict: Final state containing drug class extraction result
        """
        # Normalize firm to list
        if firm is None:
            firm = []
        elif isinstance(firm, str):
            # Handle backward compatibility if string is passed
            firm = [firm] if firm else []

        # Build tags for Langfuse tracing
        tags = [
            drug,  # Drug name as tag
            f"extraction_prompt_version:{getattr(self, 'extraction_prompt_version', 'unknown')}",
            f"extraction_model:{self.extraction_llm_config.model}",
        ]
        # Add abstract_id as tag if provided
        if abstract_id:
            tags.append(f"abstract_id:{abstract_id}")

        # Create initial state
        initial_state = {
            "drug": drug,
            "firm": firm,
            "abstract_title": abstract_title,
            "full_abstract": full_abstract,
            "drug_class_search_results": [],
            "firm_search_results": [],
            "drug_class_result": {},
            "llm_calls": 0,
        }

        # Set environment variables for Langfuse callback handler
        if self.langfuse:
            os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_config.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_config.secret_key
            os.environ["LANGFUSE_HOST"] = self.langfuse_config.host

        # Configure with Langfuse tracing and tags (single trace for all calls)
        config = RunnableConfig(
            recursion_limit=100,
            callbacks=[CallbackHandler()] if self.langfuse else [],
            metadata={"langfuse_tags": tags} if self.langfuse else {},
        )

        # Invoke the graph
        try:
            print(f"\n{'='*60}")
            print(f"🚀 Starting Drug Class Extraction for: {drug}")
            if firm:
                print(f"   Firms: {', '.join(firm)}")
            print(f"{'='*60}\n")
            
            result = self.graph.invoke(initial_state, config)
            
            print(f"\n{'='*60}")
            print(f"✅ Drug Class Extraction Complete")
            print(f"{'='*60}\n")
            
            return result
        except Exception as e:
            print(f"✗ Error during agent invocation: {e}")
            return {
                "drug": drug,
                "firm": firm,
                "abstract_title": abstract_title,
                "full_abstract": full_abstract,
                "drug_class_search_results": [],
                "firm_search_results": [],
                "drug_class_result": {
                    "drug_name": drug,
                    "drug_classes": ["NA"],
                    "content_urls": ["NA"],
                    "steps_taken": [],
                    "error": str(e),
                },
                "llm_calls": 0,
            }

    def visualize(self, output_path: str = "drug_class_agent_graph.png"):
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

