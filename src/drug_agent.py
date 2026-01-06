"""Drug Extraction Agent using LangGraph.

This module implements a drug extraction agent using the LangGraph framework.
The agent processes abstract titles through up to three steps:
1. Extraction: Extracts drugs from the abstract title
2. Validation: Validates extracted drugs for therapeutic relevance
3. Verification (optional): Verifies each drug term via Tavily search
"""

import json
import operator
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        validated_drugs_json: JSON string containing validated drugs from second LLM call
        verification_results: Dict containing verification results for each drug
        verification_removed_drugs: List of drugs removed during verification
    """

    messages: Annotated[list[BaseMessage], operator.add]
    llm_calls: int
    extracted_drugs_json: str
    abstract_title: str
    validated_drugs_json: str
    verification_results: dict
    verification_removed_drugs: list


class DrugExtractionAgent:
    """Drug Extraction Agent that extracts, validates, and optionally verifies drugs using LLM.

    This agent uses LangGraph to create a stateful conversation flow that:
    - Extracts drugs from research abstract titles (first LLM call)
    - Validates extracted drugs for therapeutic relevance (second LLM call)
    - Optionally verifies each drug term via Tavily search (third step)
    - Classifies drugs as Primary, Secondary, or Comparator
    - Returns structured JSON output with validated/verified drugs
    - Traces all operations with Langfuse (single trace for all calls)
    """

    def __init__(
        self,
        agent_name: str = "DrugExtractionAgent",
        validation_model: Optional[str] = None,
        validation_temperature: Optional[float] = None,
        validation_max_tokens: Optional[int] = None,
        enable_verification: bool = False,
        verification_model: Optional[str] = None,
        verification_temperature: Optional[float] = None,
        verification_max_tokens: Optional[int] = None,
        verification_max_parallel: int = 5,
    ):
        """Initialize the Drug Extraction Agent.

        Args:
            agent_name: Name of the agent for identification and logging
            validation_model: Optional model name for validation LLM call (uses default if not specified)
            validation_temperature: Optional temperature for validation LLM call (uses default if not specified)
            validation_max_tokens: Optional max_tokens for validation LLM call (uses default if not specified)
            enable_verification: Whether to enable Tavily-based drug verification (default: False)
            verification_model: Optional model name for verification LLM call (uses default if not specified)
            verification_temperature: Optional temperature for verification LLM call (uses default if not specified)
            verification_max_tokens: Optional max_tokens for verification LLM call (uses default if not specified)
            verification_max_parallel: Maximum parallel Tavily queries (default: 5)
        """
        self.agent_name = agent_name
        self._validation_model = validation_model
        self._validation_temperature = validation_temperature
        self._validation_max_tokens = validation_max_tokens
        self._enable_verification = enable_verification
        self._verification_model = verification_model
        self._verification_temperature = verification_temperature
        self._verification_max_tokens = verification_max_tokens
        self._verification_max_parallel = verification_max_parallel

        # Initialize Langfuse
        self.langfuse_config = get_langfuse_config()
        self.langfuse = self._initialize_langfuse() if self.langfuse_config else None

        # Initialize LLM for extraction (uses default settings)
        self.extraction_llm_config = self._get_extraction_llm_config()
        self.extraction_llm = create_llm(self.extraction_llm_config)

        # Initialize LLM for validation (can use different model)
        self.validation_llm_config = self._get_validation_llm_config()
        self.validation_llm = create_llm(self.validation_llm_config)

        # Initialize LLM for verification if enabled
        if self._enable_verification:
            self.verification_llm_config = self._get_verification_llm_config()
            self.verification_llm = create_llm(self.verification_llm_config)
            self._tavily_client = self._initialize_tavily()

        # System prompts for extraction, validation, and verification
        self.extraction_system_prompt = self._get_extraction_system_prompt()
        self.validation_system_prompt = self._get_validation_system_prompt()
        self.extraction_rules_prompt = self._get_extraction_rules_prompt()  # Reference rules for validation
        if self._enable_verification:
            self.verification_system_prompt = self._get_verification_system_prompt()

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
        """Initialize Tavily client for drug verification searches.

        Returns:
            TavilyClient instance or None if initialization fails
        """
        try:
            from tavily import TavilyClient
            
            api_key = settings.tavily.TAVILY_API_KEY
            if not api_key:
                print(f"⚠ Tavily API key not configured. Verification will be skipped.")
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

    def _get_verification_llm_config(self) -> LLMConfig:
        """Get LLM configuration for verification with optional overrides.

        Returns:
            LLMConfig: Configuration for the verification language model
        """
        return LLMConfig(
            api_key=settings.llm.LLM_API_KEY,
            model=self._verification_model or settings.llm.LLM_MODEL,
            base_url=settings.llm.LLM_BASE_URL,
            temperature=self._verification_temperature if self._verification_temperature is not None else settings.llm.LLM_TEMPERATURE,
            max_tokens=self._verification_max_tokens or settings.llm.LLM_MAX_TOKENS,
            name=f"{self.agent_name}_verification",
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

    def _get_extraction_rules_prompt(self) -> str:
        """Get the extraction rules as reference document for validation.

        Returns:
            str: Extraction rules content to be used as reference in validation
        """
        prompt_content, _ = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name="DRUG_EXTRACTION_SYSTEM_PROMPT",
            fallback_to_file=True,
        )
        return prompt_content

    def _get_verification_system_prompt(self) -> str:
        """Get the system prompt for drug verification.

        Fetches the prompt from Langfuse if configured, otherwise falls back to local file.

        Returns:
            str: System prompt content for verification
        """
        prompt_content, prompt_version = get_system_prompt(
            langfuse_client=self.langfuse,
            prompt_name="DRUG_VERIFICATION_SYSTEM_PROMPT",
            fallback_to_file=True,
        )
        # Store the prompt version for tagging
        self.verification_prompt_version = prompt_version
        return prompt_content

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the drug extraction agent.

        The graph structure depends on whether verification is enabled:
        - Without verification: START -> extraction -> validation -> END
        - With verification: START -> extraction -> validation -> verification -> END

        Returns:
            StateGraph: Compiled state graph ready for execution
        """
        # Create the state graph
        graph = StateGraph(MessagesState)

        # Add nodes for extraction and validation
        graph.add_node("extraction_llm_call", self._extraction_llm_call_node)
        graph.add_node("validation_llm_call", self._validation_llm_call_node)

        # Add verification node if enabled
        if self._enable_verification:
            graph.add_node("verification_step", self._verification_node)

        # Add edges for sequential flow
        graph.add_edge(START, "extraction_llm_call")
        graph.add_edge("extraction_llm_call", "validation_llm_call")

        if self._enable_verification:
            graph.add_edge("validation_llm_call", "verification_step")
            graph.add_edge("verification_step", END)
        else:
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
        """Validation LLM node that validates extracted drugs using 3-message pattern.

        This node handles:
        - Message 1: System Instruction (DRUG_VALIDATION_SYSTEM_PROMPT)
        - Message 2: Reference Rules (DRUG_EXTRACTION_SYSTEM_PROMPT)
        - Message 3: Extraction Result to Validate
        - Returning the validation result

        Args:
            state: Current state containing extracted_drugs_json and abstract_title

        Returns:
            dict: Updated state with validation response
        """
        # Get the abstract title and extracted JSON from state
        abstract_title = state.get("abstract_title", "")
        extracted_json = state.get("extracted_drugs_json", "{}")

        # Format the validation input (Message 3)
        validation_input = f"""Validate the extracted drugs for the following:

**Abstract Title:** {abstract_title}

**Extraction Result:**
{extracted_json}"""

        # Create messages for validation LLM call (3-message pattern)
        messages_for_llm = [
            SystemMessage(content=self.validation_system_prompt),  # Message 1: System Instruction
            HumanMessage(content=f"# REFERENCE RULES DOCUMENT\n\nThe following are the extraction rules that the extractor was instructed to follow:\n\n{self.extraction_rules_prompt}"),  # Message 2: Reference Rules
            HumanMessage(content=validation_input),  # Message 3: Extraction Result
        ]

        try:
            response: AIMessage = self.validation_llm.invoke(messages_for_llm)

            # Ensure the response has content to prevent parsing errors
            if not response.content:
                response.content = '{"validation_status": "FAIL", "validation_confidence": 0.0, "missed_drugs": [], "grounded_search_performed": false, "search_results": [], "issues_found": [], "checks_performed": {}, "validation_reasoning": "Empty response"}'

            return {
                "messages": [response],
                "llm_calls": state.get("llm_calls", 0) + 1,
                "validated_drugs_json": response.content,
            }
        except Exception as e:
            print(f"✗ Error during validation LLM call: {e}")
            # Return an error message
            error_content = '{"validation_status": "FAIL", "validation_confidence": 0.0, "missed_drugs": [], "grounded_search_performed": false, "search_results": [], "issues_found": [], "checks_performed": {}, "validation_reasoning": "Error during validation"}'
            error_message = AIMessage(content=error_content)
            return {
                "messages": [error_message],
                "llm_calls": state.get("llm_calls", 0) + 1,
                "validated_drugs_json": error_content,
            }

    def _search_drug_term(self, drug_term: str) -> dict:
        """Search for a drug term using Tavily and return search results.

        Args:
            drug_term: The drug term to search for

        Returns:
            dict: Search results with title and snippets
        """
        if not self._tavily_client:
            return {"drug_term": drug_term, "results": [], "error": "Tavily client not initialized"}

        try:
            query = f"Is {drug_term} a valid drug or drug regimen?"
            response = self._tavily_client.search(
                query=query,
                max_results=settings.tavily.TAVILY_MAX_RESULTS,
                search_depth="basic",
            )
            
            # Extract relevant information from results
            results = []
            for result in response.get("results", [])[:5]:
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("content", "")[:500],  # Limit snippet length
                    "url": result.get("url", ""),
                })
            
            return {"drug_term": drug_term, "results": results, "error": None}
        except Exception as e:
            print(f"✗ Error searching for '{drug_term}': {e}")
            return {"drug_term": drug_term, "results": [], "error": str(e)}

    def _verify_single_drug(self, drug_term: str, search_results: list, config: RunnableConfig = None) -> dict:
        """Verify a single drug term using LLM with search results.

        Args:
            drug_term: The drug term to verify
            search_results: List of search results from Tavily
            config: RunnableConfig with callbacks for Langfuse tracing

        Returns:
            dict: Verification result with is_drug and reason
        """
        # Format search results for the prompt
        formatted_results = ""
        for i, result in enumerate(search_results, 1):
            formatted_results += f"\n{i}. **{result.get('title', 'No title')}**\n"
            formatted_results += f"   {result.get('snippet', 'No content')}\n"

        if not formatted_results:
            formatted_results = "No search results available."

        verification_input = f"""**Drug Term:** {drug_term}

**Search Results:**
{formatted_results}

Based on the search results, determine if "{drug_term}" is a valid drug or drug regimen."""

        messages_for_llm = [
            SystemMessage(content=self.verification_system_prompt),
            HumanMessage(content=verification_input),
        ]

        try:
            # Use config from the graph node for Langfuse tracing (same trace)
            response: AIMessage = self.verification_llm.invoke(messages_for_llm, config=config)
            content = response.content

            # Parse JSON response
            try:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    return {
                        "drug_term": drug_term,
                        "is_drug": parsed.get("is_drug", False),
                        "reason": parsed.get("reason", "Unable to determine"),
                    }
            except json.JSONDecodeError:
                pass

            # Default response if parsing fails
            return {
                "drug_term": drug_term,
                "is_drug": False,
                "reason": "Failed to parse verification response",
            }
        except Exception as e:
            print(f"✗ Error verifying '{drug_term}': {e}")
            return {
                "drug_term": drug_term,
                "is_drug": False,
                "reason": f"Verification error: {str(e)}",
            }

    def _verification_node(self, state: MessagesState, config: RunnableConfig) -> dict:
        """Verification node that verifies each drug term via Tavily search.

        This node handles:
        - Extracting all drug terms from extraction output (uses extracted_drugs_json, not validated)
        - Querying Tavily for each drug term (in parallel with max limit)
        - Passing results to LLM for verification decision
        - Removing unverified drugs from final lists

        Note: Verification works on the original extracted drugs, not the validation output.
        The validation output now contains validation_status, issues_found, etc. instead of drug lists.

        Args:
            state: Current state containing extracted_drugs_json and validated_drugs_json
            config: RunnableConfig with callbacks for Langfuse tracing

        Returns:
            dict: Updated state with verification results and filtered drug lists
        """
        # Use extracted_drugs_json (from extraction step) for the drug lists
        extracted_json = state.get("extracted_drugs_json", "{}")
        validated_json = state.get("validated_drugs_json", "{}")
        
        # Parse extracted drugs (these contain the actual drug lists)
        try:
            import re
            json_match = re.search(r'\{.*\}', extracted_json, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
            else:
                extracted_data = json.loads(extracted_json)
        except json.JSONDecodeError:
            extracted_data = {
                "Primary Drugs": [],
                "Secondary Drugs": [],
                "Comparator Drugs": [],
            }

        # Parse validation data (contains validation status and issues)
        try:
            import re
            json_match = re.search(r'\{.*\}', validated_json, re.DOTALL)
            if json_match:
                validated_data = json.loads(json_match.group())
            else:
                validated_data = json.loads(validated_json)
        except json.JSONDecodeError:
            validated_data = {}

        # Collect all drug terms to verify from extraction
        primary_drugs = extracted_data.get("Primary Drugs", [])
        secondary_drugs = extracted_data.get("Secondary Drugs", [])
        comparator_drugs = extracted_data.get("Comparator Drugs", [])
        
        # Get validation metadata
        validation_status = validated_data.get("validation_status", "REVIEW")
        validation_issues = validated_data.get("issues_found", [])
        validation_missed_drugs = validated_data.get("missed_drugs", [])
        validation_reasoning = validated_data.get("validation_reasoning", "")

        all_drugs = list(set(primary_drugs + secondary_drugs + comparator_drugs))

        if not all_drugs or not self._tavily_client:
            # No drugs to verify or Tavily not available
            final_output = {
                "Primary Drugs": primary_drugs,
                "Secondary Drugs": secondary_drugs,
                "Comparator Drugs": comparator_drugs,
                "Validation Status": validation_status,
                "Validation Issues": validation_issues,
                "Validation Missed Drugs": validation_missed_drugs,
                "Validation Reasoning": validation_reasoning,
            }
            return {
                "messages": [AIMessage(content=json.dumps(final_output, indent=2))],
                "llm_calls": state.get("llm_calls", 0),
                "verification_results": {},
                "verification_removed_drugs": [],
            }

        # Search for all drugs in parallel with max limit
        search_results_map = {}
        with ThreadPoolExecutor(max_workers=self._verification_max_parallel) as executor:
            future_to_drug = {
                executor.submit(self._search_drug_term, drug): drug
                for drug in all_drugs
            }
            for future in as_completed(future_to_drug):
                result = future.result()
                search_results_map[result["drug_term"]] = result["results"]

        # Verify each drug with LLM (in parallel with max limit)
        print(f"🤖 Running LLM verification for {len(all_drugs)} drug(s)...")
        verification_results = {}
        llm_calls = 0
        with ThreadPoolExecutor(max_workers=self._verification_max_parallel) as executor:
            future_to_drug = {
                executor.submit(
                    self._verify_single_drug,
                    drug,
                    search_results_map.get(drug, []),
                    config  # Pass config for Langfuse tracing (same trace)
                ): drug
                for drug in all_drugs
            }
            for future in as_completed(future_to_drug):
                result = future.result()
                verification_results[result["drug_term"]] = {
                    "is_drug": result["is_drug"],
                    "reason": result["reason"],
                }
                llm_calls += 1
                print(f"  ✓ Verified '{result['drug_term']}': is_drug={result['is_drug']}")

        # Filter drugs based on verification results
        verified_primary = [d for d in primary_drugs if verification_results.get(d, {}).get("is_drug", True)]
        verified_secondary = [d for d in secondary_drugs if verification_results.get(d, {}).get("is_drug", True)]
        verified_comparator = [d for d in comparator_drugs if verification_results.get(d, {}).get("is_drug", True)]

        # Collect removed drugs from verification
        verification_removed = []
        for drug in all_drugs:
            if not verification_results.get(drug, {}).get("is_drug", True):
                verification_removed.append({
                    "Drug": drug,
                    "Reason": f"Verification failed: {verification_results[drug].get('reason', 'Unknown')}"
                })

        # Build final output
        final_output = {
            "Primary Drugs": verified_primary,
            "Secondary Drugs": verified_secondary,
            "Comparator Drugs": verified_comparator,
            "Verification Removed Drugs": verification_removed,
            "Verification Results": verification_results,
            "Validation Status": validation_status,
            "Validation Issues": validation_issues,
            "Validation Missed Drugs": validation_missed_drugs,
            "Validation Reasoning": validation_reasoning,
        }

        return {
            "messages": [AIMessage(content=json.dumps(final_output, indent=2))],
            "llm_calls": state.get("llm_calls", 0) + llm_calls,
            "verification_results": verification_results,
            "verification_removed_drugs": verification_removed,
        }

    def invoke(self, abstract_title: str, session_title: str = "", abstract_id: str = None) -> dict:
        """Invoke the drug extraction agent with abstract title.

        This method runs extraction, validation, and optionally verification steps,
        returning the final response.

        Args:
            abstract_title: The abstract title to extract drugs from
            session_title: Kept for backward compatibility but not used
            abstract_id: The abstract ID for tracking in Langfuse (optional)

        Returns:
            dict: Final state containing validated/verified drugs response
        """
        import os

        # Build tags for Langfuse tracing
        tags = [
            f"abstract_id:{abstract_id or 'unknown'}",
            f"extraction_prompt_version:{getattr(self, 'extraction_prompt_version', 'unknown')}",
            f"validation_prompt_version:{getattr(self, 'validation_prompt_version', 'unknown')}",
            f"extraction_model:{self.extraction_llm_config.model}",
            f"validation_model:{self.validation_llm_config.model}",
            f"verification_enabled:{self._enable_verification}",
        ]
        
        if self._enable_verification:
            tags.append(f"verification_prompt_version:{getattr(self, 'verification_prompt_version', 'unknown')}")
            tags.append(f"verification_model:{self.verification_llm_config.model}")

        # Format the input message with the abstract title
        input_content = f"Extract drugs from the following:\n\nabstract_title: {abstract_title}"

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=input_content)],
            "llm_calls": 0,
            "extracted_drugs_json": "",
            "abstract_title": abstract_title,
            "validated_drugs_json": "",
            "verification_results": {},
            "verification_removed_drugs": [],
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
                "validated_drugs_json": "",
                "verification_results": {},
                "verification_removed_drugs": [],
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
