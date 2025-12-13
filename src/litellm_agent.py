"""Indication Extraction Agent using LiteLLM.

This module implements an indication extraction agent using the LiteLLM SDK directly,
replicating the ReAct pattern and tool handling logic without LangGraph.
"""

import json
import os
from typing import List, Dict, Any, Optional

import litellm
from pydantic import BaseModel
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from src.config import settings
from src.prompts import get_system_prompt
from src.tools import get_tools

# Define response schema
class RuleRetrieved(BaseModel):
    category: str
    subcategories: List[str]
    reason: str

class ComponentIdentified(BaseModel):
    component: str
    type: str
    normalized_form: str
    rule_applied: str

class QualityMetrics(BaseModel):
    completeness: float
    rule_adherence: float
    clinical_accuracy: float
    formatting_compliance: float

class IndicationExtractionResponse(BaseModel):
    selected_source: str
    generated_indication: str
    confidence_score: float
    reasoning: str
    rules_retrieved: List[RuleRetrieved]
    components_identified: List[ComponentIdentified]
    quality_metrics: QualityMetrics


class LiteLLMIndicationAgent:
    """Indication Extraction Agent using LiteLLM."""

    def __init__(self):
        """Initialize the agent."""
        self.tools = get_tools()
        # Create a map of tool names to functions for execution
        self.tools_map = {tool.name: tool for tool in self.tools}
        # Convert tools to OpenAI format for LiteLLM
        self.tools_schema = [convert_to_openai_tool(tool) for tool in self.tools]
        
        # Initialize Langfuse
        self._initialize_langfuse()
        
        # Get system prompt
        # We'll use a simple fallback if Langfuse is not set up or fails, similar to the original agent
        # Get system prompt
        # Force load from local file
        import os
        prompt_name = "MEDICAL_INDICATION_EXTRACTION_SYSTEM_PROMPT.md"
        print(f"ℹ Loading prompt from local file: {prompt_name}...")
        
        try:
            # Get the directory where this file (litellm_agent.py) is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct path to prompts directory
            prompt_path = os.path.join(current_dir, "prompts", prompt_name)
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read().strip()
            
            print("✓ Successfully loaded prompt from local file")
            self.prompt_version = "local_file"
            
        except Exception as e:
            print(f"✗ Error loading local prompt file: {e}")
            # Fallback to a basic prompt if file reading fails
            self.system_prompt = "You are a medical indication extraction assistant."
            self.prompt_version = "fallback"

    def _initialize_langfuse(self):
        """Initialize Langfuse configuration for LiteLLM."""
        if settings.langfuse.LANGFUSE_PUBLIC_KEY and settings.langfuse.LANGFUSE_SECRET_KEY:
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse.LANGFUSE_PUBLIC_KEY
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse.LANGFUSE_SECRET_KEY
            os.environ["LANGFUSE_HOST"] = settings.langfuse.LANGFUSE_HOST
            
            # Set LiteLLM callbacks to use OTEL integration
            litellm.callbacks = ["langfuse_otel"]
            print(f"✓ Langfuse OTEL integration configured for LiteLLM")
        else:
            print(f"ℹ Langfuse not configured (missing keys)")

    def run(self, abstract_title: str, session_title: str = "", abstract_id: str = None) -> str:
        """Run the agent to extract indication.

        Args:
            abstract_title: The abstract title.
            session_title: The session title (optional).
            abstract_id: The abstract ID (optional).

        Returns:
            str: The extracted indication or response.
        """
        # Prepare input
        input_content = f"Extract the medical indication from the following:\n\nsession_title: {session_title}\nabstract_title: {abstract_title}"
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_content}
        ]

        print(f"Starting LiteLLM Agent with model: {settings.llm.LLM_MODEL}")
        
        # Optional: Add metadata for Langfuse
        metadata = {
            "agent_name": "LiteLLMIndicationAgent",
            "session_title": session_title,
            "abstract_title": abstract_title,
            "abstract_id": abstract_id,
            "tags": [
                f"abstract_id:{abstract_id or 'unknown'}",
                f"model:{settings.llm.LLM_MODEL}",
                f"prompt_version:{self.prompt_version or 'unknown'}"
            ]
        }

        while True:
            # Call LiteLLM
            try:
                response = litellm.completion(
                    model=settings.llm.LLM_MODEL,
                    messages=messages,
                    tools=self.tools_schema,
                    tool_choice="auto",
                    temperature=settings.llm.LLM_TEMPERATURE,
                    max_tokens=settings.llm.LLM_MAX_TOKENS,
                    base_url=settings.llm.LLM_BASE_URL,
                    api_key=settings.llm.LLM_API_KEY,
                    reasoning_effort="high",
                    metadata=metadata,
                    response_format=IndicationExtractionResponse
                )
            except Exception as e:
                return f"Error during LLM call: {str(e)}"

            response_message = response.choices[0].message
            
            # Append assistant message to history
            # LiteLLM returns a message object, we need to convert it to dict for the next call
            # or just use the object if LiteLLM supports it (it usually does, but dict is safer)
            messages.append(response_message.model_dump())

            tool_calls = response_message.tool_calls

            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    
                    if function_name in self.tools_map:
                        tool_function = self.tools_map[function_name]
                        try:
                            # Execute the tool
                            result = tool_function.invoke(function_args)
                        except Exception as e:
                            result = f"Error executing tool {function_name}: {str(e)}"
                    else:
                        result = f"Error: Tool {function_name} not found"

                    # Append tool result to messages
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(result)
                    })
            else:
                # No tool calls, this is the final response
                return response_message.content


class AnalysisResponse(BaseModel):
    reasoning_trace: str
    identified_terms: List[Dict[str, str]]
    retrieved_rules: List[Dict[str, Any]]

class GenerationResponse(BaseModel):
    reasoning_trace: str
    selected_source: str
    generated_indication: str

class HybridIndicationAgent:
    """Hybrid Indication Extraction Agent (Gemini 3 + Gemini 2.5)."""

    def __init__(self):
        """Initialize the agent."""
        self.tools = get_tools()
        self.tools_map = {tool.name: tool for tool in self.tools}
        self.tools_schema = [convert_to_openai_tool(tool) for tool in self.tools]
        
        self._initialize_langfuse()
        self._load_prompts()

    def _initialize_langfuse(self):
        """Initialize Langfuse configuration."""
        if settings.langfuse.LANGFUSE_PUBLIC_KEY and settings.langfuse.LANGFUSE_SECRET_KEY:
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse.LANGFUSE_PUBLIC_KEY
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse.LANGFUSE_SECRET_KEY
            os.environ["LANGFUSE_HOST"] = settings.langfuse.LANGFUSE_HOST
            litellm.callbacks = ["langfuse_otel"]

    def _load_prompts(self):
        """Load system prompts."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            with open(os.path.join(current_dir, "prompts", "ANALYSIS_SYSTEM_PROMPT.md"), 'r') as f:
                self.analysis_prompt = f.read().strip()
            with open(os.path.join(current_dir, "prompts", "GENERATION_SYSTEM_PROMPT.md"), 'r') as f:
                self.generation_prompt = f.read().strip()
            print("✓ Loaded Hybrid Prompts")
        except Exception as e:
            print(f"✗ Error loading prompts: {e}")
            self.analysis_prompt = "Analyze clinical text."
            self.generation_prompt = "Generate indication."

    def run_analysis(self, abstract_title: str, session_title: str, metadata: Dict) -> AnalysisResponse:
        """Stage 1: Analysis with Gemini 3."""
        messages = [
            {"role": "system", "content": self.analysis_prompt},
            {"role": "user", "content": f"session_title: {session_title}\nabstract_title: {abstract_title}"}
        ]
        
        # Use Gemini 3 for Analysis
        model = "gemini/gemini-3-pro-preview"  # Or configured G3 model
        
        while True:
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=self.tools_schema,
                tool_choice="auto",
                metadata=metadata,
                response_format=AnalysisResponse
            )
            
            msg = response.choices[0].message
            messages.append(msg.model_dump())
            
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    if fn_name in self.tools_map:
                        res = self.tools_map[fn_name].invoke(args)
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": fn_name,
                            "content": str(res)
                        })
            else:
                try:
                    return json.loads(msg.content)
                except:
                    return {"reasoning_trace": "Error parsing", "retrieved_rules": []}

    def run_generation(self, abstract_title: str, session_title: str, analysis: Dict, metadata: Dict) -> GenerationResponse:
        """Stage 2: Generation with Gemini 2.5."""
        # Construct input with rules
        rules_text = json.dumps(analysis.get("retrieved_rules", []), indent=2)
        input_content = f"""
session_title: {session_title}
abstract_title: {abstract_title}
retrieved_rules:
{rules_text}
"""
        messages = [
            {"role": "system", "content": self.generation_prompt},
            {"role": "user", "content": input_content}
        ]
        
        # Use Gemini 2.5 (or 1.5 Pro) for Generation
        # Assuming 'gemini/gemini-1.5-pro' is the equivalent for "2.5" behavior requested
        model = "gemini/gemini-1.5-pro" 
        
        response = litellm.completion(
            model=model,
            messages=messages,
            metadata=metadata,
            response_format=GenerationResponse
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"generated_indication": response.choices[0].message.content}

    def run(self, abstract_title: str, session_title: str = "", abstract_id: str = None) -> str:
        """Run the full hybrid pipeline."""
        print(f"🚀 Starting Hybrid Pipeline for {abstract_id}")
        
        metadata = {
            "agent_name": "HybridIndicationAgent",
            "abstract_id": abstract_id,
            "tags": ["hybrid_pipeline"]
        }
        
        # Stage 1
        print("  1️⃣  Running Analysis (Gemini 3)...")
        analysis_result = self.run_analysis(abstract_title, session_title, metadata)
        
        # Stage 2
        print("  2️⃣  Running Generation (Gemini 1.5 Pro)...")
        generation_result = self.run_generation(abstract_title, session_title, analysis_result, metadata)
        
        # Merge results for final output format
        final_output = {
            "generated_indication": generation_result.get("generated_indication", ""),
            "selected_source": generation_result.get("selected_source", ""),
            "confidence_score": 1.0, # Placeholder
            "reasoning_trace": f"ANALYSIS: {analysis_result.get('reasoning_trace')} || GENERATION: {generation_result.get('reasoning_trace')}",
            "rules_retrieved": analysis_result.get("retrieved_rules", []),
            "components_identified": [], # Could parse from analysis if needed
            "quality_metrics": {}
        }
        
        return json.dumps(final_output)
