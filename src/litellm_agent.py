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
        self.system_prompt, self.prompt_version = get_system_prompt(
            langfuse_client=None,  # We can add Langfuse support later if needed
            prompt_name="MEDICAL_INDICATION_EXTRACTION_SYSTEM_PROMPT",
            fallback_to_file=True,
        )

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


