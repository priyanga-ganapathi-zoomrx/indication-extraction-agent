"""Token usage tracking for LLM calls.

Provides a LangChain callback handler that accumulates token usage
across all LLM calls in a session. Works with any provider
(OpenAI, Anthropic, Google) via LangChain's standardized usage_metadata.

Usage:
    tracker = TokenUsageCallbackHandler()
    result = llm.invoke(messages, config={"callbacks": [tracker]})
    print(tracker.usage.total_tokens)  # accumulated across all calls
"""

import threading
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass
class TokenUsage:
    """Accumulated token usage from LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Accumulates token usage across all LLM calls in a session.

    Works with any LangChain chat model (OpenAI, Anthropic, Google)
    via the standardized usage_metadata on AIMessage. Thread-safe.

    Attributes:
        usage: Accumulated TokenUsage across all calls
        llm_calls: Number of LLM invocations tracked
    """

    def __init__(self) -> None:
        super().__init__()
        self.usage = TokenUsage()
        self.llm_calls = 0
        self._lock = threading.Lock()

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called after each LLM call completes. Accumulates token usage."""
        with self._lock:
            self.llm_calls += 1
            for generation_list in response.generations:
                for generation in generation_list:
                    usage = self._extract_usage(generation)
                    self.usage.input_tokens += usage.get("input_tokens", 0)
                    self.usage.output_tokens += usage.get("output_tokens", 0)
                    self.usage.total_tokens += usage.get("total_tokens", 0)

    @staticmethod
    def _extract_usage(generation) -> dict:
        """Extract token usage from a generation object.

        Tries standardized usage_metadata first (works for all providers),
        then falls back to generation_info for older integrations.
        """
        # Primary: LangChain's standardized usage_metadata on the message
        if hasattr(generation, "message") and hasattr(generation.message, "usage_metadata"):
            usage = generation.message.usage_metadata
            if usage:
                return usage

        # Fallback: generation_info (older OpenAI-style)
        if hasattr(generation, "generation_info") and generation.generation_info:
            token_usage = generation.generation_info.get("token_usage", {})
            if token_usage:
                return {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }

        return {}
