# LLM Provider Comparison Experiment

Compare Gemini model performance between LangChain ChatOpenAI and LiteLLM completion APIs.

## Purpose

This experiment measures and compares:
- **Response time** - Time from request to complete response
- **Token consumption** - Prompt tokens, completion tokens, total tokens
- **Output consistency** - Same inputs should produce similar structured outputs

## Setup

Ensure you have the required environment variables set:

```bash
export GEMINI_API_KEY="your-api-key"
# or
export GOOGLE_API_KEY="your-api-key"
```

## Usage

```bash
# Run comparison with default settings
python experiments/llm_provider_comparison/compare.py

# Specify model and number of runs
python experiments/llm_provider_comparison/compare.py \
    --model "gemini/gemini-3-flash-preview" \
    --runs 3

# Test with specific drug
python experiments/llm_provider_comparison/compare.py \
    --drug "Pembrolizumab" \
    --context "Phase 3 study in advanced melanoma"
```

## Output

Results are saved to:
- Console: Formatted comparison table
- CSV: `results/comparison_<timestamp>.csv`

## Files

| File | Description |
|------|-------------|
| `config.py` | Shared configuration (model, temperature, etc.) |
| `prompts.py` | System prompt and test inputs |
| `langchain_runner.py` | LangChain ChatOpenAI implementation |
| `litellm_runner.py` | LiteLLM completion implementation |
| `compare.py` | Main comparison script |

