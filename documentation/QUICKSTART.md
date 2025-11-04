# Quick Start Guide - Calculator Agent

Get up and running with the calculator agent in 5 minutes.

## Prerequisites

- Python 3.11+
- API keys ready (see [Getting API Keys](#getting-api-keys))

## 1. Install Dependencies

### Using pip (Fastest)

```bash
cd /Users/priyangaganapathi/Documents/projects_cursor/indication-extraction-agent
pip install -r requirements.txt
```

### Using Poetry (Recommended for development)

```bash
poetry install
poetry shell
```

## 2. Configure Environment

### Copy the template

```bash
cp .env.template .env
```

### Edit .env with your keys

```env
# Langfuse - Get from https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-your-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-key-here
LANGFUSE_HOST=https://cloud.langfuse.com

# LLM Provider - Choose one:
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=anthropic/claude-sonnet-4-20250514
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4096
```

## 3. Run the Agent

### Verify Setup

```bash
python test_setup.py
```

Expected output:
```
✓ All checks passed! The calculator agent is ready to use.
```

### Run Examples

```bash
python -m src.main
```

You should see the agent performing calculations:
```
🧮 Calculator Agent with LangGraph, LiteLLM, and Langfuse

Example 1: Simple Addition
User: Add 3 and 4
AI: The sum of 3 and 4 is 7.
```

### Try Interactive Mode

```bash
python -m src.main --interactive
```

Then type your calculation requests:
```
You: What is 50 times 2?
AI: [Calculation result]

You: exit
```

## 4. Verify Langfuse Tracing

1. Go to [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Open your project
3. Click on "Traces"
4. You should see traces from your agent runs with:
   - Input/output messages
   - Token counts
   - Tool calls
   - Execution time

## Getting API Keys

### Langfuse (Required for tracing)

1. Visit [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Sign up (free tier available)
3. Create a new project
4. Go to **Settings** → **API Keys**
5. Copy:
   - **Public Key** (starts with `pk-lf-`)
   - **Secret Key** (starts with `sk-lf-`)

### OpenAI (Option 1 for LLM)

1. Visit [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a new secret key
3. Copy the key
4. In `.env`:
   ```env
   LLM_API_KEY=sk-your-openai-key
   LLM_BASE_URL=https://api.openai.com/v1
   LLM_MODEL=openai/gpt-4
   ```

### Anthropic (Option 2 for LLM)

1. Visit [https://console.anthropic.com/](https://console.anthropic.com/)
2. Go to **API Keys**
3. Create a new key
4. Copy the key
5. In `.env`:
   ```env
   LLM_API_KEY=sk-ant-your-key
   LLM_BASE_URL=https://api.anthropic.com/v1
   LLM_MODEL=anthropic/claude-sonnet-4-20250514
   ```

## Troubleshooting

### "Configuration validation error"

Make sure your `.env` file has all required variables and no placeholder values.

### "Error initializing Langfuse"

- Check your `LANGFUSE_PUBLIC_KEY` starts with `pk-lf-`
- Check your `LANGFUSE_SECRET_KEY` starts with `sk-lf-`
- Verify keys are active in Langfuse dashboard

### "Error during LLM call"

- Verify your `LLM_API_KEY` is correct
- Check `LLM_BASE_URL` matches your provider
- Ensure `LLM_MODEL` format is correct (e.g., `anthropic/model-name`)

### "ModuleNotFoundError"

Run:
```bash
pip install -r requirements.txt
```

## What's Next?

✅ **Read the Documentation**
- [README_CALCULATOR.md](README_CALCULATOR.md) - Full documentation
- [SETUP.md](SETUP.md) - Detailed setup guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details

✅ **Explore Examples**
```bash
python examples/custom_usage.py
```

✅ **Customize the Agent**
- Add new tools in `src/tools.py`
- Modify system prompt in `src/agent.py`
- Adjust LLM settings in `.env`

✅ **Monitor with Langfuse**
- View all traces in the dashboard
- Analyze token usage
- Track performance metrics

## Project Structure

```
indication-extraction-agent/
├── src/
│   ├── main.py          ← Entry point
│   ├── agent.py         ← Agent implementation
│   ├── tools.py         ← Calculator tools
│   ├── llm_handler.py   ← LLM management
│   ├── config.py        ← Configuration
│   └── langfuse_config.py
├── examples/
│   └── custom_usage.py  ← Advanced examples
├── .env                 ← Your configuration
├── test_setup.py        ← Verification script
└── requirements.txt     ← Dependencies
```

## Key Features

🚀 **LangGraph Integration**
- State-based agent workflow
- Conditional edges for tool routing
- Proper message handling

🔧 **LiteLLM Support**
- Works with OpenAI, Anthropic, and more
- Unified interface
- Easy provider switching

📊 **Langfuse Tracing**
- Automatic trace capture
- Token counting
- Performance metrics

🛡️ **Robust Error Handling**
- Graceful LLM failures
- Tool execution errors handled
- Never crashes

## Need Help?

- Check [SETUP.md](SETUP.md) for detailed instructions
- Review [README_CALCULATOR.md](README_CALCULATOR.md) for full documentation
- Run `python test_setup.py` to diagnose issues

## Success Checklist

Before moving forward, ensure:

- [ ] `python test_setup.py` passes all checks
- [ ] `python -m src.main` runs without errors
- [ ] Traces appear in Langfuse dashboard
- [ ] Token counts are visible in traces
- [ ] Interactive mode works

If all boxes are checked, you're ready to go! 🎉

