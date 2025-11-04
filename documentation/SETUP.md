# Setup Guide for Calculator Agent

This guide will help you set up and run the calculator agent with LangGraph, LiteLLM, and Langfuse.

## Prerequisites

- Python 3.11 or higher
- Poetry (recommended) or pip
- API keys for:
  - Langfuse (for tracing)
  - LLM provider (OpenAI, Anthropic, or LiteLLM proxy)

## Installation Steps

### Option 1: Using Poetry (Recommended)

1. **Install Poetry** (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Install dependencies**:
```bash
cd /Users/priyangaganapathi/Documents/projects_cursor/indication-extraction-agent
poetry install
```

3. **Activate the virtual environment**:
```bash
poetry shell
```

### Option 2: Using pip

1. **Create a virtual environment**:
```bash
cd /Users/priyangaganapathi/Documents/projects_cursor/indication-extraction-agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Configuration

### 1. Create Environment File

Copy the template:
```bash
cp .env.template .env
```

### 2. Get Langfuse API Keys

1. Go to [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Sign up or log in
3. Create a new project
4. Go to Settings → API Keys
5. Copy your **Public Key** (starts with `pk-lf-`)
6. Copy your **Secret Key** (starts with `sk-lf-`)

### 3. Get LLM Provider API Key

#### For OpenAI:
1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key

#### For Anthropic:
1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Navigate to API Keys
3. Create a new API key
4. Copy the key

#### For LiteLLM Proxy:
If you're using a LiteLLM proxy server, use the proxy URL and credentials provided by your setup.

### 4. Update .env File

Edit `.env` with your actual credentials:

```env
# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=pk-lf-your-actual-public-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-actual-secret-key-here
LANGFUSE_HOST=https://cloud.langfuse.com

# LLM Configuration
LLM_API_KEY=your-actual-api-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=anthropic/claude-sonnet-4-20250514
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4096
```

#### Model Configuration Examples:

**For Anthropic Claude:**
```env
LLM_API_KEY=sk-ant-your-key-here
LLM_BASE_URL=https://api.anthropic.com/v1
LLM_MODEL=anthropic/claude-sonnet-4-20250514
```

**For OpenAI GPT-4:**
```env
LLM_API_KEY=sk-your-openai-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=openai/gpt-4
```

**For LiteLLM Proxy:**
```env
LLM_API_KEY=your-proxy-key
LLM_BASE_URL=http://localhost:4000
LLM_MODEL=your-model-name
```

## Running the Agent

### Run Example Calculations

```bash
python -m src.main
```

This will execute several example calculations:
- Simple addition
- Multiple operations
- Division
- Complex calculations

### Run Interactive Mode

```bash
python -m src.main --interactive
```

In interactive mode, you can type your calculation requests:
```
You: What is 25 times 4?
[Agent responds with calculation]

You: Now add 50 to that
[Agent responds with calculation]

You: exit
```

## Verify Setup

After running the agent, verify that everything is working:

### 1. Check Console Output

You should see:
```
✓ Langfuse initialized successfully for CalculatorAgent
✓ Created and cached new LLM instance for CalculatorAgent
```

### 2. Check Langfuse Dashboard

1. Go to [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Open your project
3. Navigate to "Traces"
4. You should see traces for your agent's LLM calls

### 3. Verify Token Counting

In the Langfuse dashboard, each trace should show:
- Input tokens
- Output tokens
- Total tokens
- Cost (if configured)

## Troubleshooting

### Error: "Error initializing Langfuse"

**Solution:**
- Verify your `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`
- Ensure keys start with `pk-lf-` and `sk-lf-` respectively
- Check network connectivity to Langfuse

### Error: "Error during LLM call"

**Solution:**
- Verify your `LLM_API_KEY` is correct
- Check that `LLM_BASE_URL` matches your provider:
  - OpenAI: `https://api.openai.com/v1`
  - Anthropic: `https://api.anthropic.com/v1`
  - LiteLLM Proxy: Your proxy URL
- Ensure `LLM_MODEL` format is correct (e.g., `anthropic/claude-sonnet-4-20250514`)

### Error: "pydantic.ValidationError"

**Solution:**
- Check that all required environment variables are set in `.env`
- Ensure there are no typos in variable names
- Verify the `.env` file is in the project root directory

### Error: "ModuleNotFoundError"

**Solution:**
- Make sure you've installed all dependencies:
  ```bash
  poetry install  # or pip install -r requirements.txt
  ```
- Verify you're in the correct virtual environment

### Environment Variables Not Loading

**Solution:**
- Ensure `.env` file is in the project root (same directory as `pyproject.toml`)
- Check that `.env` file is not in `.gitignore` (wait, it should be!)
- The code uses `pydantic-settings` which automatically loads `.env`
- Try setting environment variables manually:
  ```bash
  export LANGFUSE_PUBLIC_KEY=pk-lf-your-key
  export LANGFUSE_SECRET_KEY=sk-lf-your-key
  # ... etc
  ```

## Testing the Integration

### Test 1: Simple Calculation
```bash
python -c "from src.agent import CalculatorAgent; agent = CalculatorAgent(); result = agent.invoke('Add 5 and 10'); print(result['messages'][-1].content)"
```

Expected output: A response with the calculation result (15)

### Test 2: Tool Calling
```bash
python -c "from src.agent import CalculatorAgent; agent = CalculatorAgent(); result = agent.invoke('Multiply 6 by 7'); print(f'Tool calls: {len([m for m in result[\"messages\"] if hasattr(m, \"tool_calls\") and m.tool_calls])}')"
```

Expected output: Should show at least 1 tool call

### Test 3: Langfuse Tracing
After running any agent invocation:
1. Go to your Langfuse dashboard
2. Check the "Traces" tab
3. You should see a new trace with:
   - The agent's name
   - LLM calls
   - Tool executions
   - Token counts

## Next Steps

Once setup is complete:

1. **Explore the code**:
   - `src/agent.py` - Main agent implementation
   - `src/tools.py` - Calculator tools
   - `src/llm_handler.py` - LLM management
   - `src/config.py` - Configuration

2. **Customize the agent**:
   - Add new tools
   - Modify the system prompt
   - Change LLM parameters

3. **Monitor with Langfuse**:
   - Track token usage
   - Analyze performance
   - Debug issues

4. **Integrate into your application**:
   - Use the `CalculatorAgent` class in your code
   - Build on top of the patterns established

## Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

## Support

If you encounter issues not covered in this guide:

1. Check the error message carefully
2. Verify all environment variables are set correctly
3. Ensure all dependencies are installed
4. Check the Langfuse dashboard for trace errors
5. Review the code comments for additional guidance

## Summary

You should now have:
- ✅ All dependencies installed
- ✅ Environment variables configured
- ✅ Langfuse tracing working
- ✅ LLM provider connected
- ✅ Calculator agent running successfully

Enjoy building with LangGraph, LiteLLM, and Langfuse!

