"""Test script to verify the calculator agent setup.

This script performs basic checks to ensure:
1. All required packages are installed
2. Configuration is loaded correctly
3. The agent can be initialized
4. Basic calculations work
"""

import sys


def check_imports():
    """Check if all required packages are installed."""
    print("Checking imports...")
    required_packages = [
        ("langchain", "LangChain"),
        ("langchain_core", "LangChain Core"),
        ("langchain_openai", "LangChain OpenAI"),
        ("langchain_anthropic", "LangChain Anthropic"),
        ("langgraph", "LangGraph"),
        ("langfuse", "Langfuse"),
        ("pydantic", "Pydantic"),
        ("pydantic_settings", "Pydantic Settings"),
        ("httpx", "HTTPX"),
    ]

    all_imported = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_imported = False

    return all_imported


def check_config():
    """Check if configuration can be loaded."""
    print("\nChecking configuration...")
    try:
        from src.config import settings

        print("  ✓ Configuration loaded")

        # Check critical settings
        checks = [
            (settings.langfuse.LANGFUSE_PUBLIC_KEY, "Langfuse Public Key"),
            (settings.langfuse.LANGFUSE_SECRET_KEY, "Langfuse Secret Key"),
            (settings.llm.LLM_API_KEY, "LLM API Key"),
            (settings.llm.LLM_MODEL, "LLM Model"),
        ]

        all_set = True
        for value, name in checks:
            if value and not value.startswith("your-") and not value.startswith("pk-lf-your"):
                print(f"  ✓ {name} is set")
            else:
                print(f"  ✗ {name} is NOT set or using default value")
                all_set = False

        return all_set
    except Exception as e:
        print(f"  ✗ Failed to load configuration: {e}")
        return False


def check_agent_initialization():
    """Check if the agent can be initialized."""
    print("\nChecking agent initialization...")
    try:
        from src.agent import CalculatorAgent

        agent = CalculatorAgent(agent_name="TestAgent")
        print("  ✓ Agent initialized successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to initialize agent: {e}")
        return False


def check_basic_calculation():
    """Check if a basic calculation works."""
    print("\nTesting basic calculation...")
    try:
        from src.agent import CalculatorAgent

        agent = CalculatorAgent(agent_name="TestAgent")
        result = agent.invoke("Add 2 and 3")

        if result and "messages" in result:
            print("  ✓ Calculation executed")
            print(f"  ✓ Total messages: {len(result['messages'])}")
            print(f"  ✓ LLM calls: {result.get('llm_calls', 0)}")

            # Check if we got a response
            if result["messages"]:
                last_message = result["messages"][-1]
                content = getattr(last_message, "content", "")
                print(f"  ✓ Response received: {content[:100]}...")
                return True
            else:
                print("  ✗ No response messages")
                return False
        else:
            print("  ✗ Invalid result format")
            return False
    except Exception as e:
        print(f"  ✗ Failed to execute calculation: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_tools():
    """Check if tools are properly defined."""
    print("\nChecking tools...")
    try:
        from src.tools import get_calculator_tools

        tools = get_calculator_tools()
        print(f"  ✓ Found {len(tools)} tools")

        for tool in tools:
            print(f"    - {tool.name}: {tool.description[:50]}...")

        return True
    except Exception as e:
        print(f"  ✗ Failed to load tools: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 80)
    print("  CALCULATOR AGENT SETUP VERIFICATION")
    print("=" * 80)
    print()

    results = {
        "Imports": check_imports(),
        "Configuration": check_config(),
        "Tools": check_tools(),
        "Agent Initialization": check_agent_initialization(),
        "Basic Calculation": check_basic_calculation(),
    }

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    all_passed = True
    for check, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{check}: {status}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("✓ All checks passed! The calculator agent is ready to use.")
        print("\nNext steps:")
        print("  1. Run examples: python -m src.main")
        print("  2. Try interactive mode: python -m src.main --interactive")
        print("  3. Check your Langfuse dashboard for traces")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        print("\nCommon issues:")
        print("  1. Missing .env file - Copy .env.template to .env")
        print("  2. Invalid API keys - Check your Langfuse and LLM keys")
        print("  3. Missing dependencies - Run: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

