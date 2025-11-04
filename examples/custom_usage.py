"""Examples of custom usage patterns for the Calculator Agent."""

from src.agent import CalculatorAgent


def example_1_basic_usage():
    """Example 1: Basic usage of the calculator agent."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Initialize the agent
    agent = CalculatorAgent(agent_name="BasicCalculator")

    # Invoke with a simple calculation
    result = agent.invoke("What is 25 plus 75?")

    # Print all messages
    print("\nConversation:")
    for message in result["messages"]:
        role = message.__class__.__name__.replace("Message", "")
        print(f"\n{role}: {message.content}")

    print(f"\nTotal LLM calls: {result['llm_calls']}")


def example_2_error_handling():
    """Example 2: Error handling in the calculator agent."""
    print("\n" + "=" * 80)
    print("Example 2: Error Handling")
    print("=" * 80)

    agent = CalculatorAgent(agent_name="ErrorHandlingCalculator")

    # Try division by zero
    print("\nTrying division by zero:")
    result = agent.invoke("Divide 100 by 0")

    print("\nAgent response:")
    for message in result["messages"]:
        role = message.__class__.__name__.replace("Message", "")
        content = getattr(message, "content", "")
        if content:
            print(f"{role}: {content}")


def example_3_multiple_operations():
    """Example 3: Multiple operations in sequence."""
    print("\n" + "=" * 80)
    print("Example 3: Multiple Operations")
    print("=" * 80)

    agent = CalculatorAgent(agent_name="MultiOpCalculator")

    # Complex calculation requiring multiple tool calls
    query = "First multiply 10 by 5, then add 25 to that result, and finally divide by 5"
    print(f"\nQuery: {query}")

    result = agent.invoke(query)

    print("\nTool calls made:")
    tool_call_count = 0
    for message in result["messages"]:
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_count += 1
                print(
                    f"  {tool_call_count}. {tool_call['name']}({tool_call['args']})"
                )

    print(f"\nFinal answer: {result['messages'][-1].content}")
    print(f"Total LLM calls: {result['llm_calls']}")


def example_4_accessing_state():
    """Example 4: Accessing and analyzing agent state."""
    print("\n" + "=" * 80)
    print("Example 4: Accessing Agent State")
    print("=" * 80)

    agent = CalculatorAgent(agent_name="StateAnalyzer")

    result = agent.invoke("What is 144 divided by 12?")

    # Analyze the state
    print("\nState Analysis:")
    print(f"  Total messages: {len(result['messages'])}")
    print(f"  LLM calls: {result['llm_calls']}")

    # Count message types
    message_types = {}
    for message in result["messages"]:
        msg_type = message.__class__.__name__
        message_types[msg_type] = message_types.get(msg_type, 0) + 1

    print("\n  Message breakdown:")
    for msg_type, count in message_types.items():
        print(f"    {msg_type}: {count}")

    # Check for tool calls
    tool_calls_total = sum(
        len(getattr(m, "tool_calls", [])) for m in result["messages"]
    )
    print(f"\n  Total tool calls: {tool_calls_total}")


def example_5_custom_model_config():
    """Example 5: Using custom LLM configuration."""
    print("\n" + "=" * 80)
    print("Example 5: Custom Model Configuration")
    print("=" * 80)

    # You can customize the LLM config by modifying config.py or .env
    # Here we show how the agent uses the configuration

    agent = CalculatorAgent(agent_name="CustomConfigCalculator")

    print(f"\nAgent Configuration:")
    print(f"  Model: {agent.llm_config.model}")
    print(f"  Temperature: {agent.llm_config.temperature}")
    print(f"  Max Tokens: {agent.llm_config.max_tokens}")
    print(f"  Provider: {agent.llm_config.provider}")
    print(f"  Base URL: {agent.llm_config.base_url}")

    # Run a calculation
    result = agent.invoke("Add 100 and 200")
    print(f"\nResult: {result['messages'][-1].content}")


def example_6_reusing_agent():
    """Example 6: Reusing the same agent instance (cached LLM)."""
    print("\n" + "=" * 80)
    print("Example 6: Reusing Agent Instance")
    print("=" * 80)

    # Create one agent instance
    agent = CalculatorAgent(agent_name="ReusableCalculator")

    # Use it multiple times - the LLM instance is cached
    queries = [
        "What is 50 times 2?",
        "Now divide that by 10",
        "Add 5 to the result",
    ]

    print("\nRunning multiple calculations with the same agent:")
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        result = agent.invoke(query)
        # Note: Each invoke is independent, but LLM instance is reused
        print(f"   Answer: {result['messages'][-1].content}")


def main():
    """Run all examples."""
    print("\n")
    print("█" * 80)
    print("  CALCULATOR AGENT - CUSTOM USAGE EXAMPLES")
    print("█" * 80)

    example_1_basic_usage()
    example_2_error_handling()
    example_3_multiple_operations()
    example_4_accessing_state()
    example_5_custom_model_config()
    example_6_reusing_agent()

    print("\n" + "█" * 80)
    print("  ALL EXAMPLES COMPLETED")
    print("█" * 80)
    print()


if __name__ == "__main__":
    main()

