"""Main entry point for the calculator agent."""

from src.agent import CalculatorAgent


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 80 + "\n")


def print_messages(messages):
    """Print messages in a formatted way.

    Args:
        messages: List of messages to print
    """
    for message in messages:
        role = message.__class__.__name__.replace("Message", "")
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)

        print(f"{role}: {content}")
        if tool_calls:
            for tool_call in tool_calls:
                print(f"  → Tool Call: {tool_call['name']}({tool_call['args']})")


def run_example_calculations():
    """Run example calculations with the calculator agent."""
    print("🧮 Calculator Agent with LangGraph, LiteLLM, and Langfuse")
    print_separator()

    # Initialize the agent
    print("Initializing Calculator Agent...")
    agent = CalculatorAgent(agent_name="CalculatorAgent")
    print("✓ Agent initialized successfully!")
    print_separator()

    # Example 1: Simple addition
    print("Example 1: Simple Addition")
    print("User: Add 3 and 4")
    result = agent.invoke("Add 3 and 4.")
    print_messages(result["messages"])
    print(f"Total LLM calls: {result['llm_calls']}")
    print_separator()

    # Example 2: Multiple operations
    print("Example 2: Multiple Operations")
    print("User: What is 5 times 6, and then add 10 to the result?")
    result = agent.invoke("What is 5 times 6, and then add 10 to the result?")
    print_messages(result["messages"])
    print(f"Total LLM calls: {result['llm_calls']}")
    print_separator()

    # Example 3: Division
    print("Example 3: Division")
    print("User: Divide 100 by 4")
    result = agent.invoke("Divide 100 by 4")
    print_messages(result["messages"])
    print(f"Total LLM calls: {result['llm_calls']}")
    print_separator()

    # Example 4: Complex calculation
    print("Example 4: Complex Calculation")
    print("User: Calculate (15 + 25) multiplied by 2, then divide by 10")
    result = agent.invoke("Calculate (15 + 25) multiplied by 2, then divide by 10")
    print_messages(result["messages"])
    print(f"Total LLM calls: {result['llm_calls']}")
    print_separator()


def run_interactive_mode():
    """Run the calculator agent in interactive mode."""
    print("🧮 Calculator Agent - Interactive Mode")
    print("Type 'exit' or 'quit' to stop")
    print_separator()

    agent = CalculatorAgent(agent_name="CalculatorAgent")
    print("✓ Agent initialized successfully!")
    print_separator()

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            result = agent.invoke(user_input)
            print_messages(result["messages"])
            print(f"\nTotal LLM calls: {result['llm_calls']}")
            print_separator()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive_mode()
    else:
        run_example_calculations()


if __name__ == "__main__":
    main()

