"""Examples of custom usage patterns for the Indication Extraction Agent."""

from src.agent import IndicationExtractionAgent


def example_1_basic_usage():
    """Example 1: Basic usage of the indication extraction agent."""
    print("=" * 80)
    print("Example 1: Basic Indication Extraction")
    print("=" * 80)

    # Initialize the agent
    agent = IndicationExtractionAgent(agent_name="BasicExtractor")

    # Example abstract title
    abstract_title = "Brentuximab Vedotin-Based Regimens for Elderly Patients with Newly Diagnosed Classical Hodgkin Lymphoma"
    session_title = ""

    print(f"Abstract Title: {abstract_title}")
    print(f"Session Title: {session_title}")

    # Invoke the agent
    result = agent.invoke(abstract_title=abstract_title, session_title=session_title)

    # Print all messages
    print("\nConversation:")
    for message in result["messages"]:
        role = message.__class__.__name__.replace("Message", "")
        print(f"\n{role}: {message.content}")

    print(f"\nTotal LLM calls: {result['llm_calls']}")


def example_2_multiple_diseases():
    """Example 2: Multiple diseases in one title."""
    print("\n" + "=" * 80)
    print("Example 2: Multiple Diseases")
    print("=" * 80)

    agent = IndicationExtractionAgent(agent_name="MultiDiseaseExtractor")

    # Title with multiple diseases
    abstract_title = "Clinical Outcomes in Severe Refractory Asthma with Cardiovascular Risk"
    session_title = ""

    print(f"Abstract Title: {abstract_title}")

    result = agent.invoke(abstract_title=abstract_title, session_title=session_title)

    print("\nAgent response:")
    for message in result["messages"]:
        role = message.__class__.__name__.replace("Message", "")
        content = getattr(message, "content", "")
        if content:
            print(f"{role}: {content}")


def example_3_rule_retrieval():
    """Example 3: Demonstrating rule retrieval during extraction."""
    print("\n" + "=" * 80)
    print("Example 3: Rule Retrieval in Action")
    print("=" * 80)

    agent = IndicationExtractionAgent(agent_name="RuleBasedExtractor")

    # Complex title requiring multiple rules
    abstract_title = "Therapies for PD-L1 ≥ 50% Non-Small Cell Lung Cancer in Previously Treated Smokers"
    session_title = ""

    print(f"Abstract Title: {abstract_title}")

    result = agent.invoke(abstract_title=abstract_title, session_title=session_title)

    print("\nTool calls made:")
    tool_call_count = 0
    for message in result["messages"]:
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_count += 1
                print(
                    f"  {tool_call_count}. {tool_call['name']}({tool_call['args']})"
                )

    print(f"\nFinal result: {result['messages'][-1].content}")
    print(f"Total LLM calls: {result['llm_calls']}")


def example_4_accessing_state():
    """Example 4: Accessing and analyzing agent state."""
    print("\n" + "=" * 80)
    print("Example 4: Accessing Agent State")
    print("=" * 80)

    agent = IndicationExtractionAgent(agent_name="StateAnalyzer")

    abstract_title = "Overall and Progression-Free Survival in Patients Treated with Fedratinib as First-Line Myelofibrosis"
    result = agent.invoke(abstract_title=abstract_title, session_title="")

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

    agent = IndicationExtractionAgent(agent_name="CustomConfigExtractor")

    print(f"\nAgent Configuration:")
    print(f"  Model: {agent.llm_config.model}")
    print(f"  Temperature: {agent.llm_config.temperature}")
    print(f"  Max Tokens: {agent.llm_config.max_tokens}")
    print(f"  Provider: {agent.llm_config.provider}")
    print(f"  Base URL: {agent.llm_config.base_url}")

    # Run an extraction
    result = agent.invoke(abstract_title="Chronic Lymphocytic Leukemia After Prior Ibrutinib", session_title="")
    print(f"\nResult: {result['messages'][-1].content}")


def example_6_reusing_agent():
    """Example 6: Reusing the same agent instance (cached LLM)."""
    print("\n" + "=" * 80)
    print("Example 6: Reusing Agent Instance")
    print("=" * 80)

    # Create one agent instance
    agent = IndicationExtractionAgent(agent_name="ReusableExtractor")

    # Use it multiple times - the LLM instance is cached
    titles = [
        "Acute Myeloid Leukemia in Pediatric Patients",
        "Metastatic Breast Cancer Treatment",
        "Early Stage Non-Small Cell Lung Cancer",
    ]

    print("\nRunning multiple extractions with the same agent:")
    for i, title in enumerate(titles, 1):
        print(f"\n{i}. Title: {title}")
        result = agent.invoke(abstract_title=title, session_title="")
        # Note: Each invoke is independent, but LLM instance is reused
        print(f"   Result: {result['messages'][-1].content}")


def main():
    """Run all examples."""
    print("\n")
    print("█" * 80)
    print("  INDICATION EXTRACTION AGENT - CUSTOM USAGE EXAMPLES")
    print("█" * 80)

    example_1_basic_usage()
    example_2_multiple_diseases()
    example_3_rule_retrieval()
    example_4_accessing_state()
    example_5_custom_model_config()
    example_6_reusing_agent()

    print("\n" + "█" * 80)
    print("  ALL EXAMPLES COMPLETED")
    print("█" * 80)
    print()


if __name__ == "__main__":
    main()

