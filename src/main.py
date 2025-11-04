"""Main entry point for the indication extraction agent."""

import csv
import os
import random
from src.agent import IndicationExtractionAgent


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


def load_abstracts_from_csv(csv_path="data/abstract_titles.csv", max_abstracts=None, randomize=False):
    """Load abstracts from CSV file.

    Args:
        csv_path: Path to the CSV file
        max_abstracts: Maximum number of abstracts to load (None for all)
        randomize: Whether to randomize the selection

    Returns:
        List of tuples: (abstract_id, session_title, abstract_title, ground_truth, indication)
    """
    abstracts = []

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return abstracts

    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                abstracts.append((
                    row.get('abstract_id', ''),
                    row.get('Session title', ''),
                    row.get('abstract Title', ''),
                    row.get('Ground Truth', ''),
                    row.get('indication', '')
                ))

        if randomize and max_abstracts:
            abstracts = random.sample(abstracts, min(max_abstracts, len(abstracts)))
        elif max_abstracts:
            abstracts = abstracts[:max_abstracts]

        return abstracts

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def run_example_extractions(num_abstracts=5, randomize=False):
    """Run example indication extractions with the indication extraction agent.

    Args:
        num_abstracts: Number of abstracts to process
        randomize: Whether to randomize abstract selection
    """
    print("🏥 Indication Extraction Agent with LangGraph, LiteLLM, and Langfuse")
    print_separator()

    # Load abstracts from CSV
    print("Loading abstracts from CSV...")
    abstracts = load_abstracts_from_csv(max_abstracts=num_abstracts, randomize=randomize)

    if not abstracts:
        print("No abstracts loaded. Exiting.")
        return

    print(f"Loaded {len(abstracts)} abstracts for processing")
    print_separator()

    # Initialize the agent
    print("Initializing Indication Extraction Agent...")
    agent = IndicationExtractionAgent(agent_name="IndicationExtractionAgent")
    print("✓ Agent initialized successfully!")
    print_separator()

    # Process each abstract
    total_llm_calls = 0
    for i, (abstract_id, session_title, abstract_title, ground_truth, indication) in enumerate(abstracts, 1):
        print(f"Example {i}: Abstract ID {abstract_id}")
        print(f"Abstract Title: {abstract_title}")
        print(f"Session Title: {session_title}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Expected Indication: {indication}")

        result = agent.invoke(abstract_title=abstract_title, session_title=session_title)
        print_messages(result["messages"])
        print(f"Total LLM calls for this abstract: {result['llm_calls']}")
        total_llm_calls += result['llm_calls']
        print_separator()

    print(f"Processing complete! Total abstracts processed: {len(abstracts)}")
    print(f"Total LLM calls across all abstracts: {total_llm_calls}")
    print_separator()


def run_interactive_mode():
    """Run the indication extraction agent in interactive mode."""
    print("🏥 Indication Extraction Agent - Interactive Mode")
    print("Enter abstract titles and session titles for indication extraction")
    print("Format: abstract_title|session_title")
    print("Type 'exit' or 'quit' to stop")
    print_separator()

    agent = IndicationExtractionAgent(agent_name="IndicationExtractionAgent")
    print("✓ Agent initialized successfully!")
    print_separator()

    while True:
        try:
            user_input = input("Titles (abstract|session): ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Parse input - expect format: "abstract title|session title"
            if "|" in user_input:
                abstract_title, session_title = user_input.split("|", 1)
                abstract_title = abstract_title.strip()
                session_title = session_title.strip()
            else:
                # If no separator, treat as abstract title only
                abstract_title = user_input
                session_title = ""

            print(f"Abstract Title: {abstract_title}")
            print(f"Session Title: {session_title}")

            result = agent.invoke(abstract_title=abstract_title, session_title=session_title)
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
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Indication Extraction Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                    # Run 5 abstracts (default)
  python src/main.py --num 10           # Run 10 abstracts
  python src/main.py --num 20 --random  # Run 20 random abstracts
  python src/main.py --interactive      # Interactive mode
        """
    )

    parser.add_argument(
        "--num", "--number",
        type=int,
        default=5,
        help="Number of abstracts to process (default: 5)"
    )

    parser.add_argument(
        "--random", "--randomize",
        action="store_true",
        help="Randomize abstract selection"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    # Parse known args first to handle interactive mode
    args, unknown = parser.parse_known_args()

    if args.interactive or (len(sys.argv) > 1 and sys.argv[1] == "--interactive"):
        run_interactive_mode()
    else:
        run_example_extractions(num_abstracts=args.num, randomize=args.random)


if __name__ == "__main__":
    main()

