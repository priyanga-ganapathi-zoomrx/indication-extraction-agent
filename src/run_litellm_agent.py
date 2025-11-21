"""Example script to run the LiteLLM Indication Extraction Agent."""

import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.litellm_agent import LiteLLMIndicationAgent

def main():
    print("Initializing LiteLLM Agent...")
    agent = LiteLLMIndicationAgent()
    
    abstract_title = "Efficacy of Pembrolizumab in Non-Small Cell Lung Cancer"
    session_title = "Immunotherapy in NSCLC"
    
    print(f"\nProcessing:\nAbstract: {abstract_title}\nSession: {session_title}\n")
    
    result = agent.run(
        abstract_title=abstract_title, 
        session_title=session_title,
        abstract_id="12345"
    )
    
    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    print(result)

if __name__ == "__main__":
    main()
