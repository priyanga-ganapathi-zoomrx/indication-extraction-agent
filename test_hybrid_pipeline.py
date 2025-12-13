import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Disable Langfuse for test
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

from src.litellm_agent import HybridIndicationAgent

def test_hybrid_pipeline():
    print("🧪 Testing Hybrid Pipeline (Gemini 3 -> Gemini 2.5)")
    print("=" * 50)
    
    agent = HybridIndicationAgent()
    
    # Test Case: Single Source Violation
    session = "Metastatic Breast Cancer"
    abstract = "Outcomes in TNBC Patients"
    
    print(f"\n🔹 Input:")
    print(f"   Session: {session}")
    print(f"   Abstract: {abstract}")
    
    metadata = {"agent_name": "TestHybrid", "abstract_id": "test_1"}
    
    # Stage 1
    print("\n1️⃣  Running Stage 1 (Analysis)...")
    try:
        analysis = agent.run_analysis(abstract, session, metadata)
        print(f"   Reasoning: {analysis.get('reasoning_trace')}")
        print(f"   Retrieved Rules: {len(analysis.get('retrieved_rules', []))} rules found")
        for rule in analysis.get('retrieved_rules', []):
            print(f"     - {rule.get('category')}/{rule.get('subcategories')}")
    except Exception as e:
        print(f"   ❌ Stage 1 Failed: {e}")
        return

    # Stage 2
    print("\n2️⃣  Running Stage 2 (Generation)...")
    try:
        generation = agent.run_generation(abstract, session, analysis, metadata)
        print(f"   Reasoning: {generation.get('reasoning_trace')}")
        print(f"   Selected Source: {generation.get('selected_source')}")
        print(f"   Generated Indication: '{generation.get('generated_indication')}'")
        
        # Validation
        if "Metastatic" in generation.get('generated_indication', '') and "TNBC" in abstract:
             print("   ❌ FAILED: Combined sources (Poison Rule Violation)")
        elif "Triple-Negative Breast Cancer" in generation.get('generated_indication', ''):
             print("   ✅ PASSED: Correctly used Abstract only")
        else:
             print("   ⚠️  UNCERTAIN: Check output manually")
             
    except Exception as e:
        print(f"   ❌ Stage 2 Failed: {e}")

if __name__ == "__main__":
    test_hybrid_pipeline()
