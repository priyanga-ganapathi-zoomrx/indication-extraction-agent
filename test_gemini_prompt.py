import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Disable Langfuse for this test to avoid missing dependency errors
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

from src.litellm_agent import LiteLLMIndicationAgent

def test_gemini_prompt():
    print("🧪 Testing Gemini 3 Prompt Constraints")
    print("=" * 50)
    
    agent = LiteLLMIndicationAgent()
    
    test_cases = [
        {
            "name": "Single Source Violation (Session 'Metastatic' + Abstract 'TNBC')",
            "session": "Metastatic Breast Cancer",
            "abstract": "Outcomes in TNBC Patients",
            "expected_contains": "Triple-Negative Breast Cancer",
            "forbidden": "Metastatic"
        },
        {
            "name": "Casing & Noun Phrase (Lowercase Sentence)",
            "session": "",
            "abstract": "treatment of primary myelofibrosis in elderly",
            "expected_contains": "Primary Myelofibrosis",
            "forbidden": "treatment of"
        }
    ]
    
    for case in test_cases:
        print(f"\n🔹 Test Case: {case['name']}")
        print(f"   Input: Session='{case['session']}', Abstract='{case['abstract']}'")
        
        response_json = agent.run(
            abstract_title=case['abstract'],
            session_title=case['session'],
            abstract_id="test_id"
        )
        
        try:
            response = json.loads(response_json)
            indication = response.get("generated_indication", "")
            trace = response.get("reasoning_trace", "")
            
            print(f"   Reasoning Trace: {trace[:100]}...")
            print(f"   Generated Indication: '{indication}'")
            
            # Checks
            passed = True
            if case["expected_contains"].lower() not in indication.lower():
                print(f"   ❌ FAILED: Expected '{case['expected_contains']}'")
                passed = False
            
            if "forbidden" in case and case["forbidden"].lower() in indication.lower():
                print(f"   ❌ FAILED: Found forbidden term '{case['forbidden']}'")
                passed = False
                
            if passed:
                print("   ✅ PASSED")
                
        except Exception as e:
            print(f"   ❌ ERROR Parsing JSON: {e}")
            print(f"   Raw Response: {response_json}")

if __name__ == "__main__":
    test_gemini_prompt()
