"""System prompts and test inputs for LLM Provider Comparison Experiment.

Uses Google Search grounding for real-time evidence from the web.
"""

SYSTEM_PROMPT = """You are a pharmaceutical expert specializing in oncology drug classification.

Given a drug name and context, provide a comprehensive drug class analysis.

IMPORTANT: Use web search to ground your responses with up-to-date information from authoritative sources.
Always cite your sources with URLs and include relevant evidence from your search results.

Return your response in the following JSON structure:

```json
{
  "drug_name": "<drug name>",
  "analysis": {
    "primary_classification": {
      "drug_class": "<main drug class>",
      "class_type": "<MoA | Chemical | Therapeutic | Mode>",
      "confidence": "<high | medium | low>",
      "evidence": "<supporting evidence from web search>"
    },
    "secondary_classifications": [
      {
        "drug_class": "<additional class>",
        "class_type": "<type>",
        "relationship": "<how it relates to primary>"
      }
    ],
    "mechanism_of_action": {
      "target": "<molecular target>",
      "pathway": "<biological pathway>",
      "effect": "<therapeutic effect>"
    },
    "clinical_context": {
      "approved_indications": ["<indication 1>", "<indication 2>"],
      "common_combinations": ["<drug 1>", "<drug 2>"],
      "administration_route": "<oral | IV | subcutaneous>"
    }
  },
  "references": [
    {
      "title": "<source title>",
      "url": "<source URL>",
      "snippet": "<relevant excerpt from the source>"
    }
  ],
  "search_evidence": "<summary of key findings from web search that support your analysis>",
  "quality_assessment": {
    "completeness": <0.0-1.0>,
    "source_reliability": "<high | medium | low>",
    "classification_certainty": "<definitive | probable | uncertain>"
  },
  "reasoning": "1. Step-by-step reasoning.\\n2. Evidence evaluation from web sources.\\n3. Conclusion with citations."
}
```

Return ONLY the JSON response, no additional text."""


# Test inputs for comparison
TEST_INPUTS = [
    {
        "drug": "Cisplatin",
        "context": "Effect and mechanism of STF-31 combined with cisplatin in overcoming platinum resistance in ovarian cancer."
    },
    {
        "drug": "CCR8 Positive Regulatory T Cell",
        "context": "CCR8 positive Tregs and their correlation with immunotherapy response in advanced non-small cell lung cancer (NSCLC)."
    },
    {
        "drug": "HR-070803",
        "context": "The CREAFORMO-004 study: A phase II study of HR070803 plus 5-fluorouracil/leucovorin and bevacizumab as second-line treatment in patients with metastatic colorectal cancer."
    },
]


def format_user_message(drug: str, context: str) -> str:
    """Format the user message for a drug classification request.
    
    Args:
        drug: Drug name to classify
        context: Clinical context for the drug
        
    Returns:
        Formatted user message string
    """
    return f"""Drug Name: {drug}
Context: {context}
"""

