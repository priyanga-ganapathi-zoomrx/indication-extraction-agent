# Entity Mapping Workflows: High-Level Overview

---

## 1. Context and Objectives

This project addresses a core challenge in life-sciences data processing: **extracting and normalizing biomedical entities** from unstructured research abstract titles. Specifically, we are building workflows to identify and standardize:

- **Indications** (disease/condition + patient subgroup)
- **Drugs** (therapeutic agents, regimens, cell therapies)
- **Drug Classes** (mechanism of action, therapeutic class)

**The problem we are solving:**  
Conference abstracts and literature often describe studies with inconsistent terminology, varied phrasing, and implicit context. Our system transforms these into **structured, normalized entity outputs** suitable for downstream analytics, clinical pipelines, and knowledge graphs.

**Our approach:**  
We use LLM-based agents augmented with domain-specific rule retrieval and optional external verification to ensure accuracy, consistency, and clinical validity.

---

## 2. High-Level Architecture

### 2.1 System Components

| Component | Purpose |
|-----------|---------|
| **LLM Agents** | Core extraction engines using LangGraph with configurable models (Claude, GPT, Gemini) |
| **Rule Database** | CSV-based repository of domain-specific extraction rules organized by category |
| **Prompt Templates** | Detailed system prompts encoding extraction logic, formatting rules, and examples |
| **Batch Processors** | Orchestration layer for processing multiple records with parallel execution |
| **Observability** | Langfuse integration for tracing, token usage, and cost tracking |

### 2.2 Orchestration Pattern

We use **LangGraph** to orchestrate agent workflows with two primary patterns:

1. **ReAct Pattern** (Reasoning + Acting): Used for Indication and Drug Class extraction. The LLM decides when to call tools to retrieve relevant rules during extraction.

2. **Sequential Pipeline**: Used for Drug extraction. Multiple LLM calls execute in a fixed sequence (Extract → Validate → Verify).

### 2.3 External Resources

- **Extraction Rules**: Curated rule sets stored in CSV files, retrievable via tool calls. Rules are organized by category (e.g., "Gene type", "Age Group", "Occurrence") and contain action directives, examples, and validation criteria.
- **Web Search (Tavily)**: Optional verification step for drugs and required pre-fetch for drug class extraction.
- **Langfuse**: Prompt versioning and observability platform for tracking all LLM interactions.

---

## 3. Indication Entity Mapping Workflow

### 3.1 Inputs & Outputs

**Inputs:**
- Abstract title (primary source)
- Session title (fallback source if abstract title contains no disease terms)

**Outputs:**
```json
{
  "generated_indication": "KRAS G12C-Mutated Advanced Solid Tumor",
  "selected_source": "abstract_title",
  "confidence_score": 0.95,
  "reasoning": "Step-by-step extraction explanation",
  "components_identified": [...],
  "quality_metrics": { "completeness": 1.0, "clinical_accuracy": 0.95, ... }
}
```

### 3.2 Core Approach

The indication workflow follows a **ReAct (Reasoning + Acting) pattern**:

1. **Source Selection**: Applies a strict "single-source extraction principle"—if the abstract title contains any disease term, it is used exclusively. Session title is used only as a fallback.

2. **Component Identification**: The LLM scans for disease names, gene mutations, biomarkers, staging, age groups, treatment context, and other patient subgroup qualifiers.

3. **Rule Retrieval**: When the LLM identifies relevant components (e.g., a gene mutation or age qualifier), it calls the `get_indication_rules` tool to retrieve category-specific rules. Available categories include:
   - Gene Name / Gene type / Chromosome type
   - Biomarker / Stage / Grade / Risk
   - Occurrence (metastasis, recurrence, refractory status)
   - Treatment-based / Treatment Set-up
   - Onset / Age Group / Patient Sub-Group

4. **Rule Application**: The agent applies both:
   - **Generic rules** (embedded in the prompt): formatting, exclusions, disease characterization
   - **Category-specific rules** (retrieved dynamically): normalization, inclusion/exclusion logic

5. **Exclusion Logic**: Removes procedural qualifiers, sociodemographic descriptors (gender, ethnicity), drug-induced terms, and non-diagnostic items (symptoms, adverse events).

6. **Output Construction**: Combines validated components into a formatted indication string (Title Case, singular form, `;;` separator for multiple diseases).

### 3.3 Quality/Validation Logic

- **Confidence scoring**: Self-assessed by the LLM based on rule adherence and clinical accuracy
- **Quality metrics**: Completeness, rule adherence, clinical accuracy, formatting compliance
- **Strict rule compliance**: Rules must be applied exactly as retrieved; no inference or generalization

### 3.4 Current Status

**Implemented:**
- Full ReAct agent with tool-based rule retrieval
- Comprehensive prompt with generic rules and examples
- Batch processing with parallel execution (3 concurrent threads)
- Langfuse integration for observability

**Limitations:**
- Single-model extraction (no ensemble or consensus mechanism)
- Rule database requires manual curation and maintenance

---

## 4. Drug Entity Mapping Workflow

### 4.1 Inputs & Outputs

**Inputs:**
- Abstract title

**Outputs:**
```json
{
  "Primary Drugs": ["Giredestrant", "Atezolizumab", "Abemaciclib"],
  "Secondary Drugs": [],
  "Comparator Drugs": ["FOLFOX"],
  "Flagged Drugs": [{"Drug": "...", "Reason": "..."}],
  "Potential Valid Drugs": [],
  "Non-Therapeutic Drugs": [],
  "Reasoning": ["Step 1: ...", "Step 2: ...", ...]
}
```

### 4.2 Core Approach

The drug workflow uses a **sequential multi-step pipeline** (not ReAct):

**Step 1: Extraction**
- Extracts drugs from the abstract title using pattern-matching and classification rules
- Classifies each drug as **Primary** (main therapeutic agent), **Secondary** (optional combination), or **Comparator** (comparison arm)
- Applies inclusion rules for: drug names, regimens, diagnostic agents, cell therapies, vaccines
- Applies exclusion rules for: mechanisms of action, broad therapy terms, prior treatments, routes of administration, non-therapeutic items

**Step 2: Validation**
- Takes the extracted JSON and validates each drug for therapeutic intent
- Uses a separate LLM call with a validation-specific prompt
- Flags ambiguous or non-drug tokens (trial IDs, genes used as biomarkers, endpoints, institutions)
- Identifies missed therapeutic drugs from the title (added to "Potential Valid Drugs")
- Does NOT modify original classifications—only flags for manual QC

**Step 3: Verification (Optional)**
- For each extracted drug, performs a Tavily web search
- Passes search results to an LLM to determine if the term is a valid drug
- Filters out unverified drugs from final output
- Runs in parallel (configurable concurrency)

### 4.3 Quality/Validation Logic

- **Classification identifiers**: Keyword-based rules for determining Primary/Secondary/Comparator status (e.g., "plus", "with" → Primary; "with or without", "±" → Secondary; "vs", "compared with" → Comparator)
- **Multi-stage validation**: Extraction mistakes are caught by the validation step
- **Web verification**: Optional external validation via search results
- **Flagging mechanism**: Suspicious items are flagged rather than removed, preserving full audit trail

### 4.4 Current Status

**Implemented:**
- Three-step pipeline (Extraction → Validation → Verification)
- Configurable verification (can be enabled/disabled)
- Parallel processing for verification searches
- Detailed reasoning trace in output JSON

**Key differences from Indication workflow:**
- No tool calling / ReAct pattern
- Multiple sequential LLM calls instead of agentic loop
- External web search integration (Tavily)
- Classification taxonomy (Primary/Secondary/Comparator)

---

## 5. Drug Class Entity Mapping Workflow

### 5.1 Inputs & Outputs

**Inputs:**
- Drug name
- Abstract title
- Full abstract text (optional)
- Pre-fetched search results (from Tavily, cached)

**Outputs:**
```json
{
  "drug_name": "Drug A",
  "drug_classes": ["PDL1-Inhibitor"],
  "selected_sources": ["https://source.example/page"],
  "confidence_score": 0.95,
  "reasoning": "Step-by-step extraction explanation",
  "components_identified": [...],
  "quality_metrics": { "completeness": 1.0, ... }
}
```

### 5.2 Core Approach

The drug class workflow uses a **ReAct pattern** similar to indication extraction:

1. **Source Scanning**: Searches abstract title (highest priority), abstract text, and pre-fetched web content for drug class mentions.

2. **Class Type Priority**:
   - **Mechanism of Action (MoA)** is highest priority (e.g., PDL1-Inhibitor, GLP-1 Agonist)
   - If MoA is found, other class types are ignored
   - Lower-priority classes (Chemical, Mode of Action, Therapeutic) are captured only when MoA is absent

3. **Rule Retrieval**: The LLM can call `get_drug_class_rules` to retrieve category-specific rules:
   - Priority Rules (source and class type priority)
   - Class Type Rules (Inhibitors, Agonists, Antibodies, etc.)
   - Cellular Therapy Rules (cell type mappings)
   - Target Formatting Rules (hyphenation, Anti-X conversion)
   - Exclusion Rules (generic headings, context exclusions)

4. **Formatting**: Applies strict formatting rules—Title Case, hyphenation (e.g., `PDL1-Inhibitor`), singular form, target-modality combinations.

5. **Exclusion Logic**: Does not capture generic therapy terms (Chemotherapy, Immunotherapy), abbreviations alone (ADC, ICI, TKI), or drug classes mentioned in prior-treatment or adverse-event context.

### 5.3 Quality/Validation Logic

- **No hallucination rule**: If no drug class is found in the provided content, returns "NA" rather than inferring
- **Source mapping**: Each extracted class is mapped back to its source URL
- **Cellular therapy normalization**: Specific mappings (e.g., "CAR-T cell" → "CAR-T Cell Therapy")
- **Quality metrics**: Same structure as indication workflow

### 5.4 Current Status

**Implemented:**
- ReAct agent with tool-based rule retrieval
- Pre-fetched search result integration
- Batch processor with caching for search results
- Comprehensive prompt with priority rules and examples

**Relationship to other workflows:**
- Drug class extraction is a **downstream workflow** that operates on drugs identified by the drug extraction workflow
- Requires drug names as input (output from drug extraction)
- Uses web search results, which can be cached/pre-fetched for efficiency

---

## Summary Comparison

| Aspect | Indication | Drug | Drug Class |
|--------|------------|------|------------|
| **Pattern** | ReAct (agentic) | Sequential pipeline | ReAct (agentic) |
| **Tool Calling** | Yes (`get_indication_rules`) | No | Yes (`get_drug_class_rules`) |
| **LLM Calls** | Variable (depends on tool use) | 2-3 (Extract→Validate→Verify) | Variable (depends on tool use) |
| **External Search** | No | Optional (Tavily) | Yes (pre-fetched) |
| **Rule Database** | indication_extraction_rules.csv | Embedded in prompts | drug_class_extraction_rules.csv |
| **Primary Input** | Abstract title + Session title | Abstract title | Drug name + Search results |
| **Output Focus** | Disease + Patient subgroup | Drug classification | Mechanism of action / Class |

---

## Next Steps & Open Items

- Evaluation metrics and golden data comparison are in progress
- Rule database expansion based on evaluation results
- Potential ensemble approaches for improved accuracy

---

*This document reflects the current implementation as of December 2024. For technical details or code-level questions, please refer to the development team.*

