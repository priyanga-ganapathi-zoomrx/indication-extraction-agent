# Drug Term Verification Prompt

## Objective

You will be given:

1. **Drug Term** — a string that was extracted as a potential drug or drug regimen from a clinical abstract.
2. **Search Results** — the top search results from a web search query asking whether this term is a valid drug or drug regimen.

**Your task:** Analyze the search results **ONLY** to determine whether the given term is a valid drug, drug regimen, or therapeutic agent.

**CRITICAL:** You must base your decision **ENTIRELY** on the provided search results. Do NOT use any internal knowledge. If the search results do not clearly indicate the term is a drug, you MUST return `is_drug: false`.

---

## Verification Criteria

Based on the search results, a term is considered a **valid drug** if the search results explicitly indicate ANY of the following:

1. **Approved pharmaceutical compound** — Search results mention FDA, EMA, or other regulatory approval
2. **Drug regimen acronym** — Search results describe it as a chemotherapy or treatment regimen
3. **Investigational drug** — Search results indicate it's a drug in clinical trials
4. **Generic or brand name** — Search results identify it as a pharmaceutical product
5. **Biological therapy** — Search results describe it as a monoclonal antibody, CAR-T therapy, or gene therapy
6. **Combination therapy** — Search results describe it as a drug combination used therapeutically

Based on the search results, a term is **NOT a valid drug** if:

1. Search results indicate it's a **trial acronym/name** (study names like KEYNOTE, CHECKMATE)
2. Search results indicate it's a **biomarker/gene name** (EGFR, KRAS, BRCA)
3. Search results indicate it's an **institution/organization**
4. Search results indicate it's an **endpoint/outcome measure** (OS, PFS, ORR)
5. Search results indicate it's a **medical condition or disease**
6. Search results are **unclear, irrelevant, or insufficient** to determine drug status
7. Search results are **empty or missing**

---

## Analysis Guidelines

1. **ONLY use search results** — Your decision must be based solely on what the search results say. Do NOT rely on any prior knowledge.

2. **Look for explicit drug indicators in search results:**
   - Mentions of drug class or mechanism of action
   - Therapeutic indications or uses
   - Regulatory approval status (FDA, EMA approved)
   - Pharmaceutical company or manufacturer
   - Dosage, administration, or prescribing information

3. **Default to false** — If the search results:
   - Do not clearly indicate the term is a drug
   - Are ambiguous or unclear
   - Are empty or irrelevant
   - Describe something other than a therapeutic agent
   
   Then you MUST return `is_drug: false`

4. **Cite the search results** — Your reason should reference what the search results said (or didn't say).

---

## Output Format

Return **only** this JSON object:

```json
{
  "is_drug": true/false,
  "reason": "Brief explanation (1-2 sentences) citing what the search results indicate"
}
```

### Examples

**Example 1: Valid Drug (search results confirm)**
```json
{
  "is_drug": true,
  "reason": "Search results indicate Pembrolizumab is an FDA-approved PD-1 inhibitor used in cancer immunotherapy."
}
```

**Example 2: Drug Regimen (search results confirm)**
```json
{
  "is_drug": true,
  "reason": "Search results describe FOLFOX as a chemotherapy regimen combining folinic acid, fluorouracil, and oxaliplatin."
}
```

**Example 3: Not a Drug (search results indicate trial name)**
```json
{
  "is_drug": false,
  "reason": "Search results indicate KEYNOTE is a clinical trial program name, not a drug."
}
```

**Example 4: Not a Drug (search results indicate biomarker)**
```json
{
  "is_drug": false,
  "reason": "Search results describe PD-L1 as a protein biomarker, not a therapeutic agent."
}
```

**Example 5: Insufficient search results**
```json
{
  "is_drug": false,
  "reason": "Search results do not provide clear evidence that this term is a drug or therapeutic agent."
}
```

**Example 6: Empty or irrelevant search results**
```json
{
  "is_drug": false,
  "reason": "No relevant search results available to confirm this is a valid drug."
}
```

---

## Final Instructions

- Base your decision **ONLY** on the provided search results
- Do **NOT** use any internal or prior knowledge
- If search results are unclear or insufficient, return `is_drug: false`
- Your reason must cite what the search results say (or don't say)
- Return **only** the JSON object, no additional text
