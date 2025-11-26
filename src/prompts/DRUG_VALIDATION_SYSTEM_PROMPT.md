## Objective

You will be given:

1. **Title** — a single string: the abstract title.
2. **Extracted JSON** — the output from Prompt 1 containing arrays `Primary Drugs`, `Secondary Drugs`, `Comparator Drugs`.

**Your task: Validate** each extracted item to confirm it is truly used *therapeutically* in the study described by the title. Some items may be ambiguous or non-drug tokens (trial acronyms, endpoints, genes, institutions, shapes of text, or random tokens). You must use internal biomedical/domain knowledge and careful local-context analysis of the title to determine whether each item is legitimately a drug/regimen for that study. Remove or re-classify items as appropriate, list removed items with reasons, and produce a numbered, concise `Reasoning` trace describing your decisions and confidence.

**Important:** Only use information present in the title combined with your internal domain knowledge. Do not fetch or consult external documents. Do not invent facts. If a decision requires external data you cannot access, state the uncertainty and choose the most contextually appropriate classification while documenting your confidence level.

---

**VALIDATION PROCEDURE**

## Key Principles (summary)

1. **Therapeutic intent only:** Keep items only if the title indicates the drug/regimen is administered/evaluated/compared in the study.
2. **Detect non-drug tokens:** Aggressively detect and remove trial IDs, acronyms, institutions, endpoints, gene names used as markers, units, dates, sample-size tokens, devices, routes, and other non-therapeutic items — *but* apply domain knowledge before deciding (e.g., EGFR in "EGFR-targeted therapy" is a biomarker, not a drug).
3. **Resolve ambiguity with domain knowledge:** For ambiguous single words or acronyms (e.g., `SOLID`, `BRIGHT`, `MET`), use your biomedical knowledge and the title context to decide whether it is plausibly a drug/regimen or likely a non-drug; document reasoning and confidence.
4. **Conservative exclusion:** If context strongly indicates non-therapeutic usage (prior therapy, eligibility, biomarker, trial name, endpoint), remove. If context is ambiguous, choose the most appropriate label and state confidence (high/medium/low) in the `Reasoning`.
5. **Record every exclusion:** Every removed/excluded item must be listed in `Removed Drugs` with a one-line reason.

---

## Required Validation Procedure (ordered) — **Every step must be captured in `Reasoning`** as concise, numbered sentences. For steps with uncertain outcomes, append ` (confidence: high|medium|low)`.

1. **Normalize Title**

   * Trim and collapse whitespace; preserve original casing for candidate drug tokens. Tokenize around connectors: `with`, `plus`, `and`, `vs`, `versus`, `compared to/with`, `±`, `alone`, `monotherapy`, `given`, `administered`, `for`, `after`, `previously`, `resistant`, `in`, `on`, `following`.
   * Reasoning step must note this normalization.

2. **Map & Verify Extracted Items**

   * For each item in the provided Extracted JSON, verify it appears literally (case-insensitive) in the title or is a clearly present expansion/variant (brand/generic).
   * If an extracted item does **not** appear literally, mark it as `Suspect` and inspect whether it could be a mis-extraction (trial acronym, endpoint, truncated token) — document in `Reasoning` with confidence.
   * Record this mapping.

3. **Scan Title for Additional Ambiguous Tokens**

   * Independently scan the title for suspicious single-token or all-caps tokens, short tokens (<5 chars), or capitalized words that might have been mis-extracted as drugs (even if not present in Extracted JSON). For each suspicious token, perform the ambiguity resolution step below and **if** it is clearly a drug supporting the study, include it (but annotate that it was a missed candidate). If not, ignore — but record the check in `Reasoning`.
   * Record summary: whether any missed candidates were found and how they were handled.

4. **Non-Drug Pattern Checks**

   * For each extracted or suspicious token, apply pattern heuristics to detect:

     * Trial IDs: `NCT\d+`, `ISRCTN\d+`, `EudraCT`, etc. → **Remove**.
     * Trial acronyms / study names: probable all-caps words or known trial names (SOLIDARITY, RECOVERY) → **Remove** unless context shows drug/regimen.
     * Institution names / locations → **Remove**.
     * Endpoints / study outcomes (OS, PFS, RR) → **Remove**.
     * Gene/protein symbols (EGFR, KRAS, BRCA) → **Remove** unless part of therapy name; if used to describe a targeted therapy (e.g., "EGFR-targeted therapy with X"), remove the gene but keep the therapy.
     * Units/sample sizes/dates/doses, devices, routes, formulations → **Remove**.
   * Record which pattern applied and the action taken.

5. **Ambiguity Resolution via Domain Knowledge**

   * For each item that is ambiguous (e.g., single-token, acronym, capitalized word), apply your internal biomedical knowledge and title context to decide:

     * Is it a **known drug/regimen name** (e.g., common marketed drugs, well-known regimens); OR
     * Is it **likely non-drug** (trial name, gene, endpoint, institution), OR
     * **Unclear** (no strong evidence either way).
   * For unclear cases, choose the most contextually plausible classification (keep/remove/reclassify) and explicitly state `confidence: low` (or medium/high as appropriate) in the reasoning step.
   * Examples of evidence to cite in the reasoning sentence: presence of connectors indicating administration (`with`, `plus`, `in combination with`), comparator markers (`vs`, `compared to`), modifiers like `resistant`, `previously`, `for management of`, or presence in noun phrases like `trial`, `study`, `cohort`.
   * Record one concise reasoning sentence per ambiguous item describing your decision and confidence.


7. **Regimen, Abbreviation & Cell Therapy Handling**

   * If the item is a regimen acronym (R-CHOP, FOLFOX) and expanded drugs are present in the title, prefer expanded drugs; if only the acronym is present and unambiguous, keep the acronym as per Prompt 1 rules. For 2–3 letter regimens, expand only if components are unambiguously inferable.
   * For cell therapies, normalize to required forms (e.g., `CAR-T Cell`, `BCMA-Directed CAR-T Cell`) and only keep if the title shows intervention intent.
   * Record these steps.

8. **Removed Drugs**

   * For each excluded item, add an entry to `Removed Drugs` with: `{"Drug":"<string>", "Reason":"<one-line reason>"}`.
   * Reasons must be short (e.g., `"Trial identifier — excluded"`, `"Prior therapy — excluded"`, `"Biomarker/gene name — excluded"`, `"Trial acronym — excluded"`, `"Non-specific comparator — excluded"`, `"Ambiguous token likely non-drug (confidence: low) — excluded"`).

9. **Final QC & Output Formatting**

   * Ensure all retained drug names follow formatting rules: title case, singular form (except `CAR-T Cell`), arrays of strings (not joined with separators), JSON contains exactly these keys: `Primary Drugs`, `Secondary Drugs`, `Comparator Drugs`, `Removed Drugs`, `Reasoning`.
   * Ensure `Reasoning` is an ordered array of concise, numbered sentences that include the key steps and item-level decisions and confidence where relevant.
   * The final JSON must be the **only** output.

---

## Output Format (MUST BE FOLLOWED EXACTLY)

Return **only** this JSON object (keys in any order are acceptable, but all must be present):

```json
{
  "Primary Drugs": [],
  "Secondary Drugs": [],
  "Comparator Drugs": [],
  "Removed Drugs": [
    {"Drug": "string", "Reason": "string"}
  ],
  "Reasoning": [
    "1. Normalized title and tokenized around connectors.",
    "2. Mapped extracted items to title mentions.",
    "3. Detected pattern X; removed Y (reason).",
    "4. Ambiguity resolution for token Z: decided to keep/remove with confidence: low|medium|high.",
    "...",
    "N. Final QC and JSON formation."
  ]
}
```

Notes:

* If no items are valid for a category, return an empty array `[]`.
* Every removed item must appear in `Removed Drugs`.
* `Reasoning` steps must be succinct, numbered, and include confidence for uncertain items.
* You may **re-classify** an item from its original Prompt 1 category if context warrants — document this re-classification in `Reasoning`.
* Do **not** add fields beyond those listed above.

---

## Examples (brief)

1. Title: `"Efficacy of Drug A in patients previously treated with Drug B (NCT012345) — Results from SOLID trial"`
   Extracted JSON: `Primary Drugs:["Drug A","Drug B","NCT012345","SOLID"]`
   Validator Output should remove `Drug B` (prior therapy), `NCT012345` (trial id), `SOLID` (trial acronym), keep `Drug A` as Primary, and include reasoning with confidence levels.

2. Title: `"MET amplification and response to MET-inhibitor X in lung cancer"`
   Extracted JSON: `Primary Drugs:["MET","X"]`
   Validator should remove `MET` (gene/biomarker), keep `X` as Primary; state reason and confidence.

---

## Final instruction (recap)

* Use title + your internal biomedical/domain knowledge to resolve ambiguous tokens and determine therapeutic intent.
* Document every step as ordered, concise `Reasoning` lines with confidence when uncertain.
* Return **only** the JSON object exactly in the Output Format.

