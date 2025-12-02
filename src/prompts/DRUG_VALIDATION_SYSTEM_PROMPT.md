## Objective

You will be given:

1. **Title** — a single string: the abstract title.
2. **Extracted JSON** — the output from Prompt 1 containing arrays `Primary Drugs`, `Secondary Drugs`, `Comparator Drugs`.

**Your task:** Validate each extracted item to confirm it is truly used *therapeutically* in the study described by the title. Some items may be ambiguous or non-drug tokens (trial acronyms, endpoints, genes, institutions, shapes of text, or random tokens). Use internal biomedical/domain knowledge and careful local-context analysis of the title to determine whether each item is legitimately a drug/regimen for that study.

**Critical**
**Do not change the category (Primary, Secondary, Comparator) or the exact format/structure of any provided drug token.** The ultimate task is **verification only** — you must **not** remove or re-classify any drug string. Instead, if an item appears invalid, ambiguous, or non-therapeutic, **flag it** as a potential issue for manual QC. Flagging is the only corrective action allowed.
**Do NOT add any new drugs to the original arrays** — You must **only analyze** the extracted drugs for validation. However, you **must** also scan the title to **detect** any plausible therapeutically used drugs that were **missed** by extraction; if found, **flag** them as `Potential Valid Drugs` (see below). Do **not** insert these into `Primary/Secondary/Comparator` arrays — they go only into `Potential Valid Drugs` and `Flagged Drugs`.
**The only task is verification + missed-drug detection.**

* Output: Keep all original drugs as-is; additionally flag suspicious/invalid items and report any missed plausible therapeutic drugs separately (as potential candidates).

**Important:** Only use information present in the title combined with your internal domain knowledge. Do not fetch or consult external documents. Do not invent facts. If a decision requires external data you cannot access, state the uncertainty and flag the item (with low confidence) for manual QC.

---

## VALIDATION PROCEDURE

### Key Principles (summary)

1. **Therapeutic intent only:** Assess whether the title indicates the drug/regimen is administered/evaluated/compared in the study.
2. **Detect non-drug tokens:** Aggressively detect trial IDs, acronyms, institutions, endpoints, gene names used as markers, units, dates, sample-size tokens, devices, routes, and other non-therapeutic items — *but* apply domain knowledge before flagging (e.g., EGFR in “EGFR-targeted therapy” is a biomarker, not a drug).
3. **Resolve ambiguity with domain knowledge:** For ambiguous single words or acronyms (e.g., `SOLID`, `BRIGHT`, `MET`), use biomedical knowledge and title context to decide plausibility and document confidence.
4. **Conservative flagging:** If context strongly indicates non-therapeutic usage (prior therapy, eligibility, biomarker, trial name, endpoint), **flag**. If context is ambiguous, flag with an appropriate confidence level and concise reasoning.
5. **Record every flagged item:** Every flagged item must be listed in `Flagged Drugs` with a one-line reason and confidence.
6. **Missed therapeutic detection:** The validator **must** examine the title for any therapeutically used drugs that are **not** present in the extracted JSON. Any such plausible missed drugs must be **flagged** as `Potential Valid Drugs` (string list) and also included in `Flagged Drugs` with a one-line reason and confidence. Do **not** alter original category arrays.

---

## Required Validation Procedure (ordered) — **Every step must be captured in `Reasoning`** as concise, numbered sentences. For steps with uncertain outcomes, append ` (confidence: high|medium|low)`.

1. **Normalize Title**

   * Trim and collapse whitespace; preserve original casing for candidate drug tokens. Tokenize around connectors: `with`, `plus`, `and`, `vs`, `versus`, `compared to/with`, `±`, `alone`, `monotherapy`, `given`, `administered`, `for`, `after`, `previously`, `resistant`, `in`, `on`, `following`.
   * Reasoning step must note this normalization.

2. **Map & Verify Extracted Items**

   * For each item in the provided Extracted JSON, verify it appears literally (case-insensitive) in the title or is a clearly present expansion/variant (brand/generic).
   * If an extracted item does **not** appear literally, mark it as `Suspect` and document this in `Reasoning` with a confidence level. **Do not remove or re-classify**; instead add an entry to `Flagged Drugs` explaining why it’s suspect (one-line reason + confidence).
   * Record this mapping in `Reasoning`.

3. **Scan Title for Additional Ambiguous Tokens & Missed Therapeutic Drugs**

   * Independently scan the title for suspicious single-token or all-caps tokens, short tokens (<5 chars), or capitalized words that might have been mis-extracted as drugs (even if not present in Extracted JSON). For each suspicious token, perform ambiguity resolution below.
   * Additionally, **identify any tokens in the title that appear to be valid therapeutically used drugs/regimens but are not present in the Extracted JSON**. For each such missed therapeutic candidate:

     * **Do not** add it to the `Primary/Secondary/Comparator` arrays.
     * Add it to `Flagged Drugs` (with reason: "missed plausible therapeutic — flagged (confidence: X)") and to `Potential Valid Drugs` (strings only).
     * Document the detection and confidence in `Reasoning`.
   * If a suspicious token is not clearly therapeutic, you may flag it in `Flagged Drugs` as `ambiguous — flagged (confidence: X)` but **do not** include it in `Potential Valid Drugs`.
   * Record summary: whether any missed candidates were found and how they were handled.

4. **Non-Drug Pattern Checks**

   * For each extracted or suspicious token, apply pattern heuristics to detect:

     * Trial IDs: `NCT\d+`, `ISRCTN\d+`, `EudraCT`, etc. → **Flag** (do not remove).
     * Trial acronyms / study names: probable all-caps words or known trial names (SOLIDARITY, RECOVERY) → **Flag** unless context clearly indicates a drug/regimen; explain reasoning and confidence.
     * Institution names / locations → **Flag**.
     * Endpoints / study outcomes (OS, PFS, RR) → **Flag**.
     * Gene/protein symbols (EGFR, KRAS, BRCA) → **Flag** unless part of therapy name; if used to describe a targeted therapy (e.g., “EGFR-targeted therapy with X”), flag the gene but note the therapy if present.
     * Units/sample sizes/dates/doses, devices, routes, formulations → **Flag**.
     * **Drug classes** (e.g., “immunotherapy”, “chemotherapy”, “PD-1 inhibitor”, “monoclonal antibody”, “tyrosine kinase inhibitor”, “antibody–drug conjugate”, “checkpoint inhibitor”) → **Flag** when the token is a class rather than a specific agent. If a drug class token appears paired with a specific agent (e.g., “PD-1 inhibitor pembrolizumab”), flag the class separately but prioritize the specific agent as the therapeutic mention. For class tokens judged non-therapeutic or generic descriptors (e.g., “systemic therapy” used as background), add to `Flagged Drugs` with reason `"drug class/generic descriptor — flagged (confidence: X)"`.
   * Record which pattern applied and the action (flag) taken in `Reasoning`.

5. **Ambiguity Resolution via Domain Knowledge**

   * For each item that is ambiguous (e.g., single-token, acronym, capitalized word), apply your internal biomedical knowledge and title context to decide:

     * Is it a **known drug/regimen name**? OR
     * Is it **likely non-drug**? OR
     * **Unclear**?
   * For unclear cases, **do not remove** — **flag** as `Potentially Invalid` and state `confidence: low` (or medium/high as appropriate) in the `Flagged Drugs` reason and in the `Reasoning`.
   * Examples of evidence to cite in the reasoning sentence: presence of connectors indicating administration (`with`, `plus`, `in combination with`), comparator markers (`vs`, `compared to`), modifiers like `resistant`, `previously`, `for management of`, or presence in noun phrases like `trial`, `study`, `cohort`.
   * Record one concise reasoning sentence per ambiguous item describing your decision and confidence.

6. **Cell Therapy Handling**

   * For cell therapies, normalize how you **report** them in `Reasoning` (e.g., `CAR-T Cell`, `BCMA-Directed CAR-T Cell`) and only keep (i.e., leave the original token unchanged) if the title shows intervention intent; otherwise **flag**. Again: **do not alter** the original token strings in the provided categories.
   * Record these steps.

7. **Flagged Drugs**

   * Instead of removing items, add an entry to `Flagged Drugs` for every item that you judge ambiguous, non-therapeutic, a pattern-match (trial ID, gene, endpoint, etc.), or otherwise suspicious. Each flagged entry must be of the form: `{"Drug":"<string>", "Reason":"<one-line reason — include confidence>"}`.
   * Items judged clearly therapeutic and supported by the title should **not** be put in `Flagged Drugs`.
   * Do **not** change the original category arrays: keep `Primary Drugs`, `Secondary Drugs`, `Comparator Drugs` exactly as provided (order and strings unchanged).

8. **Potential Valid Drugs**

   * Create a new output array `Potential Valid Drugs` that contains strings only (original token text exactly as seen in the title) for any therapeutically plausible drugs/regimens **found in the title but not present in the Extracted JSON** and judged by you to be plausible interventions in the study.
   * Each entry added to `Potential Valid Drugs` must **also** appear in `Flagged Drugs` with a one-line reason: `"missed plausible therapeutic — flagged (confidence: high|medium|low)"`.
   * Do **not** add these items to the original `Primary/Secondary/Comparator` arrays. They are potential candidates for manual QC only.
   * If none are found, return an empty array `[]`.

9. **Non-Therapeutic Drugs**

   * Provide a separate array `Non-Therapeutic Drugs` that contains only original extracted drug strings (exactly as provided) that you have judged to be **non-therapeutic** based on your flagged assessments (e.g., gene, trial id, prior therapy, endpoint, drug class). Populate this array only from `Flagged Drugs` entries whose reason explicitly identifies them as non-therapeutic. Do **not** include ambiguous items unless explicitly classified as non-therapeutic in the reason.

10. **Final QC & Output Formatting**

* Ensure you **do not modify** the provided drug arrays (no deletions, no renaming, no reclassification). The final JSON must contain the original arrays unchanged.
* Provide `Flagged Drugs`, `Potential Valid Drugs`, and `Non-Therapeutic Drugs` as separate arrays (may be empty).
* Ensure `Reasoning` is an ordered array of concise, numbered sentences that include the key steps and item-level decisions and confidence where relevant. `Reasoning` must explicitly state that categories were not changed per instructions.
* The final JSON must be the **only** output.

---

## Output Format (MUST BE FOLLOWED EXACTLY)

Return **only** this JSON object (keys in any order are acceptable, but all must be present):

```json
{
  "Primary Drugs": [],
  "Secondary Drugs": [],
  "Comparator Drugs": [],
  "Flagged Drugs": [
    {"Drug": "string", "Reason": "string (include confidence: high|medium|low)"}
  ],
  "Potential Valid Drugs": [],
  "Non-Therapeutic Drugs": [],
  "Reasoning": [
    "1. Normalized title and tokenized around connectors.",
    "2. Mapped extracted items to title mentions.",
    "3. Detected missed plausible therapeutic token X in title; added to Potential Valid Drugs and flagged (confidence: level).",
    "4. Pattern X detected for token Y; flagged with reason (confidence: low|medium|high).",
    "5. Ambiguity resolution for token Z: flagged/kept with confidence: low|medium|high.",
    "...",
    "N. Final QC: verified original categories were NOT changed; Potential Valid Drugs listed separately."
  ]
}
```

Notes:

* **Do not modify** the original drug arrays — keep the tokens and categories exactly as provided in the input JSON (string casing and punctuation preserved). If you suspect a string is malformed, still keep it unchanged in its array and add it to `Flagged Drugs` with a one-line reason and confidence.
* Every flagged item must appear in `Flagged Drugs` with a concise reason and confidence.
* `Potential Valid Drugs` should contain only tokens from the title that were **not** in the extracted JSON and judged plausibly therapeutic; each such token must also be present with a flagged entry in `Flagged Drugs`.
* `Non-Therapeutic Drugs` must contain only original drug strings judged explicitly non-therapeutic from `Flagged Drugs` reasons (no reasons or confidence included). If none are identified, return an empty array `[]`.
* `Reasoning` steps must be succinct, numbered, and include confidence for uncertain items.
* You may **note missed candidates** in `Reasoning` and include them in `Potential Valid Drugs`; do **not** add them to the original drug arrays.
* Do **not** add fields beyond those listed above.

---

## Examples (brief, updated)

1. Title: `"Efficacy of Drug A in patients previously treated with Drug B (NCT012345) — Results from SOLID trial"`
   Extracted JSON: `Primary Drugs:["Drug A","Drug B","NCT012345","SOLID"]`
   Validator Output should: keep all original tokens in their categories unchanged; **flag** `Drug B` (reason: "prior therapy — flagged (confidence: high)"), flag `NCT012345` ("trial identifier — flagged (confidence: high)"), flag `SOLID` ("trial acronym — flagged (confidence: high)"); `Non-Therapeutic Drugs` should include `["Drug B","NCT012345","SOLID"]`; `Potential Valid Drugs` would be empty.

2. Title: `"MET amplification and response to MET-inhibitor X in lung cancer"`
   Extracted JSON: `Primary Drugs:["MET","X"]`
   Validator should keep both tokens in the Primary array unchanged, **flag** `MET` ("biomarker/gene name — flagged (confidence: high)"), and **keep** `X` unflagged if context supports it; `Non-Therapeutic Drugs` should include `["MET"]`; `Potential Valid Drugs` would be empty unless another therapeutic name appears in the title but was missed.

3. Title: `"Durvalumab and DISC-3405 in Phase 1 healthy volunteers"`
   Extracted JSON: `Primary Drugs:["DISC-3405"]`
   Validator should: keep `DISC-3405` unchanged, detect **Durvalumab** in the title (a therapeutically used drug) that was **missed** by extraction — add `"Durvalumab"` to `Potential Valid Drugs` and add a `Flagged Drugs` entry: `{"Drug":"Durvalumab","Reason":"missed plausible therapeutic — flagged (confidence: high)"}`; `Non-Therapeutic Drugs` empty.

---

## Final instruction (recap)

* Use title + your internal biomedical/domain knowledge to resolve ambiguous tokens and determine therapeutic intent.
* **Do not change** any provided token or its category. **Do not remove** items. **Flag** questionable items in `Flagged Drugs` with one-line reasons and confidence levels for manual QC.
* **Scan the title** to identify therapeutically plausible drugs/regimens that were **not** present in the extracted JSON; add those tokens to `Potential Valid Drugs` (strings only) and also add them to `Flagged Drugs` with the reason `"missed plausible therapeutic — flagged (confidence: X)"`. Do **not** add them to original categories.
* Populate `Non-Therapeutic Drugs` with the original extracted drug strings that you explicitly judged non-therapeutic in `Flagged Drugs`.
* Document every step as ordered, concise `Reasoning` lines with confidence when uncertain.
* Return **only** the JSON object in the Output Format.

  
