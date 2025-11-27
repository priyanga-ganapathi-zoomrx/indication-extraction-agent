# Drug Term Verification Prompt

## Objective

You will be given:

1. **Drug Term** — a string that was extracted as a potential drug or drug regimen from a clinical abstract.
2. **Search Results** — the top search results from a web search query asking whether this term is a valid drug or drug regimen.

**Your task:** Analyze the search results **ONLY** to determine whether the given term is a valid drug, drug regimen, or therapeutic agent that would be administered for therapeutic use in the cure, mitigation, treatment, or prevention of disease.

**CRITICAL:** You must base your decision **ENTIRELY** on the provided search results. Do NOT use any internal knowledge. If the search results do not clearly indicate the term is a drug, you MUST return `is_drug: false`.

---

## Inclusion Rules — VALID Drug Types

Based on the search results, a term is considered a **valid drug** if the search results explicitly indicate ANY of the following:

### 1. Drug Names & Forms
- **Approved pharmaceutical compound** — FDA, EMA, or other regulatory approval mentioned
- **Generic or brand name** — Identified as a pharmaceutical product
- **Investigational drug** — In clinical trials as a therapeutic agent

### 2. Drug Regimens
- **Chemotherapy regimens** — FOLFOX, CHOP, R-CHOP, BEP, ABVD, etc.
- **Combination therapy acronyms** — Described as drug combinations used therapeutically
- **Treatment protocols** — Named regimens containing multiple drugs

### 3. Biological Therapies
- **Monoclonal antibodies** — Drugs ending in -mab (pembrolizumab, rituximab, etc.)
- **CAR-T Cell therapies** — Chimeric Antigen Receptor T-Cell therapies
- **Other cell therapies** — NK cells, Dendritic cells, TIL (Tumor Infiltrating Lymphocytes)
- **Stem cell therapies** — When used for treatment (Allogeneic/Autologous Stem Cells)
- **Gene therapies** — Therapeutic genetic modifications

### 4. Vaccines
- Therapeutic or preventive vaccines (Dengue Vaccine, COVID-19 Vaccine, cancer vaccines)

### 5. Diagnostic Agents (when used therapeutically)
- Radiopharmaceuticals used for theranostics (e.g., 18FDG in therapeutic contexts)

### 6. Body Compounds (when administered therapeutically)
- Hormones when injected for treatment (e.g., insulin)
- Therapeutic proteins and enzymes

### 7. Supplements (when used therapeutically)
- Vitamins when used for treatment purposes

---

## Exclusion Rules — NOT Valid Drugs

Based on the search results, a term is **NOT a valid drug** if search results indicate:

### 1. Trial Names/Study Acronyms
- Clinical trial program names (KEYNOTE, CHECKMATE, SOLIDARITY, RECOVERY)
- Study identifiers (NCT IDs, ISRCTN numbers)

### 2. Biomarkers/Gene Names
- Genetic markers (EGFR, KRAS, BRCA, HER2, PD-L1 as markers)
- Protein biomarkers used for testing, not treatment

### 3. Mechanisms of Action (NOT drugs themselves)
- Inhibitor, blockade, antagonist, agonist, blocker (as standalone terms)
- Drug class descriptions without specific drug names

### 4. Broad Therapy Class Terms
- Chemotherapy (as general term)
- Radiotherapy / Radiation therapy
- Immunotherapy (as general term)
- Targeted therapy (as general term)

### 5. Endpoints/Outcome Measures
- OS (Overall Survival), PFS (Progression-Free Survival)
- ORR (Overall Response Rate), CR (Complete Response)

### 6. Routes of Administration
- Intravenous, I.V., subcutaneous, S.C., oral, topical
- **Exception:** Intravenous Immunoglobulin (IVIG) IS a valid drug

### 7. Non-Therapeutic Items
- Assays, diagnostic tests
- Contraceptive drugs
- Plant extracts (unless proven therapeutic)
- Medical devices
- Veterinary drugs

### 8. Institutions/Organizations
- Hospital names, research center names
- Pharmaceutical company names (as entities, not products)

### 9. Medical Conditions/Diseases
- Disease names, syndromes, conditions being treated

### 10. Other Exclusions
- Placebo
- Dosage information (15mg, 20mg)
- Drug-induced conditions (cisplatin-resistant, drug-related toxicity)
- Ointments (unless systemic therapeutic use confirmed)
- Fluoropyrimidine (drug class, not specific drug)

---

## Analysis Guidelines

1. **ONLY use search results** — Your decision must be based solely on what the search results say. Do NOT rely on any prior knowledge.

2. **Look for explicit drug indicators in search results:**
   - Mentions of drug class with specific drug identification
   - Therapeutic indications or approved uses
   - Regulatory approval status (FDA, EMA approved)
   - Pharmaceutical company as manufacturer (not just mentioned)
   - Dosage, administration, or prescribing information

3. **Apply exclusion rules strictly** — If search results indicate the term falls into any exclusion category, return `is_drug: false` even if it sounds drug-like.

4. **Default to false** — If the search results:
   - Do not clearly indicate the term is a drug
   - Are ambiguous or unclear
   - Are empty or irrelevant
   - Describe something other than a therapeutic agent
   
   Then you MUST return `is_drug: false`

5. **Cite the search results** — Your reason should reference what the search results said (or didn't say).

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

**Example 3: CAR-T Cell Therapy (search results confirm)**
```json
{
  "is_drug": true,
  "reason": "Search results indicate CD19-CAR-T Cell is an FDA-approved cell therapy for B-cell malignancies."
}
```

**Example 4: Not a Drug (search results indicate trial name)**
```json
{
  "is_drug": false,
  "reason": "Search results indicate KEYNOTE is a clinical trial program name by Merck, not a drug."
}
```

**Example 5: Not a Drug (search results indicate biomarker)**
```json
{
  "is_drug": false,
  "reason": "Search results describe PD-L1 as a protein biomarker used for patient selection, not a therapeutic agent."
}
```

**Example 6: Not a Drug (broad therapy term)**
```json
{
  "is_drug": false,
  "reason": "Search results indicate 'chemotherapy' is a general treatment category, not a specific drug or regimen."
}
```

**Example 7: Not a Drug (mechanism of action)**
```json
{
  "is_drug": false,
  "reason": "Search results indicate 'PD-1 inhibitor' is a drug class/mechanism, not a specific drug name."
}
```

**Example 8: Insufficient search results**
```json
{
  "is_drug": false,
  "reason": "Search results do not provide clear evidence that this term is a drug or therapeutic agent."
}
```

**Example 9: Empty or irrelevant search results**
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
- Apply both inclusion AND exclusion rules when analyzing search results
- If search results are unclear or insufficient, return `is_drug: false`
- Your reason must cite what the search results say (or don't say)
- Return **only** the JSON object, no additional text
