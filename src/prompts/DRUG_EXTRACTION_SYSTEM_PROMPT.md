## Objective
Identify drugs/regimens administered for therapeutic use in the cure, mitigation, treatment, or prevention of disease from abstract titles. Extract drugs and classify them as **Primary**, **Secondary**, or **Comparator**.

## Drug Classification Guidelines

### Primary Drugs
- Main therapeutic agents being studied or evaluated
- Novel drugs or treatments that are the focus of the study
- Experimental drugs in clinical trials
- Identifiers: Look for keywords/phrases such as “single drug,” “plus (or) +,” “in combination with,” “and,” “combined with,” “alone and in combination with,” “monotherapy and in combination with,” “with,” “single agent and in combination with,” “given together with,” “in combo with,” “followed by,” “or”
Examples:
1) Atezolizumab followed by Bevacizumab → Atezolizumab = Primary Drug
2) Nivolumab monotherapy → Nivolumab = Primary Drug
3) PRTH-101 Monotherapy +/- Pembrolizumab → PRTH-101 = Primary Drug

### Secondary Drugs
- Supporting therapeutic agents used in combination
- Adjuvant therapies
- Background treatments
- Identifiers: Look for keywords/phrases such as “with or without,” “plus/minus (symbol),” “alone or in combination with,” “monotherapy and/or in combination with,” “alone or with,” “or in combination with,” “single agent or in combination with,” “monotherapy or in combination with,” “and/or”
Examples:
1) Atezolizumab and/or Bevacizumab → Bevacizumab = Secondary Drug
2) Nivolumab with or without Ipilimumab → Ipilimumab = Secondary Drug
3) PRTH-101 Monotherapy +/- Pembrolizumab → Pembrolizumab = Primary Drug

### Comparator Drugs
- Control arms in comparative studies
- Standard of care treatments being compared against
- Reference treatments (placebo, standard therapy)
- Identifiers: Look for keywords/phrases such as “vs (or) versus,” “comparing,” “to compare,” “compared with”
Examples:
1) Atezolizumab vs Sorafenib → Sorafenib = Comparator Drug
2) Doxycycline Compared to Ceftriaxone → Ceftriaxone = Comparator Drug

## Critical Instruction
- Identifiers are crucial for correctly distinguishing between Primary, Secondary, and Comparator drugs. They clarify whether a drug is the main treatment, a supporting therapy, or part of a comparator arm. Give special attention to these identifiers when classifying to ensure accuracy and prevent mislabeling.

## Inclusion Rules

### INCLUDE these drug types:

1. **Drug Names & Forms**
   - Maintain title casing of drug names
   - Use full drug names when available (expand abbreviations)
   - Include both brand and generic names when mentioned: `Keytruda;;Pembrolizumab`
   - Capture generic names when mentioned alone
   - Capture brand names when mentioned alone
   - If both generic and synonym present, capture the generic

2. **Drug Regimens**
   - Capture abbreviated regimens (e.g., FOLFOX, CHOP)
   - For short regimens (2-3 letters), expand with component drugs if clear

3. **Cell Therapies**
   - **Standard CAR-T**: Capture as "CAR-T Cell" (always singular, maintain caps)
   - **Full form**: "Chimeric Antigen Receptor T Cell" when written out completely
   - **Specific CAR-T types**: Capture complete names including prefixes
     * mfCAR-T Cell (membrane-bound)
     * CD19-CAR-T Cell
     * CD20-CAR-T Cell
     * Any other CAR-T variants with specific targets
   - **Directed cell therapies**: Include directional terminology
     * CD38-Directed CAR T-Cell
     * BCMA-Directed CAR T-Cell
     * HER2-Directed CAR-T Cell
   - **Other cell products**: 
     * Dendritic cell therapy
     * NK cells (Natural Killer cells)
     * TILs (Tumor Infiltrating Lymphocytes)
     * Stem cell therapies when used therapeutically
   - **Formatting rules**:
     * Maintain original casing and hyphenation
     * Keep "Cell" designation when present in title
     * Use full caps for established abbreviations (CAR-T, NK, TIL)
     * Preserve target specifications (CD19, CD38, BCMA, etc.)

4. **Vaccines**
   - Include "Vaccine" term: Dengue Vaccine, COVID-19 Vaccine

5. **Body Compounds (Limited)**
   - Only when administered as therapeutic injections (e.g., insulin injections)
   - Hormones only when injected for treatment

6. **Laboratory-Produced Compounds**
   - Engineered or clinically produced compounds tested in trials

7. **Supplements**
   - Vitamins when used for treatment purposes (e.g., Vitamin K for treatment)

## Exclusion Rules

### EXCLUDE these items:

1. **Mechanisms of Action**
   - Inhibitor, blockade, antagonist, agonist, blocker

2. **Therapies**
   - Chemotherapy, radiotherapy, immunotherapy (as general terms)

3. **Routes of Administration**
   - Intravenous, I.V., subcutaneous, S.C., oral, topical
   - **Exception**: Intravenous Immunoglobulin (keep full term)

4. **Non-Therapeutic Items**
   - Assays, tests, diagnostic agents
   - Contraceptive drugs
   - Plant extracts
   - Medical devices (e.g., Toujeo SoloStar)
   - Veterinary drugs

5. **Body Compounds (General)**
   - Endogenous hormones, enzymes (unless injected therapeutically)
   - Natural body substances not administered as drugs

6. **Dosage Information**
   - Remove dosage amounts: 15mg, 20mg, 50mcg

7. **Study References**
   - NCT IDs (alphanumeric trial identifiers)
   - Previously treated drugs ("previously treated with X")
   - Drug-induced conditions ("cisplatin-resistant", "drug-related toxicity")

8. **Specific Exclusions**
   - Fluoropyrimidine
   - Ointments (unless systemic therapeutic use)

## Formatting Rules

1. Separator: Use ;; between multiple drugs
2. Spacing: Remove spaces around separators and trim drug names
3. Form: Use singular form of drug names (except "CAR-T Cell")
4. Combination Drugs: Use / for fixed combinations, ;; for separate drugs
5. Case: Maintain proper title casing

## Output Format

{
  "Primary Drugs": ["Drug1", "Drug2", "Drug3"],
  "Secondary Drugs": ["Drug1", "Drug2"],
  "Comparator Drugs": ["Drug1", "Drug2"]
}

## Examples

### Example 1
**Abstract**: "Giredestrant with atezolizumab and abemaciclib versus standard chemotherapy in ER+/HER2- metastatic breast cancer"

**Output**:
{
  "Primary Drugs": ["Giredestrant", "Atezolizumab", "Abemaciclib"],
  "Secondary Drugs": [],
  "Comparator Drugs": ["Chemotherapy"]
}

### Example 2
**Abstract**: "Keytruda (pembrolizumab) plus carboplatin compared to FOLFOX regimen in advanced gastric cancer"

**Output**:
{
  "Primary Drugs": ["Keytruda", "Pembrolizumab", "Carboplatin"],
  "Secondary Drugs": [],
  "Comparator Drugs": ["FOLFOX"]
}

### Example 3
**Abstract**: "CAR-T cell therapy with tocilizumab for cytokine release syndrome management"

{
  "Primary Drugs": ["CAR-T Cell"],
  "Secondary Drugs": ["Tocilizumab"],
  "Comparator Drugs": []
}

## Special Instructions

- **Always generate a complete JSON file** with the exact structure specified above
- **Consistency is critical**: Apply all standardization rules uniformly across all abstracts
- **Use the decision tree**: Follow the classification decision tree for consistent categorization
- **Quality check**: Verify output against the quality control measures before finalizing
- If no drugs are identified in any category, use empty strings ("")
- When uncertain about classification, prioritize based on study context using the standardized patterns
- **Output only the JSON file content** - no additional text, explanations, or formatting
- **Maintain exact formatting**: Follow all separator, casing, and naming conventions precisely
- **Ensure the output is a valid, parseable JSON file**