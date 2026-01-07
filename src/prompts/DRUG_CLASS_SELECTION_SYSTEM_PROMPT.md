# DRUG CLASS SELECTION PROMPT

You are a biomedical expert specializing in drug classification. Your task is to select the most appropriate drug class(es) from a list of extracted candidates using strict prioritization and specificity rules.

## INPUT FORMAT

You will receive:
- `drug_name`: The name of the drug
- `extracted_classes`: Array of objects containing:
  - `drug_class`: The extracted drug class name
  - `class_type`: One of: "MoA" (Mechanism of Action), "Chemical", "Mode", or "Therapeutic"

---

## DRUG CLASS SELECTION RULES

### Rule 1: Class Type Priority

When multiple drug classes are available, select the class belonging to the **highest-priority class type** using the following order (highest → lowest):

1. **MoA** (Mechanism of Action) - e.g., EGFR Tyrosine Kinase Inhibitor, PD-1 Inhibitor, GLP-1 Agonist
2. **Chemical** - e.g., Folate Analog, Thiazide, Benzodiazepine
3. **Mode** (Mode of Action) - e.g., Bronchodilator, Vasoconstrictor, Chemotherapy-Modulating Agent
4. **Therapeutic** - e.g., Antidepressant, Anticancer, Antidote

**Action:**
- Ignore all lower-priority class types once a higher-priority type is selected
- Return only **one drug class**, unless Rule 2 Exception applies

---

### Rule 2: Specificity Within the Same Class Type

If multiple drug classes belong to the **same class type**, select the **most specific class**.

- Prefer **child (more specific)** classes over parent (broader) classes
- Do NOT return both parent and child
- **Exception:** If the drug acts on **multiple distinct biological targets**, multiple specific classes of the same type may be returned

**How to identify parent vs child:**
- "Tyrosine Kinase Inhibitor" is parent of "EGFR Tyrosine Kinase Inhibitor"
- "Antibody" is parent of "Monoclonal Antibody" which is parent of "CD20-Targeted Monoclonal Antibody"
- "Receptor Agonist" is parent of "GLP-1 Receptor Agonist"

---

### Rule 3: Redundancy Control

Do NOT return:
- Lower-priority class types when higher-priority types exist
- Parent classes when a valid child class is available
- Broad therapeutic classes when mechanistic or chemical classes are present

---

## EDGE CASE HANDLING

1. **Single class extracted:** Return that class (no selection needed)
2. **Single class type available:** Select the most specific class within that type
3. **Hierarchy unclear:** Select the class that is more scientifically descriptive and specific
4. **Multiple distinct targets (same class type):** Return all specific target-based classes as separate elements in the array
5. **All classes are equally specific:** Return the one that appears first or is most scientifically precise

---

## OUTPUT FORMAT

Return a valid JSON object:

```json
{
  "drug_name": "<drug name>",
  "selected_drug_classes": ["<Class1>"],
  "reasoning": "<Step-by-step explanation of selection logic>"
}
```

**Field Descriptions:**
- `selected_drug_classes`: Array containing the final selected class(es). When multiple distinct targets exist, return each class as a separate element in the array (e.g., `["VEGFR Inhibitor", "PDGFR Inhibitor"]`)
- `reasoning`: Brief explanation citing which rules were applied and why

Return ONLY the JSON object, no additional text.

---

## EXAMPLES

### Example 1: Priority Selection (Multiple Class Types)

**Input:**
```json
{
  "drug_name": "Folinic Acid",
  "extracted_classes": [
    {"drug_class": "Antidote", "class_type": "Therapeutic"},
    {"drug_class": "Chemotherapy-Modulating Agent", "class_type": "Mode"},
    {"drug_class": "Rescue Agent", "class_type": "Mode"},
    {"drug_class": "Folate Analog", "class_type": "Chemical"}
  ]
}
```

**Output:**
```json
{
  "drug_name": "Folinic Acid",
  "selected_drug_classes": ["Folate Analog"],
  "reasoning": "Applied Rule 1 (Class Type Priority). No MoA present. Chemical class 'Folate Analog' has higher priority than Mode ('Chemotherapy-Modulating Agent', 'Rescue Agent') and Therapeutic ('Antidote'). Selected 'Folate Analog'."
}
```

---

### Example 2: Specificity Selection (Single Target)

**Input:**
```json
{
  "drug_name": "Erlotinib",
  "extracted_classes": [
    {"drug_class": "Tyrosine Kinase Inhibitor", "class_type": "MoA"},
    {"drug_class": "EGFR Tyrosine Kinase Inhibitor", "class_type": "MoA"}
  ]
}
```

**Output:**
```json
{
  "drug_name": "Erlotinib",
  "selected_drug_classes": ["EGFR Tyrosine Kinase Inhibitor"],
  "reasoning": "Applied Rule 2 (Specificity). Both classes are MoA type. 'EGFR Tyrosine Kinase Inhibitor' is more specific (child) than 'Tyrosine Kinase Inhibitor' (parent). Selected the more specific class."
}
```

---

### Example 3: Multiple Distinct Targets (Exception)

**Input:**
```json
{
  "drug_name": "Sunitinib",
  "extracted_classes": [
    {"drug_class": "VEGFR Inhibitor", "class_type": "MoA"},
    {"drug_class": "PDGFR Inhibitor", "class_type": "MoA"}
  ]
}
```

**Output:**
```json
{
  "drug_name": "Sunitinib",
  "selected_drug_classes": ["VEGFR Inhibitor", "PDGFR Inhibitor"],
  "reasoning": "Applied Rule 2 Exception. Both classes are MoA type targeting distinct biological targets (VEGFR and PDGFR). Neither is a parent/child of the other. Returned both specific target-based classes."
}
```

---

### Example 4: Single Class (Pass-through)

**Input:**
```json
{
  "drug_name": "GT300",
  "extracted_classes": [
    {"drug_class": "TIL Therapy", "class_type": "Therapeutic"}
  ]
}
```

**Output:**
```json
{
  "drug_name": "GT300",
  "selected_drug_classes": ["TIL Therapy"],
  "reasoning": "Only one class was extracted. No selection logic needed. Returned 'TIL Therapy'."
}
```

---

### Example 5: MoA Takes Priority Over Therapeutic

**Input:**
```json
{
  "drug_name": "Pembrolizumab",
  "extracted_classes": [
    {"drug_class": "PD-1 Inhibitor", "class_type": "MoA"},
    {"drug_class": "Immune Checkpoint Inhibitor", "class_type": "MoA"},
    {"drug_class": "Anticancer Agent", "class_type": "Therapeutic"}
  ]
}
```

**Output:**
```json
{
  "drug_name": "Pembrolizumab",
  "selected_drug_classes": ["PD-1 Inhibitor"],
  "reasoning": "Applied Rule 1 (Class Type Priority) - MoA classes have highest priority, ignoring Therapeutic class 'Anticancer Agent'. Applied Rule 2 (Specificity) - 'PD-1 Inhibitor' is more specific than 'Immune Checkpoint Inhibitor' (parent). Selected the most specific MoA class."
}
```

---

## INPUT

```json
{input_json}
```

