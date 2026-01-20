# Indication Extraction Agent

Batch processing pipelines for extracting indications, drugs, and drug classes from medical abstracts using LLMs.

## Features

- **Three Extraction Pipelines**: Indication, Drug, and Drug Class extraction
- **Dual Storage Support**: Local filesystem or Google Cloud Storage (GCS)
- **Batch Processing**: Parallel execution with progress tracking
- **Retry Logic**: Automatic retry for failed extractions
- **Status Tracking**: Per-abstract and batch-level status monitoring
- **Validation**: Separate validation step for each pipeline

## Setup

1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
poetry shell
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

### Input Format

All pipelines expect a CSV with columns: `abstract_id`, `abstract_title`, `firm`, `full_abstract`

### Data Preprocessing

#### Clean Firms Column

If your input CSV has malformed `firms` data (e.g., prefixed with `<0>` or other artifacts), use the cleaning script:

```bash
python -m src.scripts.drug_class.clean_firms [input_path] [output_path]
```

**Default paths (if no arguments provided):**
- Input: `data/drug/input/abstract_titles(in).csv`
- Output: `data/drug/input/abstract_titles_cleaned.csv`

**What it cleans:**
- Removes `<0>` (or any `<digit>`) prefix from firms values
- Preserves `;;` as the separator for multiple firms

**Example:**
```
Before: <0>Astellas Pharma;;Pfizer
After:  Astellas Pharma;;Pfizer
```

**Output statistics:**
```
Results:
  Total rows:        1006
  Rows with firms:   277
  Rows cleaned:      277
  Multi-firm rows:   73
```

**Note:** The `firms` column uses `;;` (double semicolon) as the separator for multiple firms.

### Indication Pipeline

**Extraction:**
```bash
python -m src.scripts.indication.extraction_processor \
  --input gs://bucket/Conference/abstract_titles.csv \
  --output_dir gs://bucket/Conference/indication
```

**Validation:**
```bash
python -m src.scripts.indication.validation_processor \
  --input gs://bucket/Conference/abstract_titles.csv \
  --output_dir gs://bucket/Conference/indication
```

### Drug Pipeline

**Extraction:**
```bash
python -m src.scripts.drug.extraction_processor \
  --input gs://bucket/Conference/abstract_titles.csv \
  --output_dir gs://bucket/Conference/drug
```

**Validation:**
```bash
python -m src.scripts.drug.validation_processor \
  --input gs://bucket/Conference/abstract_titles.csv \
  --output_dir gs://bucket/Conference/drug
```

### Drug Class Pipeline

**Extraction:**
```bash
python -m src.scripts.drug_class.extraction_processor \
  --input gs://bucket/Conference/abstract_titles.csv \
  --drug_output_dir gs://bucket/Conference/drug \
  --output_dir gs://bucket/Conference/drug_class
```

**Validation:**
```bash
python -m src.scripts.drug_class.validation_processor \
  --input gs://bucket/Conference/abstract_titles.csv \
  --output_dir gs://bucket/Conference/drug_class
```

### Combined QA Export

Export drug extraction/validation and drug class pipeline results to a single CSV for QA review:

```bash
python -m src.scripts.drug_drug_class_exporter \
  --input gs://bucket/Conference/abstract_titles.csv \
  --drug_output_dir gs://bucket/Conference/drug \
  --drug_class_output_dir gs://bucket/Conference/drug_class \
  --output qa_export.csv
```

**Output columns include:**
- All input CSV columns
- Drug extraction: `drug_extraction_primary_drugs`, `drug_extraction_secondary_drugs`, `drug_extraction_comparator_drugs`, `drug_extraction_reasoning`
- Drug validation: `drug_validation_status`, `drug_validation_grounded_search_performed`, `drug_validation_search_results`, `drug_validation_missed_drugs`, `drug_validation_issues_found`, `drug_validation_reasoning`
- Drug class Step 1-5: `drug_class_step1_drug_to_components`, `drug_class_step2_drug_classes`, `drug_class_step2_extraction_details`, `drug_class_step3_selected_drug_classes`, `drug_class_step3_reasoning`, `drug_class_step4_explicit_drug_classes`, `drug_class_step4_extraction_details`, `drug_class_step5_refined_explicit_classes`, `drug_class_step5_removed_classes`, `drug_class_step5_reasoning`
- Drug class validation: `drug_class_validation_missed_drug_classes`

## Storage

### Local Storage
Use local file paths:
```bash
--input data/Conference/abstract_titles.csv
--output_dir data/Conference/drug
```

### GCS Storage
Use `gs://` prefix:
```bash
--input gs://bucket-name/Conference/abstract_titles.csv
--output_dir gs://bucket-name/Conference/drug
```

The system automatically detects storage type from path prefix.

## Output Structure

```
{conference_name}/
├── input/
│   └── abstract_titles.csv
├── indication/
│   ├── batch_status.json
│   ├── abstracts/
│   │   └── {abstract_id}/
│   │       ├── extraction.json
│   │       ├── validation.json
│   │       └── status.json
│   └── extraction_*.csv
├── drug/
│   └── (same structure)
└── drug_class/
    ├── extraction_batch_status.json
    ├── validation_batch_status.json
    └── abstracts/
        └── {abstract_id}/
            ├── step1_output.json
            ├── step2_output.json
            ├── step3_output.json
            ├── step4_output.json
            ├── step5_output.json
            ├── status.json
            ├── validation.json
            └── validation_{drug}.json
```

## Common Options

- `--limit N`: Process only first N abstracts
- `--parallel_workers N`: Number of parallel workers (default: 5-50 depending on pipeline)

## License

MIT
