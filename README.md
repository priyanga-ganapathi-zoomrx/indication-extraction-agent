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
