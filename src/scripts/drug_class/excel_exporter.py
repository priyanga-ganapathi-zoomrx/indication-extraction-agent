#!/usr/bin/env python3
"""
Drug Class Excel Exporter

Reads all step outputs (Step1-Step5) and validation results from the drug class
extraction pipeline and generates an Excel file with separate sheets for each step.

Each sheet contains:
- All columns from the input CSV
- Step-specific columns with drug data grouped by drug name (as JSON objects)

Key features:
- One row per abstract (not per drug)
- Drug-specific data grouped as JSON objects with drug names as keys
- Support for both local and GCS storage
- Progress bar with tqdm

Usage:
    # Local storage
    python -m src.scripts.drug_class.excel_exporter \\
        --input data/ASCO2025/input/abstract_titles.csv \\
        --output_dir data/ASCO2025/drug_class \\
        --excel_output drug_class_export.xlsx

    # GCS storage
    python -m src.scripts.drug_class.excel_exporter \\
        --input gs://bucket/ASCO2025/input/abstract_titles.csv \\
        --output_dir gs://bucket/ASCO2025/drug_class \\
        --excel_output drug_class_export.xlsx
"""

import argparse
import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from tqdm import tqdm

from src.agents.core.storage import LocalStorageClient, GCSStorageClient, get_storage_client


def _get_timestamp() -> str:
    """Get current timestamp for filename generation."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _to_json_string(data: Any) -> str:
    """Convert data to pretty-printed JSON string for Excel cell."""
    if data is None:
        return ""
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(data)


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use in filenames.
    
    Replaces characters that are problematic in file paths.
    """
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def load_abstracts_from_csv(
    csv_filename: str,
    input_storage: Union[LocalStorageClient, GCSStorageClient],
    limit: Optional[int] = None,
) -> tuple[list[dict], list[str]]:
    """Load abstracts from input CSV file.
    
    Args:
        csv_filename: Filename of the CSV within input_storage
        input_storage: Storage client for reading input CSV
        limit: Optional limit on rows to process
        
    Returns:
        tuple: (list of row dicts, list of fieldnames/column headers)
    """
    csv_content = input_storage.download_text(csv_filename)
    reader = csv.DictReader(io.StringIO(csv_content))
    fieldnames = list(reader.fieldnames or [])
    
    rows = list(reader)
    
    if limit:
        rows = rows[:limit]
    
    return rows, fieldnames


def get_abstract_id_column(fieldnames: list[str]) -> Optional[str]:
    """Find the abstract_id column name (case-insensitive)."""
    header_map = {h.lower().strip(): h for h in fieldnames}
    return header_map.get('abstract_id') or header_map.get('id')


def read_step_output(
    storage: Union[LocalStorageClient, GCSStorageClient],
    abstract_id: str,
    step_name: str,
) -> Optional[dict]:
    """Read a step output JSON file for an abstract.
    
    Args:
        storage: Storage client for output directory
        abstract_id: The abstract ID
        step_name: Name of step file (e.g., 'step1_output', 'validation')
        
    Returns:
        dict if file exists, None otherwise
    """
    try:
        return storage.download_json(f"abstracts/{abstract_id}/{step_name}.json")
    except FileNotFoundError:
        return None


def transform_step1(step1_data: Optional[dict]) -> dict:
    """Transform Step1 output to Excel columns.
    
    Step1Output contains:
    - drug_to_components: dict[str, list[str]] - maps drug name to components
    
    Output columns:
    - drug_to_components: JSON object with drug names as keys
    """
    if not step1_data:
        return {"drug_to_components": ""}
    
    drug_to_components = step1_data.get("drug_to_components", {})
    
    return {
        "drug_to_components": _to_json_string(drug_to_components),
    }


def transform_step2(step2_data: Optional[dict]) -> dict:
    """Transform Step2 output to Excel columns.
    
    Step2Output contains:
    - extractions: dict[str, DrugExtractionResult]
    
    DrugExtractionResult has: drug_classes, selected_sources, confidence_score,
    extraction_method, extraction_details, reasoning, success
    
    Output columns (each as JSON object with drug names as keys):
    - drug_classes
    - selected_sources
    - confidence_score
    - extraction_method
    - extraction_details
    - reasoning
    - success
    """
    if not step2_data:
        return {
            "drug_classes": "",
            "selected_sources": "",
            "confidence_score": "",
            "extraction_method": "",
            "extraction_details_step2": "",
            "reasoning_step2": "",
            "success": "",
        }
    
    extractions = step2_data.get("extractions", {})
    
    drug_classes = {}
    selected_sources = {}
    confidence_score = {}
    extraction_method = {}
    extraction_details = {}
    reasoning = {}
    success = {}
    
    for drug_name, result in extractions.items():
        if isinstance(result, dict):
            drug_classes[drug_name] = result.get("drug_classes", [])
            selected_sources[drug_name] = result.get("selected_sources", [])
            confidence_score[drug_name] = result.get("confidence_score", 0.0)
            extraction_method[drug_name] = result.get("extraction_method", "")
            extraction_details[drug_name] = result.get("extraction_details", [])
            reasoning[drug_name] = result.get("reasoning", "")
            success[drug_name] = result.get("success", False)
    
    return {
        "drug_classes": _to_json_string(drug_classes),
        "selected_sources": _to_json_string(selected_sources),
        "confidence_score": _to_json_string(confidence_score),
        "extraction_method": _to_json_string(extraction_method),
        "extraction_details_step2": _to_json_string(extraction_details),
        "reasoning_step2": _to_json_string(reasoning),
        "success": _to_json_string(success),
    }


def transform_step3(step3_data: Optional[dict]) -> dict:
    """Transform Step3 output to Excel columns.
    
    Step3Output contains:
    - selections: dict[str, DrugSelectionResult]
    
    DrugSelectionResult has: selected_drug_classes, reasoning
    
    Output columns (each as JSON object with drug names as keys):
    - selected_drug_classes
    - reasoning
    """
    if not step3_data:
        return {
            "selected_drug_classes": "",
            "reasoning_step3": "",
        }
    
    selections = step3_data.get("selections", {})
    
    selected_drug_classes = {}
    reasoning = {}
    
    for drug_name, result in selections.items():
        if isinstance(result, dict):
            selected_drug_classes[drug_name] = result.get("selected_drug_classes", [])
            reasoning[drug_name] = result.get("reasoning", "")
    
    return {
        "selected_drug_classes": _to_json_string(selected_drug_classes),
        "reasoning_step3": _to_json_string(reasoning),
    }


def transform_step4(step4_data: Optional[dict]) -> dict:
    """Transform Step4 output to Excel columns.
    
    Step4Output contains:
    - explicit_drug_classes: list[str]
    - extraction_details: list[ExtractionDetail]
    - reasoning: str
    
    Output columns:
    - explicit_drug_classes: JSON array
    - extraction_details: JSON array
    - reasoning
    """
    if not step4_data:
        return {
            "explicit_drug_classes": "",
            "extraction_details": "",
            "reasoning_step4": "",
        }
    
    return {
        "explicit_drug_classes": _to_json_string(step4_data.get("explicit_drug_classes", [])),
        "extraction_details": _to_json_string(step4_data.get("extraction_details", [])),
        "reasoning_step4": step4_data.get("reasoning", ""),
    }


def transform_step5(step5_data: Optional[dict]) -> dict:
    """Transform Step5 output to Excel columns.
    
    Step5Output contains:
    - refined_explicit_classes: list[str]
    - removed_classes: list[str]
    - reasoning: str
    
    Output columns:
    - refined_explicit_classes: JSON array
    - removed_classes: JSON array
    - reasoning
    """
    if not step5_data:
        return {
            "refined_explicit_classes": "",
            "removed_classes": "",
            "reasoning_step5": "",
        }
    
    return {
        "refined_explicit_classes": _to_json_string(step5_data.get("refined_explicit_classes", [])),
        "removed_classes": _to_json_string(step5_data.get("removed_classes", [])),
        "reasoning_step5": step5_data.get("reasoning", ""),
    }


def transform_validation(
    storage: Union[LocalStorageClient, GCSStorageClient],
    abstract_id: str,
    step2_data: Optional[dict],
) -> dict:
    """Transform validation output to Excel columns.
    
    Reads per-drug validation files: validation_{sanitized_drug}.json
    
    Each per-drug validation file (ValidationOutput) contains:
    - validation_status: str (PASS/REVIEW/FAIL)
    - validation_confidence: float
    - missed_drug_classes: list[str]
    - issues_found: list[ValidationIssue]
    - checks_performed: ChecksPerformed
    - validation_reasoning: str
    - llm_calls: int
    - validation_success: bool
    
    Output columns (each as JSON object with drug names as keys):
    - validation_status
    - validation_confidence
    - missed_drug_classes
    - issues_found
    - checks_performed
    - validation_reasoning
    - llm_calls
    - validation_success
    """
    empty_result = {
        "validation_status": "",
        "validation_confidence": "",
        "missed_drug_classes": "",
        "issues_found": "",
        "checks_performed": "",
        "validation_reasoning": "",
        "llm_calls": "",
        "validation_success": "",
    }
    
    if not step2_data:
        return empty_result
    
    # Get drug names from step2 extractions
    extractions = step2_data.get("extractions", {})
    drug_names = list(extractions.keys())
    
    if not drug_names:
        return empty_result
    
    # Initialize collectors for each field
    validation_status = {}
    validation_confidence = {}
    missed_drug_classes = {}
    issues_found = {}
    checks_performed = {}
    validation_reasoning = {}
    llm_calls = {}
    validation_success = {}
    
    # Read each per-drug validation file
    for drug_name in drug_names:
        safe_drug_name = _sanitize_filename(drug_name)
        try:
            val_data = storage.download_json(
                f"abstracts/{abstract_id}/validation_{safe_drug_name}.json"
            )
            validation_status[drug_name] = val_data.get("validation_status", "")
            validation_confidence[drug_name] = val_data.get("validation_confidence", 0.0)
            missed_drug_classes[drug_name] = val_data.get("missed_drug_classes", [])
            issues_found[drug_name] = val_data.get("issues_found", [])
            checks_performed[drug_name] = val_data.get("checks_performed", {})
            validation_reasoning[drug_name] = val_data.get("validation_reasoning", "")
            llm_calls[drug_name] = val_data.get("llm_calls", 0)
            validation_success[drug_name] = val_data.get("validation_success", False)
        except FileNotFoundError:
            # Drug validation file not found - skip this drug
            pass
    
    return {
        "validation_status": _to_json_string(validation_status),
        "validation_confidence": _to_json_string(validation_confidence),
        "missed_drug_classes": _to_json_string(missed_drug_classes),
        "issues_found": _to_json_string(issues_found),
        "checks_performed": _to_json_string(checks_performed),
        "validation_reasoning": _to_json_string(validation_reasoning),
        "llm_calls": _to_json_string(llm_calls),
        "validation_success": _to_json_string(validation_success),
    }


def process_abstracts(
    rows: list[dict],
    fieldnames: list[str],
    output_storage: Union[LocalStorageClient, GCSStorageClient],
) -> dict[str, list[dict]]:
    """Process all abstracts and build data for each sheet.
    
    Args:
        rows: List of input CSV rows
        fieldnames: Column names from input CSV
        output_storage: Storage client for reading step outputs
        
    Returns:
        dict with keys 'Step1', 'Step2', 'Step3', 'Step4', 'Step5', 'Validation'
        each containing a list of row dicts
    """
    id_col = get_abstract_id_column(fieldnames)
    
    sheet_data = {
        "Step1": [],
        "Step2": [],
        "Step3": [],
        "Step4": [],
        "Step5": [],
        "Validation": [],
    }
    
    stats = {
        "step1_found": 0,
        "step2_found": 0,
        "step3_found": 0,
        "step4_found": 0,
        "step5_found": 0,
        "validation_found": 0,
    }
    
    for row in tqdm(rows, desc="Processing abstracts"):
        abstract_id = row.get(id_col, "") if id_col else ""
        
        if not abstract_id:
            continue
        
        # Read all step outputs
        step1_data = read_step_output(output_storage, abstract_id, "step1_output")
        step2_data = read_step_output(output_storage, abstract_id, "step2_output")
        step3_data = read_step_output(output_storage, abstract_id, "step3_output")
        step4_data = read_step_output(output_storage, abstract_id, "step4_output")
        step5_data = read_step_output(output_storage, abstract_id, "step5_output")
        
        # Update stats
        if step1_data: stats["step1_found"] += 1
        if step2_data: stats["step2_found"] += 1
        if step3_data: stats["step3_found"] += 1
        if step4_data: stats["step4_found"] += 1
        if step5_data: stats["step5_found"] += 1
        
        # Transform each step
        step1_cols = transform_step1(step1_data)
        step2_cols = transform_step2(step2_data)
        step3_cols = transform_step3(step3_data)
        step4_cols = transform_step4(step4_data)
        step5_cols = transform_step5(step5_data)
        
        # Transform validation (reads per-drug validation files)
        validation_cols = transform_validation(output_storage, abstract_id, step2_data)
        
        # Update validation stats (check if any validation data was found)
        if any(validation_cols.get("validation_status", "")):
            stats["validation_found"] += 1
        
        # Build row for each sheet (input columns + step-specific columns)
        sheet_data["Step1"].append({**row, **step1_cols})
        sheet_data["Step2"].append({**row, **step2_cols})
        sheet_data["Step3"].append({**row, **step3_cols})
        sheet_data["Step4"].append({**row, **step4_cols})
        sheet_data["Step5"].append({**row, **step5_cols})
        sheet_data["Validation"].append({**row, **validation_cols})
    
    # Print stats
    total = len(rows)
    print(f"\nStep output availability:")
    print(f"  Step1: {stats['step1_found']}/{total}")
    print(f"  Step2: {stats['step2_found']}/{total}")
    print(f"  Step3: {stats['step3_found']}/{total}")
    print(f"  Step4: {stats['step4_found']}/{total}")
    print(f"  Step5: {stats['step5_found']}/{total}")
    print(f"  Validation: {stats['validation_found']}/{total}")
    
    return sheet_data


def export_to_excel(
    sheet_data: dict[str, list[dict]],
    output_path: str,
) -> None:
    """Export data to Excel file with separate sheets.
    
    Args:
        sheet_data: dict with sheet names as keys and list of row dicts as values
        output_path: Path to output Excel file
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, rows in sheet_data.items():
            if rows:
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                # Create empty sheet with header
                df = pd.DataFrame()
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\nExcel file saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export drug class extraction pipeline outputs to Excel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Local storage
    python -m src.scripts.drug_class.excel_exporter \\
        --input data/ASCO2025/input/abstract_titles.csv \\
        --output_dir data/ASCO2025/drug_class \\
        --excel_output drug_class_export.xlsx

    # GCS storage
    python -m src.scripts.drug_class.excel_exporter \\
        --input gs://bucket/ASCO2025/input/abstract_titles.csv \\
        --output_dir gs://bucket/ASCO2025/drug_class \\
        --excel_output drug_class_export.xlsx
        
    # With limit
    python -m src.scripts.drug_class.excel_exporter \\
        --input data/ASCO2025/input/abstract_titles.csv \\
        --output_dir data/ASCO2025/drug_class \\
        --limit 10
        """
    )
    
    parser.add_argument(
        "--input",
        default="gs://entity-extraction-agent-data-dev/Conference/abstract_titles.csv",
        help="Input CSV file path with abstract metadata"
    )
    parser.add_argument(
        "--output_dir",
        default="gs://entity-extraction-agent-data-dev/Conference/drug_class",
        help="Output directory where step outputs are stored"
    )
    parser.add_argument(
        "--excel_output",
        default="data/drug_class_export_{timestamp}.xlsx",
        help="Output Excel file path (default: drug_class_export_{timestamp}.xlsx)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of abstracts to process"
    )
    
    args = parser.parse_args()
    
    # Determine input storage client
    # Input path could be GCS (gs://bucket/path/file.csv) or local (dir/file.csv)
    if args.input.startswith("gs://"):
        # GCS input: parse bucket and get parent directory as base
        from src.agents.core.storage import parse_gcs_path
        bucket, full_prefix = parse_gcs_path(args.input)
        # Extract directory and filename
        if "/" in full_prefix:
            base_prefix = "/".join(full_prefix.split("/")[:-1])
            csv_filename = full_prefix.split("/")[-1]
        else:
            base_prefix = ""
            csv_filename = full_prefix
        input_storage = GCSStorageClient(bucket, base_prefix)
    else:
        # Local input: parent directory is base, filename is the file
        input_path = Path(args.input)
        input_storage = LocalStorageClient(str(input_path.parent) if input_path.parent != Path(".") else ".")
        csv_filename = input_path.name
    
    # Output storage client
    output_storage = get_storage_client(args.output_dir)
    
    # Generate output filename if not provided
    excel_output = args.excel_output
    if not excel_output:
        excel_output = f"drug_class_export_{_get_timestamp()}.xlsx"
    
    print(f"Input CSV: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Excel output: {excel_output}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print()
    
    # Load abstracts from CSV
    print("Loading abstracts from CSV...")
    rows, fieldnames = load_abstracts_from_csv(csv_filename, input_storage, args.limit)
    print(f"Loaded {len(rows)} abstracts")
    print(f"Input columns: {fieldnames}")
    print()
    
    if not rows:
        print("No abstracts to process.")
        return
    
    # Process abstracts and build sheet data
    sheet_data = process_abstracts(rows, fieldnames, output_storage)
    
    # Export to Excel
    export_to_excel(sheet_data, excel_output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
