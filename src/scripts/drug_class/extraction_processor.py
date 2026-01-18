#!/usr/bin/env python3
"""
Drug Class Extraction Processor (Step-Centric)

Processes multiple abstracts through the drug class extraction pipeline
using step-centric batching for optimal parallelism control.

Key features:
- Step-centric: All abstracts go through Step 1, then Step 2, etc.
- Per-step parallelism: Configure different parallelism per step based on token usage
- Per-abstract status tracking: Resume from where each abstract left off
- Retry-friendly: Just rerun to retry failed abstracts

Usage:
    python -m src.scripts.drug_class.extraction_processor --input data/input.csv --output_dir data/output
"""

import argparse
import csv
import json
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.drug_class import (
    # Step functions
    identify_regimen,
    fetch_search_results,
    extract_with_tavily,
    extract_with_grounded,
    select_drug_class,
    needs_llm_selection,
    extract_explicit_classes,
    consolidate_drug_classes,
    # Schemas
    DrugClassInput,
    RegimenInput,
    DrugClassExtractionInput,
    SelectionInput,
    ExplicitExtractionInput,
    ConsolidationInput,
    Step1Output,
    Step2Output,
    Step3Output,
    Step4Output,
    PipelineStatus,
)
from src.agents.drug import ExtractionResult
from src.agents.core.storage import LocalStorageClient


def _get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class StepConfig:
    """Configuration for a pipeline step."""
    name: str
    parallelism: int = 10
    description: str = ""


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    step1: StepConfig = field(default_factory=lambda: StepConfig("step1_regimen", 50, "Regimen ID"))
    step2: StepConfig = field(default_factory=lambda: StepConfig("step2_extraction", 10, "Drug class extraction"))
    step3: StepConfig = field(default_factory=lambda: StepConfig("step3_selection", 30, "Class selection"))
    step4: StepConfig = field(default_factory=lambda: StepConfig("step4_explicit", 25, "Explicit extraction"))
    step5: StepConfig = field(default_factory=lambda: StepConfig("step5_consolidation", 30, "Consolidation"))


def load_abstracts(
    csv_path: str,
    drug_storage: LocalStorageClient,
    limit: int = None
) -> tuple[list[DrugClassInput], list[dict], list[str]]:
    """Load abstracts from CSV and drug extraction results from storage.
    
    CSV provides: abstract_id, abstract_title, firm, full_abstract
    Drug data comes from: drug_storage/abstracts/{abstract_id}/extraction.json
    
    Returns:
        tuple: (inputs, original_rows, fieldnames)
    """
    inputs = []
    original_rows = []
    fieldnames = []
    skipped_no_drugs = []
    
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        
        # Find column names (case-insensitive)
        header_map = {h.lower().strip(): h for h in fieldnames}
        id_col = header_map.get('abstract_id') or header_map.get('id')
        title_col = header_map.get('abstract_title') or header_map.get('title')
        abstract_col = header_map.get('full_abstract') or header_map.get('abstract')
        firms_col = header_map.get('firms') or header_map.get('firm') or header_map.get('sponsor')
        
        for row in reader:
            abstract_id = row.get(id_col, "") if id_col else ""
            abstract_title = row.get(title_col, "") if title_col else ""
            full_abstract = row.get(abstract_col, "") if abstract_col else ""
            
            if not abstract_id or not abstract_title:
                continue
            
            # Read drug extraction from storage
            try:
                drug_data = drug_storage.download_json(f"abstracts/{abstract_id}/extraction.json")
                extraction = ExtractionResult(**drug_data)
            except FileNotFoundError:
                skipped_no_drugs.append(abstract_id)
                continue
            
            # Parse firms
            firms = _parse_list(row.get(firms_col, "") if firms_col else "")
            
            inputs.append(DrugClassInput(
                abstract_id=str(abstract_id),
                abstract_title=str(abstract_title),
                full_abstract=str(full_abstract),
                primary_drugs=extraction.primary_drugs,
                secondary_drugs=extraction.secondary_drugs,
                comparator_drugs=extraction.comparator_drugs,
                firms=firms,
            ))
            original_rows.append(row)
    
    if skipped_no_drugs:
        print(f"⚠ Skipped {len(skipped_no_drugs)} abstracts without drug extraction: {skipped_no_drugs[:5]}{'...' if len(skipped_no_drugs) > 5 else ''}")
    
    if limit:
        return inputs[:limit], original_rows[:limit], fieldnames
    return inputs, original_rows, fieldnames


def _parse_list(value: str) -> list[str]:
    """Parse list from JSON array or comma-separated string."""
    if not value or not value.strip():
        return []
    
    value = value.strip()
    
    # Try JSON array first
    if value.startswith('['):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(d).strip() for d in parsed if d and str(d).strip()]
        except json.JSONDecodeError:
            pass
    
    # Fall back to comma/semicolon separated
    return [d.strip() for d in value.replace(';', ',').split(',') if d.strip()]


def get_abstract_status(abstract_id: str, storage: LocalStorageClient) -> Optional[PipelineStatus]:
    """Load status for an abstract if it exists."""
    try:
        status_data = storage.download_json(f"abstracts/{abstract_id}/status.json")
        return PipelineStatus(**status_data)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def get_step_from_status(status: Optional[PipelineStatus]) -> str:
    """Determine which step an abstract needs based on its status."""
    if not status:
        return "step1_regimen"
    
    steps_order = ["step1_regimen", "step2_extraction", "step3_selection", "step4_explicit", "step5_consolidation"]
    
    for step in steps_order:
        step_data = status.steps.get(step, {})
        step_status = step_data.get("status", "pending")
        
        if step_status in ["pending", "running", "failed"]:
            return step
    
    # All steps complete
    return "complete"


def get_abstracts_at_step(
    inputs: list[DrugClassInput],
    step_name: str,
    storage: LocalStorageClient
) -> list[DrugClassInput]:
    """Get abstracts that need a specific step."""
    pending = []
    
    for inp in inputs:
        status = get_abstract_status(inp.abstract_id, storage)
        current_step = get_step_from_status(status)
        
        if current_step == step_name:
            pending.append(inp)
    
    return pending


def process_step1_single(inp: DrugClassInput, storage: LocalStorageClient) -> dict:
    """Process Step 1 for a single abstract.
    
    Uses per-drug error handling to preserve partial progress.
    """
    abstract_id = inp.abstract_id
    
    # Initialize or load status
    status = get_abstract_status(abstract_id, storage)
    if not status:
        status = PipelineStatus(abstract_id=abstract_id, abstract_title=inp.abstract_title)
        status.last_updated = _get_timestamp()
    
    # Get all drugs
    all_drugs = inp.primary_drugs + inp.secondary_drugs + inp.comparator_drugs
    if not all_drugs:
        return {"abstract_id": abstract_id, "success": False, "error": "No drugs to process"}
    
    # Load existing step1 output if available, or create new
    try:
        existing_data = storage.download_json(f"abstracts/{abstract_id}/step1_output.json")
        step1_output = Step1Output(**existing_data)
    except FileNotFoundError:
        step1_output = Step1Output()
    
    step1_llm_calls = 0
    
    for drug in all_drugs:
        if step1_output.is_drug_done(drug):
            continue
        
        try:
            components = identify_regimen(RegimenInput(
                abstract_id=abstract_id,
                abstract_title=inp.abstract_title,
                drug=drug,
            ))
            step1_output.mark_success(drug, components)
            step1_llm_calls += 1
        except Exception as e:
            # Mark individual drug as failed, continue to others
            step1_output.mark_failed(drug, str(e))
        
        # Save progress after each drug (incremental checkpoint)
        storage.upload_json(f"abstracts/{abstract_id}/step1_output.json", step1_output.model_dump())
    
    # Update status based on completion
    if step1_output.is_complete(all_drugs):
        status.steps["step1_regimen"] = {
            "status": "success",
            "llm_calls": step1_llm_calls,
        }
        status.last_completed_step = "step1_regimen"
    else:
        failed_drugs = [d for d in all_drugs if step1_output.drug_status.get(d) == "failed"]
        status.steps["step1_regimen"] = {"status": "failed", "error": f"Failed drugs: {failed_drugs}"}
    
    status.total_llm_calls += step1_llm_calls
    status.last_updated = _get_timestamp()
    storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
    
    success_count = sum(1 for d in all_drugs if step1_output.drug_status.get(d) == "success")
    return {
        "abstract_id": abstract_id,
        "success": step1_output.is_complete(all_drugs),
        "llm_calls": step1_llm_calls,
        "drugs_succeeded": success_count,
        "drugs_failed": len(all_drugs) - success_count,
    }


def process_step2_single(inp: DrugClassInput, storage: LocalStorageClient) -> dict:
    """Process Step 2 for a single abstract.
    
    Uses per-drug error handling to preserve partial progress.
    """
    abstract_id = inp.abstract_id
    
    # Load status and step1 output
    status = get_abstract_status(abstract_id, storage)
    if not status:
        return {"abstract_id": abstract_id, "success": False, "error": "No status found"}
    
    try:
        step1_data = storage.download_json(f"abstracts/{abstract_id}/step1_output.json")
        step1_output = Step1Output(**step1_data)
    except FileNotFoundError:
        return {"abstract_id": abstract_id, "success": False, "error": "Step 1 output not found"}
    
    all_components = step1_output.get_all_components()
    
    if not all_components:
        return {"abstract_id": abstract_id, "success": False, "error": "No components from Step 1"}
    
    # Load existing step2 output if available, or create new
    try:
        existing_data = storage.download_json(f"abstracts/{abstract_id}/step2_output.json")
        step2_output = Step2Output(**existing_data)
    except FileNotFoundError:
        step2_output = Step2Output()
    
    step2_llm_calls = 0
    
    for drug in all_components:
        if step2_output.is_drug_done(drug):
            continue
        
        try:
            # Fetch search results (with caching)
            drug_results, firm_results = fetch_search_results(drug, inp.firms, storage)
            
            # Try Tavily extraction first
            result = extract_with_tavily(DrugClassExtractionInput(
                abstract_id=abstract_id,
                abstract_title=inp.abstract_title,
                drug=drug,
                full_abstract=inp.full_abstract,
                firms=inp.firms,
                drug_class_results=drug_results,
                firm_search_results=firm_results,
            ))
            step2_llm_calls += 1
            
            # Fallback to grounded search if needed
            if not result.drug_classes or result.drug_classes == ["NA"]:
                result = extract_with_grounded(DrugClassExtractionInput(
                    abstract_id=abstract_id,
                    abstract_title=inp.abstract_title,
                    drug=drug,
                    full_abstract=inp.full_abstract,
                    firms=inp.firms,
                    drug_class_results=drug_results,
                    firm_search_results=firm_results,
                ))
                step2_llm_calls += 1
            
            step2_output.mark_success(drug, result)
            
        except Exception as e:
            # Mark individual drug as failed, continue to others
            step2_output.mark_failed(drug, str(e))
        
        # Save progress after each drug (incremental checkpoint)
        storage.upload_json(f"abstracts/{abstract_id}/step2_output.json", step2_output.model_dump())
    
    # Update status based on completion
    if step2_output.is_complete(all_components):
        status.steps["step2_extraction"] = {
            "status": "success",
            "llm_calls": step2_llm_calls,
        }
        status.last_completed_step = "step2_extraction"
    else:
        failed_drugs = [d for d in all_components if step2_output.drug_status.get(d) == "failed"]
        status.steps["step2_extraction"] = {"status": "failed", "error": f"Failed drugs: {failed_drugs}"}
    
    status.total_llm_calls += step2_llm_calls
    status.last_updated = _get_timestamp()
    storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
    
    success_count = sum(1 for d in all_components if step2_output.drug_status.get(d) == "success")
    return {
        "abstract_id": abstract_id,
        "success": step2_output.is_complete(all_components),
        "llm_calls": step2_llm_calls,
        "drugs_succeeded": success_count,
        "drugs_failed": len(all_components) - success_count,
    }


def process_step3_single(inp: DrugClassInput, storage: LocalStorageClient) -> dict:
    """Process Step 3 for a single abstract.
    
    Uses per-drug error handling to preserve partial progress.
    """
    abstract_id = inp.abstract_id
    
    status = get_abstract_status(abstract_id, storage)
    if not status:
        return {"abstract_id": abstract_id, "success": False, "error": "No status found"}
    
    try:
        step2_data = storage.download_json(f"abstracts/{abstract_id}/step2_output.json")
        step2_output = Step2Output(**step2_data)
    except FileNotFoundError:
        return {"abstract_id": abstract_id, "success": False, "error": "Step 2 output not found"}
    
    # Load existing step3 output if available, or create new
    try:
        existing_data = storage.download_json(f"abstracts/{abstract_id}/step3_output.json")
        step3_output = Step3Output(**existing_data)
    except FileNotFoundError:
        step3_output = Step3Output()
    
    step3_llm_calls = 0
    all_drugs = list(step2_output.extractions.keys())
    
    for drug_name, extraction_result in step2_output.extractions.items():
        if step3_output.is_drug_done(drug_name):
            continue
        
        try:
            # Check if LLM selection is needed
            if not needs_llm_selection(extraction_result.extraction_details):
                # No LLM needed - just copy the single class
                from src.agents.drug_class.schemas import DrugSelectionResult
                if extraction_result.extraction_details:
                    first_detail = extraction_result.extraction_details[0]
                    selected = [first_detail.normalized_form or first_detail.extracted_text or "NA"]
                else:
                    selected = extraction_result.drug_classes or ["NA"]
                
                result = DrugSelectionResult(
                    drug_name=drug_name,
                    selected_drug_classes=selected,
                    reasoning="Single class - no selection needed",
                )
            else:
                result = select_drug_class(SelectionInput(
                    abstract_id=abstract_id,
                    drug_name=drug_name,
                    extraction_details=extraction_result.extraction_details,
                ))
                step3_llm_calls += 1
            
            step3_output.mark_success(drug_name, result)
            
        except Exception as e:
            # Mark individual drug as failed, continue to others
            step3_output.mark_failed(drug_name, str(e))
        
        # Save progress after each drug (incremental checkpoint)
        storage.upload_json(f"abstracts/{abstract_id}/step3_output.json", step3_output.model_dump())
    
    # Update status based on completion
    if step3_output.is_complete(all_drugs):
        status.steps["step3_selection"] = {
            "status": "success",
            "llm_calls": step3_llm_calls,
        }
        status.last_completed_step = "step3_selection"
    else:
        failed_drugs = [d for d in all_drugs if step3_output.drug_status.get(d) == "failed"]
        status.steps["step3_selection"] = {"status": "failed", "error": f"Failed drugs: {failed_drugs}"}
    
    status.total_llm_calls += step3_llm_calls
    status.last_updated = _get_timestamp()
    storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
    
    success_count = sum(1 for d in all_drugs if step3_output.drug_status.get(d) == "success")
    return {
        "abstract_id": abstract_id,
        "success": step3_output.is_complete(all_drugs),
        "llm_calls": step3_llm_calls,
        "drugs_succeeded": success_count,
        "drugs_failed": len(all_drugs) - success_count,
    }


def process_step4_single(inp: DrugClassInput, storage: LocalStorageClient) -> dict:
    """Process Step 4 for a single abstract."""
    abstract_id = inp.abstract_id
    
    status = get_abstract_status(abstract_id, storage)
    if not status:
        return {"abstract_id": abstract_id, "success": False, "error": "No status found"}
    
    step4_llm_calls = 0
    
    try:
        step4_output = extract_explicit_classes(ExplicitExtractionInput(
            abstract_id=abstract_id,
            abstract_title=inp.abstract_title,
        ))
        
        # Count LLM call only if title was non-empty
        if inp.abstract_title and inp.abstract_title.strip():
            step4_llm_calls = 1
        
        status.steps["step4_explicit"] = {
            "status": "success",
            "llm_calls": step4_llm_calls,
        }
        status.last_completed_step = "step4_explicit"
        status.total_llm_calls += step4_llm_calls
        status.last_updated = _get_timestamp()
        
        storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
        storage.upload_json(f"abstracts/{abstract_id}/step4_output.json", step4_output.model_dump())
        
        return {"abstract_id": abstract_id, "success": True, "llm_calls": step4_llm_calls}
        
    except Exception as e:
        status.steps["step4_explicit"] = {"status": "failed", "error": str(e)}
        status.last_updated = _get_timestamp()
        storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
        return {"abstract_id": abstract_id, "success": False, "error": str(e)}


def process_step5_single(inp: DrugClassInput, storage: LocalStorageClient) -> dict:
    """Process Step 5 for a single abstract."""
    abstract_id = inp.abstract_id
    
    status = get_abstract_status(abstract_id, storage)
    if not status:
        return {"abstract_id": abstract_id, "success": False, "error": "No status found"}
    
    # Load step3 and step4 outputs
    try:
        step3_data = storage.download_json(f"abstracts/{abstract_id}/step3_output.json")
        step4_data = storage.download_json(f"abstracts/{abstract_id}/step4_output.json")
        step3_output = Step3Output(**step3_data)
        step4_output = Step4Output(**step4_data)
    except FileNotFoundError:
        return {"abstract_id": abstract_id, "success": False, "error": "Previous step outputs not found"}
    
    step5_llm_calls = 0
    
    try:
        # Build drug selections for consolidation
        drug_selections = [
            {"drug_name": s.drug_name, "selected_classes": s.selected_drug_classes}
            for s in step3_output.get_results_list()
        ]
        
        step5_output = consolidate_drug_classes(ConsolidationInput(
            abstract_id=abstract_id,
            abstract_title=inp.abstract_title,
            explicit_drug_classes=step4_output.explicit_drug_classes,
            drug_selections=drug_selections,
        ))
        
        # Count LLM call only if consolidation was needed
        has_explicit = step4_output.explicit_drug_classes and step4_output.explicit_drug_classes != ["NA"]
        has_selections = bool(drug_selections)
        if has_explicit and has_selections:
            step5_llm_calls = 1
        
        status.steps["step5_consolidation"] = {
            "status": "success",
            "llm_calls": step5_llm_calls,
        }
        status.last_completed_step = "step5_consolidation"
        status.pipeline_status = "success"
        status.total_llm_calls += step5_llm_calls
        status.last_updated = _get_timestamp()
        
        storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
        storage.upload_json(f"abstracts/{abstract_id}/step5_output.json", step5_output.model_dump())
        
        return {"abstract_id": abstract_id, "success": True, "llm_calls": step5_llm_calls}
        
    except Exception as e:
        status.steps["step5_consolidation"] = {"status": "failed", "error": str(e)}
        status.last_updated = _get_timestamp()
        storage.upload_json(f"abstracts/{abstract_id}/status.json", status.to_dict())
        return {"abstract_id": abstract_id, "success": False, "error": str(e)}


def run_step_batch(
    step_name: str,
    process_func,
    inputs: list[DrugClassInput],
    storage: LocalStorageClient,
    parallelism: int,
) -> tuple[int, int]:
    """Run a step for all pending abstracts.
    
    Returns:
        tuple: (successful_count, failed_count)
    """
    pending = get_abstracts_at_step(inputs, step_name, storage)
    
    if not pending:
        print(f"  No abstracts pending for {step_name}")
        return 0, 0
    
    print(f"  Processing {len(pending)} abstracts (parallelism: {parallelism})")
    
    successful = 0
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_to_inp = {
            executor.submit(process_func, inp, storage): inp
            for inp in pending
        }
        
        for future in concurrent.futures.as_completed(future_to_inp):
            inp = future_to_inp[future]
            try:
                result = future.result()
                if result.get("success"):
                    successful += 1
                    print(f"    ✓ {inp.abstract_id}")
                else:
                    failed += 1
                    print(f"    ✗ {inp.abstract_id}: {result.get('error', 'unknown')}")
            except Exception as e:
                failed += 1
                print(f"    ✗ {inp.abstract_id}: {e}")
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(description="Drug Class Extraction Processor (Step-Centric)")
    parser.add_argument("--input", default="data/drug_class/input/abstract_titles.csv", help="Input CSV file with abstracts")
    parser.add_argument("--drug_output_dir", default="data/drug/output", help="Directory containing drug extraction results")
    parser.add_argument("--output_dir", default="data/drug_class/output", help="Output directory for results")
    parser.add_argument("--limit", type=int, default=None, help="Limit abstracts to process")
    
    # Per-step parallelism
    parser.add_argument("--step1_parallelism", type=int, default=50, help="Parallelism for Step 1")
    parser.add_argument("--step2_parallelism", type=int, default=10, help="Parallelism for Step 2 (expensive)")
    parser.add_argument("--step3_parallelism", type=int, default=30, help="Parallelism for Step 3")
    parser.add_argument("--step4_parallelism", type=int, default=25, help="Parallelism for Step 4")
    parser.add_argument("--step5_parallelism", type=int, default=30, help="Parallelism for Step 5")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧬 Drug Class Extraction Processor (Step-Centric)")
    print("=" * 60)
    print(f"Input:          {args.input}")
    print(f"Drug data from: {args.drug_output_dir}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Limit:          {args.limit or 'all'}")
    print()
    print("Per-step parallelism:")
    print(f"  Step 1 (Regimen):      {args.step1_parallelism}")
    print(f"  Step 2 (Extraction):   {args.step2_parallelism}")
    print(f"  Step 3 (Selection):    {args.step3_parallelism}")
    print(f"  Step 4 (Explicit):     {args.step4_parallelism}")
    print(f"  Step 5 (Consolidation):{args.step5_parallelism}")
    print()
    
    # Initialize storage clients
    drug_storage = LocalStorageClient(base_dir=args.drug_output_dir)
    storage = LocalStorageClient(base_dir=args.output_dir)
    
    # Load abstracts (reads drug data from storage)
    print("Loading abstracts...")
    inputs, original_rows, fieldnames = load_abstracts(args.input, drug_storage, args.limit)
    print(f"✓ Loaded {len(inputs)} abstracts with drug extraction data")
    print()
    
    # Process step by step
    steps = [
        ("step1_regimen", process_step1_single, args.step1_parallelism),
        ("step2_extraction", process_step2_single, args.step2_parallelism),
        ("step3_selection", process_step3_single, args.step3_parallelism),
        ("step4_explicit", process_step4_single, args.step4_parallelism),
        ("step5_consolidation", process_step5_single, args.step5_parallelism),
    ]
    
    total_success = 0
    total_failed = 0
    
    for step_name, process_func, parallelism in steps:
        print(f"→ {step_name.upper()}")
        success, failed = run_step_batch(step_name, process_func, inputs, storage, parallelism)
        total_success += success
        total_failed += failed
        print(f"  Done: {success} success, {failed} failed")
        print()
    
    # Summary
    print("=" * 60)
    print("📊 Summary:")
    
    # Count final status
    complete = 0
    incomplete = 0
    for inp in inputs:
        status = get_abstract_status(inp.abstract_id, storage)
        if status and status.pipeline_status == "success":
            complete += 1
        else:
            incomplete += 1
    
    print(f"   Total abstracts: {len(inputs)}")
    print(f"   Complete:        {complete}")
    print(f"   Incomplete:      {incomplete}")
    print(f"   Output:          {args.output_dir}")


if __name__ == "__main__":
    main()
