"""Drug Class Extraction Pipeline with Per-Step Checkpointing.

Orchestrates all 5 steps of drug class extraction with:
- Per-step checkpointing to GCS or local filesystem
- Resumable execution from last successful step
- Status tracking for monitoring and debugging
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

from langfuse import observe

from src.agents.drug_class.schemas import (
    DrugClassInput,
    DrugClassExtractionInput,
    DrugClassPipelineResult,
    PipelineStatus,
    Step1Output,
    Step2Output,
    Step3Output,
    Step4Output,
    Step5Output,
    DrugClassPipelineError,
    DrugClassExtractionError,
    StepName,
    RegimenInput,
    SelectionInput,
    ExplicitExtractionInput,
    ConsolidationInput,
)
from src.agents.drug_class.step1_regimen import identify_regimen
from src.agents.drug_class.step2_search import fetch_search_results
from src.agents.drug_class.step2_extraction import extract_with_tavily, extract_with_grounded
from src.agents.drug_class.step3_selection import select_drug_class, needs_llm_selection
from src.agents.drug_class.step4_explicit import extract_explicit_classes
from src.agents.drug_class.step5_consolidation import consolidate_drug_classes


# =============================================================================
# STORAGE PROTOCOL (for GCS or local filesystem)
# =============================================================================

class StorageClient(Protocol):
    """Protocol for storage operations (GCS or local filesystem)."""
    
    def download_json(self, path: str) -> dict:
        """Download JSON from storage."""
        ...
    
    def upload_json(self, path: str, data: dict) -> None:
        """Upload JSON to storage."""
        ...
    
    def exists(self, path: str) -> bool:
        """Check if path exists in storage."""
        ...


class LocalStorageClient:
    """Local filesystem storage client for development/testing."""
    
    def __init__(self, base_dir: str = "output"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, path: str) -> Path:
        return self.base_dir / path
    
    def download_json(self, path: str) -> dict:
        file_path = self._get_path(path)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def upload_json(self, path: str, data: dict) -> None:
        file_path = self._get_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def exists(self, path: str) -> bool:
        return self._get_path(path).exists()


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def _get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def _load_status(abstract_id: str, storage: StorageClient) -> Optional[PipelineStatus]:
    """Load existing pipeline status from storage.
    
    Args:
        abstract_id: Abstract ID
        storage: Storage client
        
    Returns:
        PipelineStatus if exists, None otherwise
    """
    status_path = f"{abstract_id}/status.json"
    
    try:
        if storage.exists(status_path):
            data = storage.download_json(status_path)
            return PipelineStatus.from_dict(data)
    except Exception as e:
        print(f"  ⚠ Could not load status: {e}")
    
    return None


def _save_status(status: PipelineStatus, storage: StorageClient) -> None:
    """Save pipeline status to storage.
    
    Args:
        status: Pipeline status
        storage: Storage client
    """
    status_path = f"{status.abstract_id}/status.json"
    status.last_updated = _get_timestamp()
    storage.upload_json(status_path, status.to_dict())


def _load_step_output(
    abstract_id: str,
    step_name: StepName,
    storage: StorageClient,
) -> Optional[dict]:
    """Load step output from storage.
    
    Args:
        abstract_id: Abstract ID
        step_name: Step name
        storage: Storage client
        
    Returns:
        Step output dict if exists, None otherwise
    """
    output_path = f"{abstract_id}/{step_name}.json"
    
    try:
        if storage.exists(output_path):
            return storage.download_json(output_path)
    except Exception as e:
        print(f"  ⚠ Could not load {step_name} output: {e}")
    
    return None


def _save_step_output(
    abstract_id: str,
    step_name: StepName,
    output: dict,
    storage: StorageClient,
) -> None:
    """Save step output to storage.
    
    Args:
        abstract_id: Abstract ID
        step_name: Step name
        output: Step output dict
        storage: Storage client
    """
    output_path = f"{abstract_id}/{step_name}.json"
    storage.upload_json(output_path, output)


def _update_step_status(
    status: PipelineStatus,
    step_name: StepName,
    step_status: str,
    llm_calls: int = 0,
    error: Optional[str] = None,
) -> None:
    """Update step status in pipeline status.
    
    Args:
        status: Pipeline status to update
        step_name: Step name
        step_status: New status ("pending", "running", "success", "failed")
        llm_calls: Number of LLM calls made
        error: Error message if failed
    """
    timestamp = _get_timestamp()
    
    if step_name not in status.steps:
        status.steps[step_name] = {}
    
    status.steps[step_name]["status"] = step_status
    status.steps[step_name]["llm_calls"] = llm_calls
    
    if step_status == "running":
        status.steps[step_name]["started_at"] = timestamp
    elif step_status == "success":
        status.steps[step_name]["completed_at"] = timestamp
        status.steps[step_name]["output_file"] = f"{step_name}.json"
        status.last_completed_step = step_name
    elif step_status == "failed":
        status.steps[step_name]["failed_at"] = timestamp
        status.steps[step_name]["error"] = error
        status.failed_step = step_name
        status.error = error
        
        # Increment attempts
        attempts = status.steps[step_name].get("attempts", 0)
        status.steps[step_name]["attempts"] = attempts + 1


def _is_step_complete(status: PipelineStatus, step_name: StepName) -> bool:
    """Check if a step is already complete.
    
    Args:
        status: Pipeline status
        step_name: Step name
        
    Returns:
        True if step is complete
    """
    step_info = status.steps.get(step_name, {})
    return step_info.get("status") == "success"


# =============================================================================
# MAIN PIPELINE
# =============================================================================

@observe(name="drug-class-pipeline")
def run_drug_class_pipeline(
    input_data: DrugClassInput,
    storage: Optional[StorageClient] = None,
    force_restart: bool = False,
) -> DrugClassPipelineResult:
    """Run the complete drug class extraction pipeline with checkpointing.
    
    Args:
        input_data: Pipeline input with abstract and extracted drugs
        storage: Storage client for checkpointing (defaults to local filesystem)
        force_restart: If True, ignore existing checkpoint and start fresh
        
    Returns:
        DrugClassPipelineResult with all outputs
        
    Raises:
        DrugClassPipelineError: If pipeline fails
    """
    abstract_id = input_data.abstract_id
    
    # Initialize storage client
    if storage is None:
        storage = LocalStorageClient()
    
    print(f"\n{'='*60}")
    print(f"Drug Class Extraction Pipeline: {abstract_id}")
    print(f"{'='*60}")
    
    # =========================================================================
    # LOAD OR CREATE STATUS
    # =========================================================================
    if not force_restart:
        status = _load_status(abstract_id, storage)
        if status and status.pipeline_status == "success":
            print(f"✓ Pipeline already complete for {abstract_id}, skipping...")
            return _build_final_result(status, abstract_id, storage)
    else:
        status = None
    
    if status is None:
        status = PipelineStatus(
            abstract_id=abstract_id,
            abstract_title=input_data.abstract_title,
            pipeline_status="running",
        )
        # Initialize all steps as pending
        for step in ["step1_regimen", "step2_extraction", "step3_selection",
                     "step4_explicit", "step5_consolidation"]:
            status.steps[step] = {"status": "pending"}
    
    status.pipeline_status = "running"
    _save_status(status, storage)
    
    # Combine all drugs
    all_drugs = (
        input_data.primary_drugs +
        input_data.secondary_drugs +
        input_data.comparator_drugs
    )
    
    if not all_drugs:
        print("⚠ No drugs provided, skipping pipeline")
        return DrugClassPipelineResult(
            abstract_id=abstract_id,
            abstract_title=input_data.abstract_title,
            drug_class_mappings=[],
            explicit_drug_classes=["NA"],
            success=True,
            total_llm_calls=0,
        )
    
    try:
        # =====================================================================
        # STEP 1: REGIMEN IDENTIFICATION (Per-Drug Checkpointing)
        # =====================================================================
        print(f"\n→ Step 1: Regimen Identification...")
        
        # Load existing checkpoint or create new
        step1_data = _load_step_output(abstract_id, "step1_regimen", storage)
        if step1_data:
            step1_output = Step1Output(**step1_data)
        else:
            step1_output = Step1Output()
            # Initialize all drugs as pending
            for drug in all_drugs:
                if drug and drug.strip():
                    step1_output.drug_status[drug.strip()] = "pending"
        
        # Check if Step 1 is already complete
        if step1_output.is_complete(all_drugs):
            print(f"  ✓ Step 1 already complete, all drugs processed")
        else:
            _update_step_status(status, "step1_regimen", "running")
            _save_status(status, storage)
            
            # Process only pending/failed drugs
            pending_drugs = step1_output.get_pending_drugs(all_drugs)
            step1_llm_calls = 0
            
            for drug in pending_drugs:
                if not drug or not drug.strip():
                    continue
                drug = drug.strip()
                
                print(f"    Processing: {drug}")
                try:
                    components = identify_regimen(RegimenInput(
                        abstract_id=abstract_id,
                        abstract_title=input_data.abstract_title,
                        drug=drug,
                    ))
                    step1_output.mark_success(drug, components)
                    step1_llm_calls += 1
                    print(f"      ✓ {drug} → {components}")
                    
                    # Save checkpoint after each successful drug
                    _save_step_output(abstract_id, "step1_regimen", step1_output.model_dump(), storage)
                    
                except Exception as e:
                    print(f"      ✗ {drug} failed: {e}")
                    step1_output.mark_failed(drug)
                    _save_step_output(abstract_id, "step1_regimen", step1_output.model_dump(), storage)
                    # Continue with other drugs instead of failing entire step
            
            status.total_llm_calls += step1_llm_calls
            
            # Check if all drugs succeeded
            if step1_output.is_complete(all_drugs):
                _update_step_status(status, "step1_regimen", "success", step1_llm_calls)
                print(f"  ✓ Step 1 complete ({step1_llm_calls} LLM calls)")
            else:
                # Some drugs failed - mark step as failed and stop pipeline
                failed_drugs = [d for d in all_drugs if step1_output.drug_status.get(d) == "failed"]
                _update_step_status(status, "step1_regimen", "failed", step1_llm_calls, 
                                    error=f"Failed drugs: {failed_drugs}")
                _save_status(status, storage)
                print(f"  ✗ Step 1 failed: {len(failed_drugs)} drugs failed")
                raise DrugClassPipelineError(
                    f"Step 1 regimen identification failed for {len(failed_drugs)} drugs: {failed_drugs}",
                    step="step1_regimen"
                )
            
            _save_status(status, storage)
        
        # =====================================================================
        # STEP 2: DRUG CLASS EXTRACTION (Per-Drug Checkpointing)
        # =====================================================================
        print(f"\n→ Step 2: Drug Class Extraction...")
        
        # Load existing checkpoint or create new
        step2_data = _load_step_output(abstract_id, "step2_extraction", storage)
        if step2_data:
            step2_output = Step2Output(**step2_data)
        else:
            step2_output = Step2Output()
            # Initialize all component drugs as pending
            for drug in step1_output.get_all_components():
                if drug and drug.strip():
                    step2_output.drug_status[drug.strip()] = "pending"
        
        # Check if Step 2 is already complete
        component_drugs = step1_output.get_all_components()
        firms = input_data.firms  # Firms for firm search
        
        if step2_output.is_complete(component_drugs):
            print(f"  ✓ Step 2 already complete, all drugs extracted")
        else:
            _update_step_status(status, "step2_extraction", "running")
            _save_status(status, storage)
            
            # Process only pending/failed drugs
            pending_drugs = step2_output.get_pending_drugs(component_drugs)
            step2_llm_calls = 0
            
            for drug in pending_drugs:
                if not drug or not drug.strip():
                    continue
                drug = drug.strip()
                
                print(f"    Processing: {drug}")
                try:
                    # Fetch search results (with caching)
                    drug_class_results, firm_results = fetch_search_results(
                        drug, firms, storage
                    )
                    
                    # Try Tavily extraction first
                    extraction_input = DrugClassExtractionInput(
                        abstract_id=abstract_id,
                        abstract_title=input_data.abstract_title,
                        drug=drug,
                        firms=firms,
                        drug_class_results=drug_class_results,
                        firm_search_results=firm_results,
                    )
                    
                    try:
                        result = extract_with_tavily(extraction_input)
                        step2_llm_calls += 1
                        
                        # Check if we got meaningful results
                        if result.success:
                            step2_output.mark_success(drug, result)
                            print(f"      ✓ Tavily: {result.drug_classes}")
                        else:
                            # Fallback to grounded search
                            raise DrugClassExtractionError("Tavily returned NA")
                            
                    except DrugClassExtractionError:
                        # Fallback to grounded search
                        print(f"      → Falling back to grounded search...")
                        result = extract_with_grounded(extraction_input)
                        step2_llm_calls += 1
                        step2_output.mark_success(drug, result)
                        print(f"      ✓ Grounded: {result.drug_classes}")
                    
                    # Save checkpoint after each successful drug
                    _save_step_output(abstract_id, "step2_extraction", step2_output.model_dump(), storage)
                    
                except Exception as e:
                    print(f"      ✗ {drug} failed: {e}")
                    step2_output.mark_failed(drug, str(e))
                    _save_step_output(abstract_id, "step2_extraction", step2_output.model_dump(), storage)
                    # Continue with other drugs instead of failing entire step
            
            status.total_llm_calls += step2_llm_calls
            
            # Check if all drugs succeeded
            if step2_output.is_complete(component_drugs):
                _update_step_status(status, "step2_extraction", "success", step2_llm_calls)
                print(f"  ✓ Step 2 complete ({step2_llm_calls} LLM calls)")
            else:
                # Some drugs failed - mark step as failed and stop pipeline
                failed_drugs = [d for d in component_drugs if step2_output.drug_status.get(d) == "failed"]
                _update_step_status(status, "step2_extraction", "failed", step2_llm_calls, 
                                    error=f"Failed drugs: {failed_drugs}")
                _save_status(status, storage)
                print(f"  ✗ Step 2 failed: {len(failed_drugs)} drugs failed")
                raise DrugClassPipelineError(
                    f"Step 2 extraction failed for {len(failed_drugs)} drugs: {failed_drugs}",
                    step="step2_extraction"
                )
            
            _save_status(status, storage)
        
        # =====================================================================
        # STEP 3: DRUG CLASS SELECTION (Per-Drug Checkpointing)
        # =====================================================================
        print(f"\n→ Step 3: Drug Class Selection...")
        
        # Load existing checkpoint or create new
        step3_data = _load_step_output(abstract_id, "step3_selection", storage)
        if step3_data:
            step3_output = Step3Output(**step3_data)
        else:
            step3_output = Step3Output()
        
        # Get all drugs to process from Step 2 extractions
        extractions_list = step2_output.get_results_list()
        drugs_to_select = [e.drug_name for e in extractions_list if e.drug_name]
        
        # Initialize drug statuses for new drugs
        for drug in drugs_to_select:
            if drug not in step3_output.drug_status:
                step3_output.drug_status[drug] = "pending"
        
        # Check if Step 3 is already complete
        if step3_output.is_complete(drugs_to_select):
            print(f"  ✓ Step 3 already complete, all drugs processed")
        else:
            _update_step_status(status, "step3_selection", "running")
            _save_status(status, storage)
            
            # Build extraction lookup by drug name
            extraction_lookup = {e.drug_name: e for e in extractions_list}
            
            # Process only pending/failed drugs
            pending_drugs = step3_output.get_pending_drugs(drugs_to_select)
            step3_llm_calls = 0
            
            for drug in pending_drugs:
                if not drug:
                    continue
                
                extraction = extraction_lookup.get(drug)
                if not extraction:
                    continue
                
                extraction_details = [d.model_dump() for d in extraction.extraction_details]
                
                # If no extraction details, use drug_classes directly (no LLM call needed)
                if not extraction_details:
                    from src.agents.drug_class.schemas import DrugSelectionResult
                    result = DrugSelectionResult(
                        drug_name=drug,
                        selected_drug_classes=extraction.drug_classes if extraction.drug_classes else ["NA"],
                        reasoning="No extraction details available for selection.",
                    )
                    step3_output.mark_success(drug, result)
                    _save_step_output(abstract_id, "step3_selection", step3_output.model_dump(), storage)
                    print(f"    ⏭ {drug}: using extracted classes directly (no details)")
                    continue
                
                print(f"    Processing: {drug}")
                
                try:
                    result = select_drug_class(SelectionInput(
                        abstract_id=abstract_id,
                        drug_name=drug,
                        extraction_details=extraction_details,
                    ))
                    
                    # Count LLM call only if selection was actually performed
                    if needs_llm_selection(extraction_details):
                        step3_llm_calls += 1
                    
                    step3_output.mark_success(drug, result)
                    print(f"      ✓ Selected: {result.selected_drug_classes}")
                    
                    # Checkpoint after each successful drug
                    _save_step_output(abstract_id, "step3_selection", step3_output.model_dump(), storage)
                    
                except Exception as e:
                    print(f"      ✗ {drug} failed: {e}")
                    step3_output.mark_failed(drug, str(e))
                    _save_step_output(abstract_id, "step3_selection", step3_output.model_dump(), storage)
                    # Continue with other drugs instead of failing entire step
            
            status.total_llm_calls += step3_llm_calls
            
            # Check if all drugs succeeded
            if step3_output.is_complete(drugs_to_select):
                _update_step_status(status, "step3_selection", "success", step3_llm_calls)
                print(f"  ✓ Step 3 complete ({step3_llm_calls} LLM calls)")
            else:
                # Some drugs failed - mark step as failed and stop pipeline
                failed_drugs = [d for d in drugs_to_select if step3_output.drug_status.get(d) == "failed"]
                _update_step_status(status, "step3_selection", "failed", step3_llm_calls, 
                                    error=f"Failed drugs: {failed_drugs}")
                _save_status(status, storage)
                print(f"  ✗ Step 3 failed: {len(failed_drugs)} drugs failed")
                raise DrugClassPipelineError(
                    f"Step 3 selection failed for {len(failed_drugs)} drugs: {failed_drugs}",
                    step="step3_selection"
                )
            
            _save_status(status, storage)
        
        # =====================================================================
        # STEP 4: EXPLICIT DRUG CLASS EXTRACTION
        # =====================================================================
        if _is_step_complete(status, "step4_explicit"):
            print(f"\n✓ Step 4 already complete, loading from checkpoint...")
            step4_data = _load_step_output(abstract_id, "step4_explicit", storage)
            step4_output = Step4Output(**step4_data)
        else:
            print(f"\n→ Step 4: Explicit Drug Class Extraction...")
            _update_step_status(status, "step4_explicit", "running")
            _save_status(status, storage)
            
            step4_output = extract_explicit_classes(ExplicitExtractionInput(
                abstract_id=abstract_id,
                abstract_title=input_data.abstract_title,
            ))
            
            # Always 1 LLM call for explicit extraction (unless empty title)
            llm_calls = 1 if input_data.abstract_title and input_data.abstract_title.strip() else 0
            status.total_llm_calls += llm_calls
            _update_step_status(status, "step4_explicit", "success", llm_calls)
            _save_step_output(abstract_id, "step4_explicit", step4_output.model_dump(), storage)
            _save_status(status, storage)
            print(f"  ✓ Step 4 complete ({llm_calls} LLM calls)")
        
        # =====================================================================
        # STEP 5: CONSOLIDATION
        # =====================================================================
        if _is_step_complete(status, "step5_consolidation"):
            print(f"\n✓ Step 5 already complete, loading from checkpoint...")
            step5_data = _load_step_output(abstract_id, "step5_consolidation", storage)
            step5_output = Step5Output(**step5_data)
        else:
            print(f"\n→ Step 5: Consolidation...")
            _update_step_status(status, "step5_consolidation", "running")
            _save_status(status, storage)
            
            # Convert selections to dicts for step 5
            drug_selections = [
                {
                    "drug_name": s.drug_name,
                    "selected_classes": s.selected_drug_classes,
                }
                for s in step3_output.get_results_list()
            ]
            
            step5_output = consolidate_drug_classes(ConsolidationInput(
                abstract_id=abstract_id,
                abstract_title=input_data.abstract_title,
                explicit_drug_classes=step4_output.explicit_drug_classes,
                drug_selections=drug_selections,
            ))
            
            # LLM call is made unless explicit classes are NA or empty
            has_explicit = step4_output.explicit_drug_classes and step4_output.explicit_drug_classes != ["NA"]
            has_selections = bool(drug_selections)
            llm_calls = 1 if (has_explicit and has_selections) else 0
            status.total_llm_calls += llm_calls
            _update_step_status(status, "step5_consolidation", "success", llm_calls)
            _save_step_output(abstract_id, "step5_consolidation", step5_output.model_dump(), storage)
            _save_status(status, storage)
            print(f"  ✓ Step 5 complete ({llm_calls} LLM calls)")
        
        # =====================================================================
        # BUILD FINAL RESULT
        # =====================================================================
        status.pipeline_status = "success"
        _save_status(status, storage)
        
        final_result = _build_final_result_from_steps(
            abstract_id=abstract_id,
            abstract_title=input_data.abstract_title,
            step1_output=step1_output,
            step3_output=step3_output,
            step5_output=step5_output,
            total_llm_calls=status.total_llm_calls,
        )
        
        # Save final result
        _save_step_output(abstract_id, "final_result", final_result.to_dict(), storage)
        
        print(f"\n{'='*60}")
        print(f"✓ Pipeline complete for {abstract_id}")
        print(f"  Total LLM calls: {status.total_llm_calls}")
        print(f"{'='*60}\n")
        
        return final_result
        
    except Exception as e:
        # Mark pipeline as failed
        status.pipeline_status = "failed"
        status.error = str(e)
        _save_status(status, storage)
        
        raise DrugClassPipelineError(
            f"Pipeline failed at {status.failed_step or 'unknown step'}: {e}",
            step=status.failed_step,
        ) from e


def _build_final_result_from_steps(
    abstract_id: str,
    abstract_title: str,
    step1_output: Step1Output,
    step3_output: Step3Output,
    step5_output: Step5Output,
    total_llm_calls: int,
) -> DrugClassPipelineResult:
    """Build final result from step outputs.
    
    Args:
        abstract_id: Abstract ID
        abstract_title: Abstract title
        step1_output: Regimen identification output
        step3_output: Selection output
        step5_output: Consolidation output
        total_llm_calls: Total LLM calls made
        
    Returns:
        DrugClassPipelineResult
    """
    # Build drug class mappings from step1 (drug_to_components) and step3 (selections)
    drug_class_mappings = []
    
    # Create lookup from selections
    selection_lookup = {
        s.drug_name: s.selected_drug_classes
        for s in step3_output.get_results_list()
    }
    
    # Iterate over drug_to_components mapping
    for drug, components in step1_output.drug_to_components.items():
        component_data = []
        for comp in components:
            component_data.append({
                "component": comp,
                "selected_classes": selection_lookup.get(comp, ["NA"]),
            })
        
        drug_class_mappings.append({
            "drug": drug,
            "components": component_data,
        })
    
    return DrugClassPipelineResult(
        abstract_id=abstract_id,
        abstract_title=abstract_title,
        drug_class_mappings=drug_class_mappings,
        explicit_drug_classes=step5_output.refined_explicit_classes,
        success=True,
        total_llm_calls=total_llm_calls,
        checkpoint_files={
            "step1_regimen": f"{abstract_id}/step1_regimen.json",
            "step2_extraction": f"{abstract_id}/step2_extraction.json",
            "step3_selection": f"{abstract_id}/step3_selection.json",
            "step4_explicit": f"{abstract_id}/step4_explicit.json",
            "step5_consolidation": f"{abstract_id}/step5_consolidation.json",
            "final_result": f"{abstract_id}/final_result.json",
            "status": f"{abstract_id}/status.json",
        }
    )


def _build_final_result(
    status: PipelineStatus,
    abstract_id: str,
    storage: StorageClient,
) -> DrugClassPipelineResult:
    """Build final result from existing checkpoint.
    
    Args:
        status: Pipeline status
        abstract_id: Abstract ID
        storage: Storage client
        
    Returns:
        DrugClassPipelineResult loaded from checkpoint
    """
    # Try to load existing final result
    try:
        final_data = _load_step_output(abstract_id, "final_result", storage)
        if final_data:
            return DrugClassPipelineResult(
                abstract_id=final_data["abstract_id"],
                abstract_title=final_data["abstract_title"],
                drug_class_mappings=final_data.get("drug_class_mappings", []),
                explicit_drug_classes=final_data.get("explicit_drug_classes", ["NA"]),
                success=final_data.get("success", True),
                error=final_data.get("error"),
                total_llm_calls=final_data.get("total_llm_calls", 0),
                checkpoint_files=final_data.get("checkpoint_files", {}),
            )
    except Exception:
        pass
    
    # Fallback: rebuild from step outputs
    step1_data = _load_step_output(abstract_id, "step1_regimen", storage)
    step3_data = _load_step_output(abstract_id, "step3_selection", storage)
    step5_data = _load_step_output(abstract_id, "step5_consolidation", storage)
    
    return _build_final_result_from_steps(
        abstract_id=abstract_id,
        abstract_title=status.abstract_title,
        step1_output=Step1Output(**step1_data) if step1_data else Step1Output(),
        step3_output=Step3Output(**step3_data) if step3_data else Step3Output(),
        step5_output=Step5Output(**step5_data) if step5_data else Step5Output(),
        total_llm_calls=status.total_llm_calls,
    )

