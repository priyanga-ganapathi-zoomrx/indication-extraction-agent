"""Abstract Extraction Workflow - Temporal orchestration for medical abstract processing.

Single flat workflow that processes one abstract through up to three pipelines:
1. Drug Pipeline (extraction + validation)
2. Drug Class Pipeline (5-step pipeline + validation)
3. Indication Pipeline (extraction + validation)

Pipeline selection via `pipelines` field on AbstractExtractionInput.
Per-step checkpointing via _run_with_checkpoint.
"""

from datetime import timedelta
from typing import Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.temporal.activities.checkpoint import (
        load_workflow_status,
        save_workflow_status,
        load_step_output,
        save_step_output,
    )
    from src.temporal.config import (
        TaskQueues,
        Timeouts,
        RetryPolicies,
    )
    from src.temporal.schemas.status import (
        WorkflowStatus,
        StepStatus,
    )
    from src.temporal.schemas.workflow import (
        AbstractExtractionInput,
        AbstractExtractionOutput,
        StepResult,
    )
    # Drug activities + schemas
    from src.agents.drug.schemas import (
        DrugInput,
        ValidationInput as DrugValidationInput,
    )
    from src.temporal.activities.drug import extract_drugs, validate_drugs
    # Drug class activities + schemas
    from src.agents.drug_class.schemas import (
        RegimenInput,
        DrugClassExtractionInput,
        SelectionInput,
        ExplicitExtractionInput,
        ConsolidationInput,
        ValidationInput as DrugClassValidationInput,
    )
    from src.temporal.activities.drug_class import (
        step1_regimen,
        step2_fetch_search_results,
        step2_extract_with_tavily,
        step2_extract_with_grounded,
        step3_selection,
        step4_explicit,
        step5_consolidation,
        validate_drug_class_activity,
    )
    # Indication activities + schemas
    from src.agents.indication.schemas import IndicationInput
    from src.temporal.activities.indication import (
        extract_indication,
        validate_indication,
    )


# =============================================================================
# WORKFLOW DEFINITION
# =============================================================================

@workflow.defn(name="AbstractExtractionWorkflow")
class AbstractExtractionWorkflow:
    """Orchestrates complete extraction pipeline for a medical conference abstract.

    Processes a single abstract through up to three sequential pipelines:
    1. Drug extraction and validation
    2. Drug class extraction (depends on drug results) - 5 steps + validation
    3. Indication extraction and validation

    Pipeline selection is controlled by the `pipelines` input field.
    """

    def __init__(self) -> None:
        self._output: Optional[AbstractExtractionOutput] = None
        self._status: Optional[WorkflowStatus] = None
        self._current_step: str = "initialized"

    @workflow.run
    async def run(self, input: AbstractExtractionInput) -> AbstractExtractionOutput:
        """Execute the extraction pipeline for the requested pipelines."""
        self._output = AbstractExtractionOutput(abstract_id=input.abstract_id)

        workflow.logger.info(
            f"Starting extraction for abstract {input.abstract_id} "
            f"(pipelines: {input.pipelines})"
        )

        self._current_step = "loading_status"
        await self._load_status(input)

        try:
            # --- Drug Pipeline ---
            if "drug" in input.pipelines:
                if self._status.should_run_drug_pipeline():
                    self._current_step = "drug_pipeline"
                    await self._run_drug_pipeline(input)
                else:
                    workflow.logger.info(
                        f"Drug pipeline already complete for {input.abstract_id}, "
                        "loading from checkpoint"
                    )
                    self._output.drug.extraction = (
                        await self._load_checkpoint(input, "drug_extraction") or {}
                    )
                    self._output.drug.validation = (
                        await self._load_checkpoint(input, "drug_validation")
                    )

            # --- Drug Class Pipeline (depends on drug extraction) ---
            if "drug_class" in input.pipelines:
                if not self._output.drug.extraction:
                    self._output.drug.extraction = (
                        await self._load_checkpoint(input, "drug_extraction") or {}
                    )
                primary_drugs = self._output.drug.extraction.get("primary_drugs", [])

                if primary_drugs and self._status.should_run_drug_class_pipeline():
                    self._current_step = "drug_class_pipeline"
                    await self._run_drug_class_pipeline(input, primary_drugs)
                elif not primary_drugs:
                    workflow.logger.info(
                        f"No primary drugs for {input.abstract_id}, "
                        "skipping drug class pipeline"
                    )
                else:
                    workflow.logger.info(
                        f"Drug class pipeline already complete for {input.abstract_id}"
                    )

            # --- Indication Pipeline ---
            if "indication" in input.pipelines:
                if self._status.should_run_indication_pipeline():
                    self._current_step = "indication_pipeline"
                    await self._run_indication_pipeline(input)
                else:
                    workflow.logger.info(
                        f"Indication pipeline already complete for {input.abstract_id}, "
                        "loading from checkpoint"
                    )
                    self._output.indication.extraction = (
                        await self._load_checkpoint(input, "indication_extraction") or {}
                    )
                    self._output.indication.validation = (
                        await self._load_checkpoint(input, "indication_validation")
                    )

            self._output.completed = True
            if self._output.errors:
                self._status.mark_partial_success()
                workflow.logger.warning(
                    f"Partial success for abstract {input.abstract_id}: "
                    f"{len(self._output.errors)} error(s)"
                )
            else:
                self._status.mark_success()
                workflow.logger.info(
                    f"Completed extraction for abstract {input.abstract_id}"
                )

        except Exception as e:
            workflow.logger.error(f"Workflow error for {input.abstract_id}: {e}")
            self._output.errors.append(str(e))
            self._status.mark_failed(str(e))

        self._current_step = "saving_status"
        await self._save_status(input)
        return self._output

    # =========================================================================
    # QUERIES
    # =========================================================================

    @workflow.query
    def current_step(self) -> str:
        return self._current_step

    @workflow.query
    def get_output(self) -> Optional[AbstractExtractionOutput]:
        return self._output

    @workflow.query
    def get_status(self) -> Optional[dict]:
        return self._status.to_dict() if self._status else None

    # =========================================================================
    # STATUS MANAGEMENT
    # =========================================================================

    async def _load_status(self, input: AbstractExtractionInput) -> None:
        """Load existing status from checkpoint or create new."""
        if input.storage_path:
            status_dict = await workflow.execute_activity(
                load_workflow_status,
                args=[input.storage_path, input.abstract_id],
                task_queue=TaskQueues.CHECKPOINT,
                start_to_close_timeout=Timeouts.STORAGE,
                retry_policy=RetryPolicies.STORAGE,
            )
            if status_dict:
                self._status = WorkflowStatus.from_dict(status_dict)
                # Clear stale errors from previous runs so they don't
                # carry forward into a retry
                self._status.errors = []
                self._status.status = "running"
                workflow.logger.info(
                    f"Loaded existing status for {input.abstract_id}, "
                    "cleared previous errors for fresh run"
                )
                return

        self._status = WorkflowStatus(
            abstract_id=input.abstract_id,
            abstract_title=input.abstract_title,
        )
        workflow.logger.info(f"Created new status for {input.abstract_id}")

    async def _save_status(self, input: AbstractExtractionInput) -> None:
        """Save current status to checkpoint."""
        if not input.storage_path:
            return
        self._status.update_timestamp()
        await workflow.execute_activity(
            save_workflow_status,
            args=[input.storage_path, input.abstract_id, self._status.to_dict()],
            task_queue=TaskQueues.CHECKPOINT,
            start_to_close_timeout=Timeouts.STORAGE,
            retry_policy=RetryPolicies.STORAGE,
        )
        workflow.logger.info(f"Saved status for {input.abstract_id}")

    # =========================================================================
    # CHECKPOINT HELPERS
    # =========================================================================

    async def _load_checkpoint(
        self, input: AbstractExtractionInput, step_name: str
    ) -> Optional[dict]:
        """Load a step checkpoint from storage."""
        if not input.storage_path:
            return None
        return await workflow.execute_activity(
            load_step_output,
            args=[input.storage_path, input.abstract_id, step_name],
            task_queue=TaskQueues.CHECKPOINT,
            start_to_close_timeout=Timeouts.STORAGE,
            retry_policy=RetryPolicies.STORAGE,
        )

    async def _save_checkpoint(
        self, input: AbstractExtractionInput, step_name: str, data: dict
    ) -> None:
        """Save a step checkpoint to storage."""
        if not input.storage_path:
            return
        await workflow.execute_activity(
            save_step_output,
            args=[input.storage_path, input.abstract_id, step_name, data],
            task_queue=TaskQueues.CHECKPOINT,
            start_to_close_timeout=Timeouts.STORAGE,
            retry_policy=RetryPolicies.STORAGE,
        )
        workflow.logger.info(f"Saved {step_name} checkpoint for {input.abstract_id}")

    async def _run_with_checkpoint(
        self,
        input: AbstractExtractionInput,
        step_name: str,
        activity_fn,
        task_queue: str,
        timeout: timedelta,
        retry_policy: RetryPolicy,
        activity_input=None,
        activity_args: list = None,
    ) -> StepResult:
        """Run an activity with checkpoint support.

        1. Load existing checkpoint -> return cached result if found
        2. Execute activity (single input or multiple args)
        3. Save result as checkpoint
        """
        existing = await self._load_checkpoint(input, step_name)
        if existing is not None:
            workflow.logger.info(
                f"Loaded {step_name} from checkpoint for {input.abstract_id}"
            )
            return StepResult(
                status="success", output=existing, from_checkpoint=True
            )

        try:
            if activity_args is not None:
                result = await workflow.execute_activity(
                    activity_fn,
                    args=activity_args,
                    task_queue=task_queue,
                    start_to_close_timeout=timeout,
                    retry_policy=retry_policy,
                )
            else:
                result = await workflow.execute_activity(
                    activity_fn,
                    activity_input,
                    task_queue=task_queue,
                    start_to_close_timeout=timeout,
                    retry_policy=retry_policy,
                )
        except Exception as e:
            workflow.logger.error(f"Activity {step_name} failed: {e}")
            return StepResult(status="failed", error=str(e))

        await self._save_checkpoint(input, step_name, result)
        return StepResult(status="success", output=result, from_checkpoint=False)

    # =========================================================================
    # DRUG PIPELINE
    # =========================================================================

    async def _run_drug_pipeline(self, input: AbstractExtractionInput) -> None:
        """Run drug extraction + validation with per-step checkpointing."""
        workflow.logger.info(f"Running drug pipeline for abstract {input.abstract_id}")

        # Extraction
        extraction = await self._run_with_checkpoint(
            input, "drug_extraction", extract_drugs,
            TaskQueues.DRUG, Timeouts.FAST_LLM, RetryPolicies.FAST_LLM,
            activity_input=DrugInput(
                abstract_id=input.abstract_id,
                abstract_title=input.abstract_title,
            ),
        )
        self._status.drug.extraction = extraction.to_step_status()
        if extraction.status != "success":
            raise RuntimeError(f"Drug extraction failed: {extraction.error}")
        self._output.drug.extraction = extraction.output

        # Validation
        validation = await self._run_with_checkpoint(
            input, "drug_validation", validate_drugs,
            TaskQueues.DRUG, Timeouts.FAST_LLM, RetryPolicies.FAST_LLM,
            activity_input=DrugValidationInput(
                abstract_id=input.abstract_id,
                abstract_title=input.abstract_title,
                extraction_result=extraction.output,
            ),
        )
        self._status.drug.validation = validation.to_step_status()
        if validation.status != "success":
            self._output.errors.append(validation.error or "Drug validation failed")
        else:
            self._output.drug.validation = validation.output

        await self._save_status(input)
        workflow.logger.info(f"Drug pipeline completed for {input.abstract_id}")

    # =========================================================================
    # INDICATION PIPELINE
    # =========================================================================

    async def _run_indication_pipeline(self, input: AbstractExtractionInput) -> None:
        """Run indication extraction + validation with per-step checkpointing."""
        workflow.logger.info(
            f"Running indication pipeline for abstract {input.abstract_id}"
        )

        indication_input = IndicationInput(
            abstract_id=input.abstract_id,
            abstract_title=input.abstract_title,
            session_title=input.session_title,
        )

        # Extraction (fast LLM)
        extraction = await self._run_with_checkpoint(
            input, "indication_extraction", extract_indication,
            TaskQueues.INDICATION_EXTRACTION,
            Timeouts.FAST_LLM, RetryPolicies.FAST_LLM,
            activity_input=indication_input,
        )
        self._status.indication.extraction = extraction.to_step_status()
        if extraction.status != "success":
            raise RuntimeError(f"Indication extraction failed: {extraction.error}")
        self._output.indication.extraction = extraction.output

        # Validation (slow LLM - Sonnet 4.5, multi-arg activity)
        validation = await self._run_with_checkpoint(
            input, "indication_validation", validate_indication,
            TaskQueues.INDICATION_VALIDATION,
            Timeouts.SLOW_LLM, RetryPolicies.SLOW_LLM,
            activity_args=[indication_input, extraction.output],
        )
        self._status.indication.validation = validation.to_step_status()
        if validation.status != "success":
            self._output.errors.append(
                validation.error or "Indication validation failed"
            )
        else:
            self._output.indication.validation = validation.output

        await self._save_status(input)
        workflow.logger.info(
            f"Indication pipeline completed for {input.abstract_id}"
        )

    # =========================================================================
    # DRUG CLASS PIPELINE
    # =========================================================================

    async def _run_drug_class_pipeline(
        self, input: AbstractExtractionInput, primary_drugs: list[str]
    ) -> None:
        """Run the full drug class pipeline (steps 1-5 + validation)."""
        workflow.logger.info(
            f"Running drug class pipeline for abstract {input.abstract_id} "
            f"with {len(primary_drugs)} drugs"
        )

        # ---- Steps 1-3 (combined checkpoint) ----
        steps1_3_data = await self._load_checkpoint(input, "drug_class_steps1_3")
        if steps1_3_data is None:
            steps1_3_data = await self._run_drug_class_steps1_3(input, primary_drugs)

        # Check if any drug had errors during steps 1-3
        drug_errors = [
            d["error"] for d in steps1_3_data.get("drug_results", [])
            if d.get("error")
        ]
        if drug_errors:
            # Do NOT save checkpoint when there are errors - allows retry
            error_msg = f"Drug class steps 1-3 errors: {drug_errors}"
            self._status.drug_class.step2_extraction = StepStatus.failed(error_msg)
            self._output.errors.append(error_msg)
            workflow.logger.error(error_msg)
            await self._save_status(input)
            return

        # Only checkpoint on success so retries re-execute
        await self._save_checkpoint(input, "drug_class_steps1_3", steps1_3_data)

        self._output.drug_class.drug_results = steps1_3_data.get("drug_results", [])
        all_drug_selections = steps1_3_data.get("drug_selections", [])
        all_search_results = steps1_3_data.get("search_results", {})
        all_extraction_results = steps1_3_data.get("extraction_results", {})
        self._status.drug_class.step1_regimen = StepStatus.success()
        self._status.drug_class.step2_extraction = StepStatus.success()
        self._status.drug_class.step3_selection = StepStatus.success()

        # ---- Step 4: Explicit extraction from title ----
        step4 = await self._run_with_checkpoint(
            input, "drug_class_step4", step4_explicit,
            TaskQueues.DRUG_CLASS, Timeouts.FAST_LLM, RetryPolicies.FAST_LLM,
            activity_input=ExplicitExtractionInput(
                abstract_id=input.abstract_id,
                abstract_title=input.abstract_title,
            ),
        )
        self._status.drug_class.step4_explicit = step4.to_step_status()
        if step4.status != "success":
            self._output.errors.append(step4.error or "Drug class step 4 failed")
            await self._save_status(input)
            return
        self._output.drug_class.explicit_classes = step4.output.get(
            "explicit_drug_classes", []
        )

        # ---- Step 5: Consolidation ----
        explicit = self._output.drug_class.explicit_classes
        if explicit and explicit != ["NA"]:
            step5 = await self._run_with_checkpoint(
                input, "drug_class_step5", step5_consolidation,
                TaskQueues.DRUG_CLASS, Timeouts.FAST_LLM, RetryPolicies.FAST_LLM,
                activity_input=ConsolidationInput(
                    abstract_id=input.abstract_id,
                    abstract_title=input.abstract_title,
                    explicit_drug_classes=explicit,
                    drug_selections=all_drug_selections,
                ),
            )
            self._status.drug_class.step5_consolidation = step5.to_step_status()
            if step5.status == "success":
                self._output.drug_class.refined_explicit_classes = (
                    step5.output.get("refined_explicit_classes", explicit)
                )
            else:
                self._output.drug_class.refined_explicit_classes = explicit
                self._output.errors.append(step5.error or "Drug class step 5 failed")
                await self._save_status(input)
                return
        else:
            self._output.drug_class.refined_explicit_classes = explicit
            self._status.drug_class.step5_consolidation = StepStatus.success()

        # ---- Step 6: Validation (per component) ----
        validation_data = await self._run_drug_class_validation(
            input, all_extraction_results, all_search_results
        )
        self._output.drug_class.validation_results = validation_data.get("results", [])
        if validation_data.get("errors"):
            self._output.errors.extend(validation_data["errors"])
            self._status.drug_class.validation = StepStatus.failed(
                "; ".join(validation_data["errors"])
            )
        else:
            self._status.drug_class.validation = StepStatus.success()

        await self._save_status(input)
        workflow.logger.info(
            f"Drug class pipeline completed for {input.abstract_id}"
        )

    async def _run_drug_class_steps1_3(
        self, input: AbstractExtractionInput, primary_drugs: list[str],
    ) -> dict:
        """Run steps 1-3 for all primary drugs (per-drug loops).

        Returns dict with drug_results, drug_selections, search_results,
        extraction_results.
        """
        drug_results = []
        all_drug_selections = []
        all_search_results = {}
        all_extraction_results = {}

        for drug in primary_drugs:
            drug_data = {
                "drug": drug, "components": [],
                "extractions": {}, "selections": {},
            }
            try:
                # Step 1: Regimen identification
                workflow.logger.info(
                    f"Step 1 - Regimen for drug '{drug}' in {input.abstract_id}"
                )
                components = await workflow.execute_activity(
                    step1_regimen,
                    RegimenInput(
                        abstract_id=input.abstract_id,
                        abstract_title=input.abstract_title,
                        drug=drug,
                    ),
                    task_queue=TaskQueues.DRUG_CLASS,
                    start_to_close_timeout=Timeouts.FAST_LLM,
                    retry_policy=RetryPolicies.FAST_LLM,
                )
                drug_data["components"] = components

                # Steps 2-3: For each component
                for component in components:
                    # Step 2a: Fetch search results
                    search_result = await workflow.execute_activity(
                        step2_fetch_search_results,
                        args=[component, input.firms, input.storage_path],
                        task_queue=TaskQueues.DRUG_CLASS,
                        start_to_close_timeout=Timeouts.SEARCH,
                        retry_policy=RetryPolicies.SEARCH,
                    )
                    all_search_results[component] = search_result.get(
                        "drug_class_results", []
                    )

                    # Step 2b: Extract with Tavily
                    ext_input = DrugClassExtractionInput(
                        abstract_id=input.abstract_id,
                        abstract_title=input.abstract_title,
                        drug=component,
                        full_abstract=input.full_abstract,
                        firms=input.firms,
                        drug_class_results=search_result.get("drug_class_results", []),
                        firm_search_results=search_result.get("firm_search_results", []),
                    )
                    extraction_result = await workflow.execute_activity(
                        step2_extract_with_tavily,
                        ext_input,
                        task_queue=TaskQueues.DRUG_CLASS,
                        start_to_close_timeout=Timeouts.FAST_LLM,
                        retry_policy=RetryPolicies.FAST_LLM,
                    )

                    # Fallback to grounded search if Tavily returns NA
                    drug_classes = extraction_result.get("drug_classes", [])
                    if not drug_classes or drug_classes == ["NA"]:
                        workflow.logger.info(
                            f"Tavily returned NA for {component}, trying grounded search"
                        )
                        extraction_result = await workflow.execute_activity(
                            step2_extract_with_grounded,
                            ext_input,
                            task_queue=TaskQueues.DRUG_CLASS,
                            start_to_close_timeout=Timeouts.FAST_LLM,
                            retry_policy=RetryPolicies.FAST_LLM,
                        )

                    drug_data["extractions"][component] = extraction_result
                    all_extraction_results[component] = extraction_result

                    # Step 3: Selection (if extraction has details)
                    extraction_details = extraction_result.get("extraction_details", [])
                    if extraction_details:
                        selection_result = await workflow.execute_activity(
                            step3_selection,
                            SelectionInput(
                                abstract_id=input.abstract_id,
                                drug_name=component,
                                extraction_details=extraction_details,
                            ),
                            task_queue=TaskQueues.DRUG_CLASS,
                            start_to_close_timeout=Timeouts.FAST_LLM,
                            retry_policy=RetryPolicies.FAST_LLM,
                        )
                        drug_data["selections"][component] = selection_result
                        all_drug_selections.append({
                            "drug_name": component,
                            "selected_classes": selection_result.get(
                                "selected_drug_classes", []
                            ),
                        })

            except Exception as e:
                workflow.logger.error(
                    f"Drug class steps 1-3 error for drug '{drug}': {e}"
                )
                drug_data["error"] = str(e)

            drug_results.append(drug_data)

        return {
            "drug_results": drug_results,
            "drug_selections": all_drug_selections,
            "search_results": all_search_results,
            "extraction_results": all_extraction_results,
        }

    async def _run_drug_class_validation(
        self,
        input: AbstractExtractionInput,
        extraction_results: dict[str, dict],
        search_results: dict[str, list[dict]],
    ) -> dict:
        """Run validation for each drug component. Loads checkpoint first."""
        existing = await self._load_checkpoint(input, "drug_class_validation")
        if existing is not None:
            workflow.logger.info(
                f"Loaded drug class validation from checkpoint for {input.abstract_id}"
            )
            return existing

        results = []
        errors = []

        for component, extraction_result in extraction_results.items():
            drug_classes = extraction_result.get("drug_classes", [])
            if not drug_classes or drug_classes == ["NA"]:
                continue
            try:
                validation_result = await workflow.execute_activity(
                    validate_drug_class_activity,
                    DrugClassValidationInput(
                        abstract_id=input.abstract_id,
                        drug_name=component,
                        abstract_title=input.abstract_title,
                        full_abstract=input.full_abstract,
                        search_results=search_results.get(component, []),
                        extraction_result=extraction_result,
                    ),
                    task_queue=TaskQueues.DRUG_CLASS,
                    start_to_close_timeout=Timeouts.FAST_LLM,
                    retry_policy=RetryPolicies.FAST_LLM,
                )
                results.append({
                    "drug_name": component, "validation": validation_result,
                })
            except Exception as e:
                workflow.logger.error(f"Validation failed for drug '{component}': {e}")
                errors.append(f"Validation error for {component}: {e}")

        validation_data = {"results": results, "errors": errors}
        # Only checkpoint if all validations passed - allows retry of failures
        if not errors:
            await self._save_checkpoint(input, "drug_class_validation", validation_data)
        return validation_data
