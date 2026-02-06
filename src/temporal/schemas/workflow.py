"""Workflow input/output schemas for the abstract extraction workflow.

These are pure data classes with no dependency on Temporal's workflow module.
They are imported by the workflow via `workflow.unsafe.imports_passed_through()`.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.temporal.schemas.status import StepStatus


@dataclass
class AbstractExtractionInput:
    """Input for the abstract extraction workflow.

    Best Practice: Single dataclass input allows adding fields
    without breaking existing workflow executions.
    """
    abstract_id: str
    abstract_title: str
    session_title: str = ""
    full_abstract: str = ""
    firms: list[str] = field(default_factory=list)

    # Storage path for checkpoints (gs://bucket/prefix or local path)
    # If empty, no checkpointing is performed
    storage_path: str = ""

    # Which pipelines to run (default: all three)
    # Options: "drug", "drug_class", "indication"
    pipelines: list[str] = field(
        default_factory=lambda: ["drug", "drug_class", "indication"]
    )


@dataclass
class StepResult:
    """Result of a single step execution.

    Returned by _run_with_checkpoint to indicate step outcome.
    """
    status: str  # "success" or "failed"
    output: Optional[dict] = None
    from_checkpoint: bool = False
    error: Optional[str] = None

    def to_step_status(self, llm_calls: int = 1, tokens: int = 0) -> StepStatus:
        """Convert to StepStatus for status.json."""
        if self.status == "success":
            return StepStatus.success(llm_calls=llm_calls, tokens=tokens)
        return StepStatus.failed(self.error or "Unknown error")


@dataclass
class DrugResult:
    """Result of drug extraction and validation."""
    extraction: dict = field(default_factory=dict)
    validation: Optional[dict] = None


@dataclass
class DrugClassResult:
    """Result of drug class pipeline."""
    drug_results: list[dict] = field(default_factory=list)
    explicit_classes: list[str] = field(default_factory=list)
    refined_explicit_classes: list[str] = field(default_factory=list)
    validation_results: list[dict] = field(default_factory=list)


@dataclass
class IndicationResult:
    """Result of indication extraction and validation."""
    extraction: dict = field(default_factory=dict)
    validation: Optional[dict] = None


@dataclass
class AbstractExtractionOutput:
    """Output from the abstract extraction workflow."""
    abstract_id: str
    drug: DrugResult = field(default_factory=DrugResult)
    drug_class: DrugClassResult = field(default_factory=DrugClassResult)
    indication: IndicationResult = field(default_factory=IndicationResult)
    completed: bool = False
    errors: list[str] = field(default_factory=list)
