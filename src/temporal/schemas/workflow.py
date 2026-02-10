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
    Token usage is automatically extracted from activity output
    and stored here for status.json updates.
    """
    status: str  # "success" or "failed"
    output: Optional[dict] = None
    from_checkpoint: bool = False
    error: Optional[str] = None
    token_usage: Optional[dict] = None  # {"input_tokens": N, "output_tokens": N, "total_tokens": N}
    llm_calls: int = 1

    def to_step_status(self) -> StepStatus:
        """Convert to StepStatus for status.json.

        Reads token usage from self.token_usage automatically.
        """
        if self.status == "success":
            return StepStatus.success(
                llm_calls=self.llm_calls,
                tokens=self.token_usage.get("total_tokens", 0) if self.token_usage else 0,
                input_tokens=self.token_usage.get("input_tokens", 0) if self.token_usage else 0,
                output_tokens=self.token_usage.get("output_tokens", 0) if self.token_usage else 0,
            )
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

    @property
    def status(self) -> str:
        """Derive status from completion state and errors."""
        if not self.completed:
            return "failed"
        if self.errors:
            return "partial_success"
        return "success"
