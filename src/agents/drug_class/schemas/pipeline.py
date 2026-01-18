"""Pipeline-level schemas for drug class extraction.

Contains status tracking and final result schemas.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.agents.drug_class.schemas.base import StepName, StepStatus


# =============================================================================
# STEP RESULT
# =============================================================================

@dataclass
class StepResult:
    """Result of a single pipeline step (for checkpointing)."""
    step_name: StepName
    status: StepStatus
    output: Optional[dict] = None  # The actual step output as dict
    llm_calls: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0


# =============================================================================
# PIPELINE STATUS
# =============================================================================

@dataclass
class PipelineStatus:
    """Status tracking for the pipeline."""
    abstract_id: str
    abstract_title: str
    
    # Overall status
    pipeline_status: StepStatus = "pending"
    last_completed_step: Optional[StepName] = None
    failed_step: Optional[StepName] = None
    error: Optional[str] = None
    
    # Per-step status
    steps: dict[StepName, dict] = field(default_factory=dict)
    
    # Metrics
    total_llm_calls: int = 0
    last_updated: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "abstract_id": self.abstract_id,
            "abstract_title": self.abstract_title,
            "pipeline_status": self.pipeline_status,
            "last_completed_step": self.last_completed_step,
            "failed_step": self.failed_step,
            "error": self.error,
            "steps": self.steps,
            "total_llm_calls": self.total_llm_calls,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PipelineStatus":
        """Create from dictionary."""
        return cls(
            abstract_id=data["abstract_id"],
            abstract_title=data["abstract_title"],
            pipeline_status=data.get("pipeline_status", "pending"),
            last_completed_step=data.get("last_completed_step"),
            failed_step=data.get("failed_step"),
            error=data.get("error"),
            steps=data.get("steps", {}),
            total_llm_calls=data.get("total_llm_calls", 0),
            last_updated=data.get("last_updated"),
        )


# =============================================================================
# DRUG CLASS MAPPING (Per-Drug Result)
# =============================================================================

@dataclass
class DrugClassMapping:
    """Mapping of a drug to its components and selected classes."""
    drug: str
    components: list[dict]  # [{component, selected_classes}, ...]


# =============================================================================
# FINAL PIPELINE RESULT
# =============================================================================

@dataclass
class DrugClassPipelineResult:
    """Final output of the drug class extraction pipeline."""
    abstract_id: str
    abstract_title: str
    
    # Per-drug results (from Steps 1-3)
    drug_class_mappings: list[dict] = field(default_factory=list)
    
    # Explicit classes (from Steps 4-5)
    explicit_drug_classes: list[str] = field(default_factory=list)
    
    # Metadata
    success: bool = True
    error: Optional[str] = None
    total_llm_calls: int = 0
    
    # References to checkpoint files
    checkpoint_files: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "abstract_id": self.abstract_id,
            "abstract_title": self.abstract_title,
            "drug_class_mappings": self.drug_class_mappings,
            "explicit_drug_classes": self.explicit_drug_classes,
            "success": self.success,
            "error": self.error,
            "total_llm_calls": self.total_llm_calls,
            "checkpoint_files": self.checkpoint_files,
        }

