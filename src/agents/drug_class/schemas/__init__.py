"""Drug class extraction schemas package.

This package organizes schemas by concern:
- base: Type aliases and shared types
- inputs: Dataclass inputs for each step
- llm_responses: LLM structured output schemas
- outputs: Step outputs and result schemas
- pipeline: Status tracking and final result
- validation: Validation-specific schemas
- errors: Custom exception classes

All schemas are re-exported here for backward compatibility.
"""

# =============================================================================
# BASE TYPES
# =============================================================================
from src.agents.drug_class.schemas.base import (
    StepName,
    StepStatus,
    DrugStatus,
    ClassType,
    ConfidenceLevel,
)

# =============================================================================
# INPUTS
# =============================================================================
from src.agents.drug_class.schemas.inputs import (
    DrugClassInput,
    RegimenInput,
    DrugClassExtractionInput,
    SelectionInput,
    ExplicitExtractionInput,
    ConsolidationInput,
    ValidationInput,
)

# =============================================================================
# LLM RESPONSES
# =============================================================================
from src.agents.drug_class.schemas.llm_responses import (
    # Step 1
    RegimenLLMResponse,
    # Step 2 - Tavily
    ExtractionDetail,
    DrugClassLLMResponse,
    # Step 2 - Grounded
    GroundedSearchClassDetail,
    GroundedSearchLLMResponse,
    # Step 4
    ExplicitExtractionDetail,
    ExplicitLLMResponse,
    # Step 5
    RemovedClassInfo,
    RefinedExplicitClasses,
    ConsolidationLLMResponse,
)

# =============================================================================
# OUTPUTS
# =============================================================================
from src.agents.drug_class.schemas.outputs import (
    # Step 1
    Step1Output,
    # Step 2
    DrugSearchCache,
    DrugExtractionResult,
    Step2Output,
    # Step 3
    DrugSelectionResult,
    Step3Output,
    # Step 4
    Step4Output,
    # Step 5
    Step5Output,
)

# =============================================================================
# VALIDATION
# =============================================================================
from src.agents.drug_class.schemas.validation import (
    ValidationIssue,
    CheckResult,
    ChecksPerformed,
    ValidationLLMResponse,
    ValidationOutput,
)

# =============================================================================
# PIPELINE
# =============================================================================
from src.agents.drug_class.schemas.pipeline import (
    StepResult,
    PipelineStatus,
    DrugClassMapping,
    DrugClassPipelineResult,
)

# =============================================================================
# ERRORS
# =============================================================================
from src.agents.drug_class.schemas.errors import (
    DrugClassExtractionError,
    DrugClassPipelineError,
)

__all__ = [
    # Base types
    "StepName",
    "StepStatus",
    "DrugStatus",
    "ClassType",
    "ConfidenceLevel",
    # Inputs
    "DrugClassInput",
    "RegimenInput",
    "DrugClassExtractionInput",
    "SelectionInput",
    "ExplicitExtractionInput",
    "ConsolidationInput",
    "ValidationInput",
    # LLM Responses
    "RegimenLLMResponse",
    "ExtractionDetail",
    "DrugClassLLMResponse",
    "GroundedSearchClassDetail",
    "GroundedSearchLLMResponse",
    "ExplicitExtractionDetail",
    "ExplicitLLMResponse",
    "RemovedClassInfo",
    "RefinedExplicitClasses",
    "ConsolidationLLMResponse",
    # Outputs
    "Step1Output",
    "DrugSearchCache",
    "DrugExtractionResult",
    "Step2Output",
    "DrugSelectionResult",
    "Step3Output",
    "Step4Output",
    "Step5Output",
    # Validation
    "ValidationIssue",
    "CheckResult",
    "ChecksPerformed",
    "ValidationLLMResponse",
    "ValidationOutput",
    # Pipeline
    "StepResult",
    "PipelineStatus",
    "DrugClassMapping",
    "DrugClassPipelineResult",
    # Errors
    "DrugClassExtractionError",
    "DrugClassPipelineError",
]

