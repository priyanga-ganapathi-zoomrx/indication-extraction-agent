"""Drug class extraction module.

Revamped module with function-based architecture and per-step checkpointing.
Designed for Temporal integration.

Pipeline Steps:
1. Regimen Identification - Split drug regimens into components
2. Drug Class Extraction - Extract classes via Tavily + Grounded search
3. Drug Class Selection - Select best class per drug
4. Explicit Extraction - Extract classes from abstract title
5. Consolidation - Deduplicate and refine classes
"""

# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================
from src.agents.drug_class.pipeline import run_drug_class_pipeline
from src.agents.core.storage import StorageClient, LocalStorageClient

# =============================================================================
# STEP FUNCTIONS (Single-item, atomic functions for Temporal activities)
# =============================================================================
from src.agents.drug_class.step1_regimen import identify_regimen
from src.agents.drug_class.step2_search import (
    fetch_search_results,
    search_drug_class,
    search_drug_firm,
    load_search_cache,
    save_search_cache,
)
from src.agents.drug_class.step2_extraction import (
    extract_with_tavily,
    extract_with_grounded,
)
from src.agents.drug_class.step3_selection import (
    select_drug_class,
    needs_llm_selection,
)
from src.agents.drug_class.step4_explicit import extract_explicit_classes
from src.agents.drug_class.step5_consolidation import consolidate_drug_classes
from src.agents.drug_class.validation import validate_drug_class

# =============================================================================
# SCHEMAS
# =============================================================================
from src.agents.drug_class.schemas import (
    # Pipeline input/output
    DrugClassInput,
    DrugClassPipelineResult,
    # Step 1 schemas
    RegimenInput,
    RegimenLLMResponse,
    Step1Output,
    # Step 2 schemas
    DrugSearchCache,
    DrugClassExtractionInput,
    DrugClassLLMResponse,
    GroundedSearchClassDetail,
    GroundedSearchLLMResponse,
    DrugExtractionResult,
    ExtractionDetail,
    Step2Output,
    # Step 3 schemas
    SelectionInput,
    DrugSelectionResult,
    Step3Output,
    # Step 4 schemas
    ExplicitExtractionInput,
    ExplicitExtractionDetail,
    ExplicitLLMResponse,
    Step4Output,
    # Step 5 schemas
    ConsolidationInput,
    RemovedClassInfo,
    RefinedExplicitClasses,
    ConsolidationLLMResponse,
    Step5Output,
    # Validation schemas
    ValidationInput,
    ValidationIssue,
    CheckResult,
    ChecksPerformed,
    ValidationLLMResponse,
    ValidationOutput,
    # Status tracking
    PipelineStatus,
    StepResult,
    # Errors
    DrugClassExtractionError,
    DrugClassPipelineError,
)

# =============================================================================
# CONFIG
# =============================================================================
from src.agents.drug_class.config import config

# =============================================================================
# PROMPTS
# =============================================================================
from src.agents.drug_class.prompts import (
    # Helper
    extract_section,
    # Prompt loaders
    get_system_prompt,
    get_extraction_title_prompt,
    get_extraction_rules_prompt,
    get_validation_prompt,
    get_selection_prompt,
    get_grounded_search_prompt,
    get_consolidation_prompt,
    get_regimen_identification_prompt,
    # Parsed prompt loaders
    get_extraction_rules_prompt_parts,
    get_grounded_search_prompt_parts,
    get_selection_prompt_parts,
    get_explicit_extraction_prompt_parts,
    get_consolidation_prompt_parts,
    get_validation_prompt_parts,
    # Prompt names
    EXTRACTION_TITLE_PROMPT_NAME,
    EXTRACTION_RULES_PROMPT_NAME,
    VALIDATION_PROMPT_NAME,
    SELECTION_PROMPT_NAME,
    GROUNDED_SEARCH_PROMPT_NAME,
    CONSOLIDATION_PROMPT_NAME,
    REGIMEN_IDENTIFICATION_PROMPT_NAME,
)

__all__ = [
    # Main pipeline
    "run_drug_class_pipeline",
    "StorageClient",
    "LocalStorageClient",
    
    # Step 1 functions
    "identify_regimen",
    
    # Step 2 functions (search + extraction)
    "fetch_search_results",
    "search_drug_class",
    "search_drug_firm",
    "load_search_cache",
    "save_search_cache",
    "extract_with_tavily",
    "extract_with_grounded",
    
    # Step 3-5 functions
    "select_drug_class",
    "needs_llm_selection",
    "extract_explicit_classes",
    "consolidate_drug_classes",
    
    # Validation function
    "validate_drug_class",
    
    # Pipeline schemas
    "DrugClassInput",
    "DrugClassPipelineResult",
    
    # Step 1 schemas
    "RegimenInput",
    "RegimenLLMResponse",
    "Step1Output",
    
    # Step 2 schemas
    "DrugSearchCache",
    "DrugClassExtractionInput",
    "DrugClassLLMResponse",
    "GroundedSearchClassDetail",
    "GroundedSearchLLMResponse",
    "DrugExtractionResult",
    "ExtractionDetail",
    "Step2Output",
    
    # Step 3 schemas
    "SelectionInput",
    "DrugSelectionResult",
    "Step3Output",
    
    # Step 4 schemas
    "ExplicitExtractionInput",
    "ExplicitExtractionDetail",
    "ExplicitLLMResponse",
    "Step4Output",
    
    # Step 5 schemas
    "ConsolidationInput",
    "RemovedClassInfo",
    "RefinedExplicitClasses",
    "ConsolidationLLMResponse",
    "Step5Output",
    
    # Validation schemas
    "ValidationInput",
    "ValidationIssue",
    "CheckResult",
    "ChecksPerformed",
    "ValidationLLMResponse",
    "ValidationOutput",
    
    # Status tracking
    "PipelineStatus",
    "StepResult",
    
    # Errors
    "DrugClassExtractionError",
    "DrugClassPipelineError",
    
    # Config
    "config",
    
    # Prompts
    "extract_section",
    "get_system_prompt",
    "get_extraction_title_prompt",
    "get_extraction_rules_prompt",
    "get_validation_prompt",
    "get_selection_prompt",
    "get_grounded_search_prompt",
    "get_consolidation_prompt",
    "get_regimen_identification_prompt",
    "get_extraction_rules_prompt_parts",
    "get_grounded_search_prompt_parts",
    "get_selection_prompt_parts",
    "get_explicit_extraction_prompt_parts",
    "get_consolidation_prompt_parts",
    "get_validation_prompt_parts",
    "EXTRACTION_TITLE_PROMPT_NAME",
    "EXTRACTION_RULES_PROMPT_NAME",
    "VALIDATION_PROMPT_NAME",
    "SELECTION_PROMPT_NAME",
    "GROUNDED_SEARCH_PROMPT_NAME",
    "CONSOLIDATION_PROMPT_NAME",
    "REGIMEN_IDENTIFICATION_PROMPT_NAME",
]
