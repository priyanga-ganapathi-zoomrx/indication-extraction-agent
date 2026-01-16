# src/scripts/drug_class/__init__.py
"""Drug class processing scripts package.

This package contains batch processors for drug class extraction:
- batch_processor: Step-centric batch processing (all steps)
- step1_processor: Regimen identification only
- step2_processor: Drug class extraction only
- step3_processor: Drug class selection only
- step4_processor: Explicit extraction only
- step5_processor: Consolidation only
- validation_processor: Validation of extractions

Step-Centric Processing:
    Processes all abstracts through Step 1, then all through Step 2, etc.
    This approach provides better control over parallelism and token usage.

Per-Abstract Status:
    Each abstract has a status file (abstracts/{id}/status.json) that tracks
    which steps have completed. Rerunning the script will automatically
    skip completed steps and resume from where each abstract left off.

Example usage:
    # Run full pipeline step-by-step
    python -m src.scripts.drug_class.batch_processor --input data/input.csv
    
    # Run individual steps for testing
    python -m src.scripts.drug_class.step1_processor --input data/input.csv
    python -m src.scripts.drug_class.step2_processor --input data/input.csv
    
    # Validate extractions
    python -m src.scripts.drug_class.validation_processor --input data/extraction_results.csv
"""

from src.scripts.drug_class.batch_processor import (
    load_abstracts,
    get_abstract_status,
    get_step_from_status,
    get_abstracts_at_step,
    run_step_batch,
    main as batch_main,
)

from src.scripts.drug_class.step1_processor import (
    process_single as step1_process_single,
    main as step1_main,
)

from src.scripts.drug_class.step2_processor import (
    process_single as step2_process_single,
    main as step2_main,
)

from src.scripts.drug_class.step3_processor import (
    process_single as step3_process_single,
    main as step3_main,
)

from src.scripts.drug_class.step4_processor import (
    process_single as step4_process_single,
    main as step4_main,
)

from src.scripts.drug_class.step5_processor import (
    process_single as step5_process_single,
    main as step5_main,
)

from src.scripts.drug_class.validation_processor import (
    load_abstracts as load_validation_abstracts,
    process_single as validation_process_single,
    main as validation_main,
)

__all__ = [
    # Batch processor
    "load_abstracts",
    "get_abstract_status",
    "get_step_from_status",
    "get_abstracts_at_step",
    "run_step_batch",
    "batch_main",
    # Step processors
    "step1_process_single",
    "step1_main",
    "step2_process_single",
    "step2_main",
    "step3_process_single",
    "step3_main",
    "step4_process_single",
    "step4_main",
    "step5_process_single",
    "step5_main",
    # Validation processor
    "load_validation_abstracts",
    "validation_process_single",
    "validation_main",
]

