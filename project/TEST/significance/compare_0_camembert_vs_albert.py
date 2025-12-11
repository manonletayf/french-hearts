"""
Comparison 0: CamemBERT-FR Final vs ALBERT-baseline

Objective: Compare CamemBERT (pre-trained French model) with ALBERT
(multilingual English model) on stereotype detection in French.

This comparison allows us to validate the hypothesis that using a
French model (CamemBERT) improves performance compared to an
English model (ALBERT) on French data.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from statistical_comparison import (
    load_and_align_predictions,
    mcnemar_test,
    bootstrap_f1_difference,
    save_results
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# CamemBERT-FR Final (French model)
PATH_MODEL_A = "../../Model/result_output_camembert_final/crows_fr_final/full_results.csv"
MODEL_A_NAME = "CamemBERT-FR-Final"

# ALBERT-baseline (English model)
# Uses the simple script: baseline_albert_crows_simple.py
PATH_MODEL_B = "../../Model/result_output_albert_crows_simple/crows_fr_trained/crows_fr/full_results.csv"
MODEL_B_NAME = "ALBERT-base-v2"

# Output folder
OUTPUT_DIR = "../../results_significance/camembert_vs_albert"

# Bootstrap parameters
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COMPARISON 0: CamemBERT-FR Final vs ALBERT-baseline")
    print("=" * 70)
    print("\nObjective: French model (CamemBERT) vs English model (ALBERT)")
    print("           on stereotype detection in French\n")

    # 1. Load and align predictions
    print("Loading results...")
    try:
        df_aligned = load_and_align_predictions(PATH_MODEL_A, PATH_MODEL_B)
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("\nPlease first run the ALBERT training script:")
        print("  cd ../../")
        print("  python baseline_albert_crows_simple.py")
        sys.exit(1)

    # 2. McNemar's test
    mcnemar_results = mcnemar_test(df_aligned)

    # 3. Bootstrap for F1 difference
    bootstrap_results = bootstrap_f1_difference(
        df_aligned,
        n_iterations=N_BOOTSTRAP,
        random_seed=RANDOM_SEED
    )

    # 4. Save results
    save_results(mcnemar_results, bootstrap_results, OUTPUT_DIR,
                 MODEL_A_NAME, MODEL_B_NAME)

    # 5. Summary in terminal
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{MODEL_A_NAME} (French model):")
    print(f"  F1-macro: {bootstrap_results['f1_model_a']:.4f}")
    print(f"\n{MODEL_B_NAME} (English model):")
    print(f"  F1-macro: {bootstrap_results['f1_model_b']:.4f}")
    print(f"\nDifference (B - A): {bootstrap_results['delta_f1_observed']:+.4f}")
    print(f"95% CI: [{bootstrap_results['ci_95_lower']:.4f}, {bootstrap_results['ci_95_upper']:.4f}]")
    print(f"\nMcNemar's test: p = {mcnemar_results['p_value']:.6f} {mcnemar_results['significance']}")
    print(f"Bootstrap: {bootstrap_results['interpretation']}")

    # 6. Specific interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if mcnemar_results['p_value'] < 0.05 and bootstrap_results['ci_95_lower'] > 0:
        print("\n✓ CamemBERT (French) is SIGNIFICANTLY BETTER than ALBERT (English)")
        print("  for stereotype detection in French.")
        print("\n→ Conclusion: Using a pre-trained French model")
        print("  significantly improves performance on this task.")
    elif mcnemar_results['p_value'] < 0.05 and bootstrap_results['ci_95_upper'] < 0:
        print("\n✗ ALBERT (English) is SIGNIFICANTLY BETTER than CamemBERT (French)")
        print("  for stereotype detection in French.")
        print("\n→ Unexpected result: The English model outperforms the French model.")
    else:
        print("\n○ No significant difference between CamemBERT and ALBERT.")
        print("\n→ Both models have comparable performance on this task.")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print(f"Results saved in: {OUTPUT_DIR}/")
    print("=" * 70)
