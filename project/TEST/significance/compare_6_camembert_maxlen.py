"""
Comparison 6: CamemBERT max_length (hyperparameter)

Objective: Ablation study to evaluate the impact of the max_length parameter
on model performance.

CONFIGURATION ACTUELLE:
- Model A: CamemBERT Final (max_length par défaut, probablement 512)
- Model B: Ablation max_length=128

Si vous avez d'autres variantes (256), ajustez les chemins ci-dessous.
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

# CamemBERT-FR Final (max_length par défaut, probablement 512)
PATH_MODEL_A = "../../Model/result_output_camembert_final/crows_fr_final/full_results.csv"
MODEL_A_NAME = "CamemBERT-maxlen-512"

# Ablation max_length=128
PATH_MODEL_B = "../../result_output_ablation_maxlen128/crows_fr/full_results.csv"
MODEL_B_NAME = "Ablation-maxlen-128"

# Output folder
OUTPUT_DIR = "../../results_significance/camembert_maxlen_ablation"

# Bootstrap parameters
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COMPARISON 6: CamemBERT max_length=512 vs max_length=128")
    print("=" * 70)
    print("\nObjectif: Impact du paramètre max_length sur les performances\n")

    # 1. Load and align predictions
    print("Loading results...")
    df_aligned = load_and_align_predictions(PATH_MODEL_A, PATH_MODEL_B)

    # 2. McNemar's test
    mcnemar_results = mcnemar_test(df_aligned)

    # 3. Bootstrap for the difference de F1
    bootstrap_results = bootstrap_f1_difference(
        df_aligned,
        n_iterations=N_BOOTSTRAP,
        random_seed=RANDOM_SEED
    )

    # 4. Save the results
    save_results(mcnemar_results, bootstrap_results, OUTPUT_DIR,
                 MODEL_A_NAME, MODEL_B_NAME)

    # 5. Summary in terminal
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{MODEL_A_NAME}:")
    print(f"  F1-macro: {bootstrap_results['f1_model_a']:.4f}")
    print(f"\n{MODEL_B_NAME}:")
    print(f"  F1-macro: {bootstrap_results['f1_model_b']:.4f}")
    print(f"\nDifference (B - A): {bootstrap_results['delta_f1_observed']:+.4f}")
    print(f"95% CI: [{bootstrap_results['ci_95_lower']:.4f}, {bootstrap_results['ci_95_upper']:.4f}]")
    print(f"\nMcNemar's test: p = {mcnemar_results['p_value']:.6f} {mcnemar_results['significance']}")
    print(f"Bootstrap: {bootstrap_results['interpretation']}")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print(f"Results saved in: {OUTPUT_DIR}/")
    print("=" * 70)
