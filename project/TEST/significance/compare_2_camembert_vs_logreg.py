"""
Comparison 2: CamemBERT-FR Final vs Logistic Regression (baseline ML)

Objective: Demonstrate that the transformer model outperforms traditional
ML approaches (logistic regression + TF-IDF).
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

# CamemBERT-FR Final
PATH_MODEL_A = "../../result_output_camembert_final/crows_fr_final/full_results.csv"
MODEL_A_NAME = "CamemBERT-FR-Final"

# Logistic Regression + TF-IDF
PATH_MODEL_B = "../../result_output_LR_tfidf/crows_fr/full_results.csv"
MODEL_B_NAME = "LogisticRegression-TFIDF"

# Output folder
OUTPUT_DIR = "../../results_significance/camembert_vs_logreg"

# Bootstrap parameters
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COMPARISON 2: CamemBERT-FR Final vs Logistic Regression")
    print("=" * 70)
    print("\nObjective: Transformer (CamemBERT) vs traditional ML (LR + TF-IDF)\n")

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
