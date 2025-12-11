"""
Statistical tests to compare two models de classification de texte
Uses McNemar's test and Bootstrap pour évaluer la significativité des différences

Author: HEARTS Project
Date: 2025
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import f1_score, precision_recall_fscore_support
import os

# ============================================================================
# PARAMETERS TO CONFIGURE
# ============================================================================

# Paths to the result files of the two models
PATH_MODEL_A = "../../Model/result_output_camembert_final/crows_fr_final/full_results.csv"
PATH_MODEL_B = "../../result_output_distilbert/crows_fr/full_results.csv"

# Names of the models (for reports)
MODEL_A_NAME = "CamemBERT"
MODEL_B_NAME = "DistilBERT"

# Output folder for results
OUTPUT_DIR = "../../results_significance_tests"

# Number of bootstrap iterations
N_BOOTSTRAP = 2000

# Seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_and_align_predictions(path_a, path_b):
    """
    Load two full_results.csv files and align the predictions

    Args:
        path_a: Path to model A file
        path_b: Path to model B file

    Returns:
        DataFrame with columns: text, actual_label, pred_a, pred_b
    """
    print("Loading result files...")

    # Charger les deux fichiers
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    print(f"Model A: {len(df_a)} prédictions")
    print(f"Model B: {len(df_b)} prédictions")

    # Select necessary columns
    df_a = df_a[['text', 'actual_label', 'predicted_label']].rename(
        columns={'predicted_label': 'pred_a'}
    )
    df_b = df_b[['text', 'actual_label', 'predicted_label']].rename(
        columns={'predicted_label': 'pred_b'}
    )

    # Merge on text and actual_label
    df_aligned = df_a.merge(df_b, on=['text', 'actual_label'], how='inner')

    print(f"After alignment: {len(df_aligned)} examples communs")

    if len(df_aligned) == 0:
        raise ValueError("No common examples found between the two models!")

    return df_aligned


def mcnemar_test(df_aligned):
    """
    Calculate McNemar's test pour comparer deux classifieurs

    Args:
        df_aligned: DataFrame with columns actual_label, pred_a, pred_b

    Returns:
        dict with the résultats du test
    """
    print("\n" + "="*60)
    print("MCNEMAR'S TEST")
    print("="*60)

    y_true = df_aligned['actual_label'].values
    y_pred_a = df_aligned['pred_a'].values
    y_pred_b = df_aligned['pred_b'].values

    # Calculate correct/incorrect
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table
    # b = A correct, B incorrect
    # c = A incorrect, B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    # Cases où les deux sont corrects ou incorrects
    both_correct = np.sum(correct_a & correct_b)
    both_incorrect = np.sum(~correct_a & ~correct_b)

    print(f"\nContingency table:")
    print(f"  Both correct:     {both_correct:4d}")
    print(f"  Both incorrect:   {both_incorrect:4d}")
    print(f"  A correct, B incorrect (b): {b:4d}")
    print(f"  A incorrect, B correct (c): {c:4d}")

    # McNemar's test with continuity correction
    if b + c == 0:
        print("\nWARNING: b + c = 0, cannot calculate the test")
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)

    print(f"\nResults:")
    print(f"  Chi² = {chi2_stat:.4f}")
    print(f"  p-value = {p_value:.6f}")

    if p_value < 0.001:
        significance = "*** (p < 0.001)"
    elif p_value < 0.01:
        significance = "** (p < 0.01)"
    elif p_value < 0.05:
        significance = "* (p < 0.05)"
    else:
        significance = "ns (not significant)"

    print(f"  Significance: {significance}")

    results = {
        'both_correct': both_correct,
        'both_incorrect': both_incorrect,
        'b_correct_a_incorrect': b,
        'c_incorrect_a_correct': c,
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'significance': significance
    }

    return results


def bootstrap_f1_difference(df_aligned, n_iterations=2000, random_seed=42):
    """
    Non-parametric bootstrap pour évaluer la différence de macro-F1

    Args:
        df_aligned: DataFrame with columns actual_label, pred_a, pred_b
        n_iterations: Nombre d'iterations bootstrap
        random_seed: Seed for reproducibility

    Returns:
        dict with the résultats du bootstrap
    """
    print("\n" + "="*60)
    print("BOOTSTRAP FOR MACRO-F1 DIFFERENCE")
    print("="*60)

    np.random.seed(random_seed)

    y_true = df_aligned['actual_label'].values
    y_pred_a = df_aligned['pred_a'].values
    y_pred_b = df_aligned['pred_b'].values

    n_samples = len(y_true)

    # Calculer les F1 sur l'ensemble complet (baseline)
    f1_a_full = f1_score(y_true, y_pred_a, average='macro', zero_division=0)
    f1_b_full = f1_score(y_true, y_pred_b, average='macro', zero_division=0)
    delta_f1_full = f1_b_full - f1_a_full

    print(f"\nF1 macro on the full dataset:")
    print(f"  Model A: {f1_a_full:.4f}")
    print(f"  Model B: {f1_b_full:.4f}")
    print(f"  ΔF1 (B - A): {delta_f1_full:.4f}")

    # Bootstrap
    print(f"\nRunning bootstrap ({n_iterations} iterations)...")
    delta_f1_distribution = []

    for i in range(n_iterations):
        # Random sampling with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        y_true_boot = y_true[indices]
        y_pred_a_boot = y_pred_a[indices]
        y_pred_b_boot = y_pred_b[indices]

        # Calculate F1 for each model
        f1_a_boot = f1_score(y_true_boot, y_pred_a_boot, average='macro', zero_division=0)
        f1_b_boot = f1_score(y_true_boot, y_pred_b_boot, average='macro', zero_division=0)

        # Calculate the difference
        delta_f1_boot = f1_b_boot - f1_a_boot
        delta_f1_distribution.append(delta_f1_boot)

    delta_f1_distribution = np.array(delta_f1_distribution)

    # Distribution statistics
    mean_delta = np.mean(delta_f1_distribution)
    std_delta = np.std(delta_f1_distribution)
    ci_lower = np.percentile(delta_f1_distribution, 2.5)
    ci_upper = np.percentile(delta_f1_distribution, 97.5)

    print(f"\nBootstrap results:")
    print(f"  Mean ΔF1: {mean_delta:.4f}")
    print(f"  Standard deviation: {std_delta:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Test if 0 is in the confidence interval
    if ci_lower > 0:
        interpretation = "Model B significantly better (CI does not contain 0)"
    elif ci_upper < 0:
        interpretation = "Model A significantly better (CI does not contain 0)"
    else:
        interpretation = "No significant difference (IC contient 0)"

    print(f"  Interprétation: {interpretation}")

    results = {
        'f1_model_a': f1_a_full,
        'f1_model_b': f1_b_full,
        'delta_f1_observed': delta_f1_full,
        'delta_f1_mean': mean_delta,
        'delta_f1_std': std_delta,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'interpretation': interpretation,
        'distribution': delta_f1_distribution
    }

    return results


def save_results(mcnemar_results, bootstrap_results, output_dir, model_a_name, model_b_name):
    """
    Save the results dans des fichiers texte

    Args:
        mcnemar_results: Results of McNemar's test
        bootstrap_results: Bootstrap results
        output_dir: Output folder
        model_a_name: Nom du modèle A
        model_b_name: Nom du modèle B
    """
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # McNemar file
    mcnemar_file = os.path.join(output_dir, "mcnemar_results.txt")
    with open(mcnemar_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("MCNEMAR'S TEST\n")
        f.write("="*60 + "\n\n")
        f.write(f"Comparison: {model_a_name} vs {model_b_name}\n\n")
        f.write("Contingency table:\n")
        f.write(f"  Both correct:          {mcnemar_results['both_correct']:4d}\n")
        f.write(f"  Both incorrect:        {mcnemar_results['both_incorrect']:4d}\n")
        f.write(f"  {model_a_name} correct, {model_b_name} incorrect (b): {mcnemar_results['b_correct_a_incorrect']:4d}\n")
        f.write(f"  {model_a_name} incorrect, {model_b_name} correct (c): {mcnemar_results['c_incorrect_a_correct']:4d}\n\n")
        f.write("Results:\n")
        f.write(f"  Chi² statistic: {mcnemar_results['chi2_statistic']:.4f}\n")
        f.write(f"  p-value:          {mcnemar_results['p_value']:.6f}\n")
        f.write(f"  Significance:  {mcnemar_results['significance']}\n\n")

        if mcnemar_results['p_value'] < 0.05:
            f.write("CONCLUSION: There is a significant difference entre les deux modèles.\n")
        else:
            f.write("CONCLUSION: No significant difference entre les deux modèles.\n")

    print(f"McNemar results saved: {mcnemar_file}")

    # Bootstrap file
    bootstrap_file = os.path.join(output_dir, "bootstrap_results.txt")
    with open(bootstrap_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("BOOTSTRAP FOR MACRO-F1 DIFFERENCE\n")
        f.write("="*60 + "\n\n")
        f.write(f"Comparison: {model_a_name} vs {model_b_name}\n")
        f.write(f"Nombre d'iterations: {N_BOOTSTRAP}\n\n")
        f.write("F1 macro on the full dataset:\n")
        f.write(f"  {model_a_name}: {bootstrap_results['f1_model_a']:.4f}\n")
        f.write(f"  {model_b_name}: {bootstrap_results['f1_model_b']:.4f}\n")
        f.write(f"  Observed ΔF1 ({model_b_name} - {model_a_name}): {bootstrap_results['delta_f1_observed']:.4f}\n\n")
        f.write("Bootstrap results:\n")
        f.write(f"  Mean ΔF1:     {bootstrap_results['delta_f1_mean']:.4f}\n")
        f.write(f"  Standard deviation:         {bootstrap_results['delta_f1_std']:.4f}\n")
        f.write(f"  95% CI:             [{bootstrap_results['ci_95_lower']:.4f}, {bootstrap_results['ci_95_upper']:.4f}]\n\n")
        f.write(f"INTERPRETATION: {bootstrap_results['interpretation']}\n")

    print(f"Bootstrap results saved: {bootstrap_file}")

    # Also save the complete distribution
    distribution_file = os.path.join(output_dir, "bootstrap_distribution.csv")
    pd.DataFrame({
        'delta_f1': bootstrap_results['distribution']
    }).to_csv(distribution_file, index=False)
    print(f"Bootstrap distribution saved: {distribution_file}")


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("STATISTICAL TESTS FOR MODEL COMPARISON")
    print("="*60)
    print(f"\nModel A: {MODEL_A_NAME}")
    print(f"File: {PATH_MODEL_A}")
    print(f"\nModel B: {MODEL_B_NAME}")
    print(f"File: {PATH_MODEL_B}")

    # 1. Load and align predictions
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
    save_results(mcnemar_results, bootstrap_results, OUTPUT_DIR, MODEL_A_NAME, MODEL_B_NAME)

    print("\n" + "="*60)
    print("TESTS COMPLETED")
    print("="*60)
