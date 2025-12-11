"""
Failure Case Analysis - CamemBERT-FR Final

This script generates a detailed analysis of model errors :
1. Confusion matrix (visualization + statistics)
2. F1 variance between groups of stereotypes
3. Analysis by group (gender, race, religion, etc.)

Usage:
    python analyze_failures.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import defaultdict
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model results file to analyze
RESULTS_FILE = "../../result_output_camembert_final/crows_fr_final/full_results.csv"
MODEL_NAME = "CamemBERT-FR-Final"

# Output folder for analyses
OUTPUT_DIR = "../../failure_analysis_results"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_results(filepath):
    """
    Load model results

    Args:
        filepath: Path to file full_results.csv

    Returns:
        DataFrame with predictions
    """
    df = pd.read_csv(filepath)
    print(f"✓ Results loaded: {len(df)} examples")
    print(f"  Columns: {list(df.columns)}")
    return df


def generate_confusion_matrix(df, output_dir, model_name):
    """
    Generate and save the confusion matrix

    Args:
        df: DataFrame with actual_label and predicted_label
        output_dir: Output folder
        model_name: Name of the model

    Returns:
        dict with the statistics of the matrix
    """
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    # Extract labels
    y_true = df['actual_label'].values
    y_pred = df['predicted_label'].values

    # Unique labels
    labels = sorted(df['actual_label'].unique())
    print(f"\nClasses detected: {labels}")

    # Calculer the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Create visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Number of predictions'})
    plt.title(f'Matrice de Confusion - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    # Save the figure
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix sauvegardée: {confusion_matrix_path}")
    plt.close()

    # Calculer les statistiques
    # Normalized matrix (by row = by true class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Save text results
    results_path = os.path.join(output_dir, "confusion_matrix_stats.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"CONFUSION MATRIX - {model_name}\n")
        f.write("=" * 70 + "\n\n")

        f.write("RAW MATRIX (nombre d'examples):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'':15}")
        for label in labels:
            f.write(f"{label:>15}")
        f.write("\n")
        for i, label_true in enumerate(labels):
            f.write(f"{label_true:15}")
            for j in range(len(labels)):
                f.write(f"{cm[i, j]:>15}")
            f.write("\n")
        f.write("\n")

        f.write("NORMALIZED MATRIX (% by true class):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'':15}")
        for label in labels:
            f.write(f"{label:>15}")
        f.write("\n")
        for i, label_true in enumerate(labels):
            f.write(f"{label_true:15}")
            for j in range(len(labels)):
                f.write(f"{cm_normalized[i, j]:>14.1%}")
            f.write("\n")
        f.write("\n")

        # Rapport de classification
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-" * 70 + "\n")
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        f.write(report)
        f.write("\n")

        # Analyse des erreurs par classe
        f.write("ERROR ANALYSIS BY CLASS:\n")
        f.write("-" * 70 + "\n")
        for i, label in enumerate(labels):
            total = cm[i, :].sum()
            correct = cm[i, i]
            errors = total - correct
            accuracy = correct / total if total > 0 else 0

            f.write(f"\n{label}:\n")
            f.write(f"  Total d'examples: {total}\n")
            f.write(f"  Correctly classified: {correct} ({accuracy:.1%})\n")
            f.write(f"  Errors: {errors} ({(1-accuracy):.1%})\n")

            if errors > 0:
                f.write(f"  Main confusions:\n")
                for j, other_label in enumerate(labels):
                    if i != j and cm[i, j] > 0:
                        pct = cm[i, j] / total
                        f.write(f"    → {other_label}: {cm[i, j]} ({pct:.1%})\n")

    print(f"✓ Statistics saved: {results_path}")

    return {
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'labels': labels
    }


def analyze_group_f1_variance(df, output_dir, model_name):
    """
    Analyze the F1 variance between groups of stereotypes

    Args:
        df: DataFrame with actual_label, predicted_label, and group
        output_dir: Output folder
        model_name: Name of the model

    Returns:
        dict with F1 per group and variance statistics
    """
    print("\n" + "=" * 70)
    print("F1 VARIANCE BETWEEN GROUPS")
    print("=" * 70)

    # Check if the 'group' column exists
    if 'group' not in df.columns:
        print("⚠ 'group' column not found. Analysis by group impossible.")
        return None

    # Extract unique groups
    groups = df['group'].unique()
    groups = [g for g in groups if pd.notna(g)]  # Remove NaN
    print(f"\nGroups detected: {len(groups)}")
    print(f"  {groups}")

    # Calculate F1-macro par groupe
    group_f1_scores = {}
    group_stats = {}

    for group in groups:
        df_group = df[df['group'] == group]

        if len(df_group) == 0:
            continue

        y_true = df_group['actual_label'].values
        y_pred = df_group['predicted_label'].values

        # F1-macro for this group
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        group_f1_scores[group] = f1_macro

        # Group statistics
        correct = (y_true == y_pred).sum()
        total = len(df_group)
        accuracy = correct / total

        group_stats[group] = {
            'n_examples': total,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'n_correct': correct,
            'n_errors': total - correct
        }

    # Calculate variance statistics
    f1_values = list(group_f1_scores.values())
    f1_mean = np.mean(f1_values)
    f1_std = np.std(f1_values)
    f1_min = np.min(f1_values)
    f1_max = np.max(f1_values)
    f1_range = f1_max - f1_min

    print(f"\nAverage global macro-F1: {f1_mean:.4f}")
    print(f"Standard deviation: {f1_std:.4f}")
    print(f"Min: {f1_min:.4f} | Max: {f1_max:.4f} | Range: {f1_range:.4f}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Graph 1: F1 by group (bar plot)
    sorted_groups = sorted(group_f1_scores.items(), key=lambda x: x[1], reverse=True)
    group_names = [g[0] for g in sorted_groups]
    f1_scores = [g[1] for g in sorted_groups]

    colors = ['green' if f1 >= f1_mean else 'red' for f1 in f1_scores]

    ax1.barh(group_names, f1_scores, color=colors, alpha=0.7)
    ax1.axvline(f1_mean, color='blue', linestyle='--', linewidth=2, label=f'Average: {f1_mean:.3f}')
    ax1.set_xlabel('F1-macro', fontsize=11)
    ax1.set_ylabel('Stereotype group', fontsize=11)
    ax1.set_title(f'F1-macro per Group - {model_name}', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Graph 2: F1 Distribution
    ax2.hist(f1_scores, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(f1_mean, color='blue', linestyle='--', linewidth=2, label=f'Average: {f1_mean:.3f}')
    ax2.axvline(f1_mean - f1_std, color='red', linestyle=':', linewidth=1.5, label=f'±1 std: [{f1_mean-f1_std:.3f}, {f1_mean+f1_std:.3f}]')
    ax2.axvline(f1_mean + f1_std, color='red', linestyle=':', linewidth=1.5)
    ax2.set_xlabel('F1-macro', fontsize=11)
    ax2.set_ylabel('Number of groups', fontsize=11)
    ax2.set_title('F1 Distribution entre les Groups', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save the figure
    variance_plot_path = os.path.join(output_dir, "group_f1_variance.png")
    plt.savefig(variance_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Variance plot saved: {variance_plot_path}")
    plt.close()

    # Save text results
    variance_results_path = os.path.join(output_dir, "group_f1_variance.txt")
    with open(variance_results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"F1 VARIANCE ENTRE LES GROUPES - {model_name}\n")
        f.write("=" * 70 + "\n\n")

        f.write("GLOBAL STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of groups: {len(group_f1_scores)}\n")
        f.write(f"Average macro-F1: {f1_mean:.4f}\n")
        f.write(f"Standard deviation: {f1_std:.4f}\n")
        f.write(f"Coefficient of variation: {(f1_std/f1_mean)*100:.2f}%\n")
        f.write(f"Minimum F1: {f1_min:.4f}\n")
        f.write(f"Maximum F1: {f1_max:.4f}\n")
        f.write(f"Range (max - min): {f1_range:.4f}\n\n")

        f.write("MACRO-F1 BY GROUP (sorted from best to worst):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Group':<30} {'F1-macro':>10} {'Accuracy':>10} {'N':>8} {'Erreurs':>10}\n")
        f.write("-" * 70 + "\n")

        for group, f1 in sorted_groups:
            stats = group_stats[group]
            status = "✓" if f1 >= f1_mean else "✗"
            f.write(f"{group:<30} {f1:>10.4f} {stats['accuracy']:>9.1%} {stats['n_examples']:>8} "
                   f"{stats['n_errors']:>10} {status}\n")

        f.write("\n")
        f.write("VARIANCE ANALYSIS:\n")
        f.write("-" * 70 + "\n")

        # Groups au-dessus de la moyenne
        above_mean = [g for g, f1 in group_f1_scores.items() if f1 >= f1_mean]
        below_mean = [g for g, f1 in group_f1_scores.items() if f1 < f1_mean]

        f.write(f"\nGroups au-dessus de la moyenne ({len(above_mean)}):\n")
        for group in sorted(above_mean, key=lambda g: group_f1_scores[g], reverse=True):
            f.write(f"  • {group}: F1 = {group_f1_scores[group]:.4f} "
                   f"(+{group_f1_scores[group] - f1_mean:.4f})\n")

        f.write(f"\nGroups en-dessous de la moyenne ({len(below_mean)}):\n")
        for group in sorted(below_mean, key=lambda g: group_f1_scores[g]):
            f.write(f"  • {group}: F1 = {group_f1_scores[group]:.4f} "
                   f"({group_f1_scores[group] - f1_mean:.4f})\n")

        # Identify the 5 best et 5 worst groups
        f.write("\n" + "=" * 70 + "\n")
        f.write("TOP 5 BEST GROUPS:\n")
        f.write("-" * 70 + "\n")
        for i, (group, f1) in enumerate(sorted_groups[:5], 1):
            stats = group_stats[group]
            f.write(f"{i}. {group}\n")
            f.write(f"   F1-macro: {f1:.4f} | Accuracy: {stats['accuracy']:.1%} | "
                   f"Examples: {stats['n_examples']} | Errors: {stats['n_errors']}\n")

        f.write("\n")
        f.write("TOP 5 WORST GROUPS:\n")
        f.write("-" * 70 + "\n")
        for i, (group, f1) in enumerate(sorted_groups[-5:][::-1], 1):
            stats = group_stats[group]
            f.write(f"{i}. {group}\n")
            f.write(f"   F1-macro: {f1:.4f} | Accuracy: {stats['accuracy']:.1%} | "
                   f"Examples: {stats['n_examples']} | Errors: {stats['n_errors']}\n")

    print(f"✓ Variance results saved: {variance_results_path}")

    return {
        'group_f1_scores': group_f1_scores,
        'group_stats': group_stats,
        'mean': f1_mean,
        'std': f1_std,
        'min': f1_min,
        'max': f1_max,
        'range': f1_range
    }


def extract_failure_cases(df, output_dir, top_n=50):
    """
    Extract and save failure cases (incorrect predictions)

    Args:
        df: DataFrame with results
        output_dir: Output folder
        top_n: Number of failure cases to extract
    """
    print("\n" + "=" * 70)
    print("EXTRACTION OF FAILURE CASES")
    print("=" * 70)

    # Identify errors
    df_errors = df[df['actual_label'] != df['predicted_label']].copy()

    print(f"\nTotal errors: {len(df_errors)} / {len(df)} ({len(df_errors)/len(df)*100:.1f}%)")

    if len(df_errors) == 0:
        print("✓ No errors detected (perfect model!)")
        return

    # Sort by probability of prediction (errors with high confidence are more interesting)
    if 'predicted_probability' in df_errors.columns:
        df_errors = df_errors.sort_values('predicted_probability', ascending=False)

    # Save top N errors
    failure_cases_path = os.path.join(output_dir, "failure_cases.csv")
    df_errors.head(top_n).to_csv(failure_cases_path, index=False, encoding='utf-8')
    print(f"✓ Top {min(top_n, len(df_errors))} failure cases saved: {failure_cases_path}")

    # Create a summary text
    summary_path = os.path.join(output_dir, "failure_cases_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ANALYSIS OF FAILURE CASES\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total errors: {len(df_errors)} / {len(df)} ({len(df_errors)/len(df)*100:.1f}%)\n\n")

        # Types d'erreurs les plus fréquents
        f.write("ERROR TYPES MOST FREQUENT:\n")
        f.write("-" * 70 + "\n")
        error_types = df_errors.groupby(['actual_label', 'predicted_label']).size().sort_values(ascending=False)
        for (actual, predicted), count in error_types.head(10).items():
            pct = count / len(df_errors) * 100
            f.write(f"{actual} → {predicted}: {count} erreurs ({pct:.1f}%)\n")

        f.write("\n")
        f.write(f"EXAMPLES DE CAS D'ÉCHEC (Top {min(20, len(df_errors))}):\n")
        f.write("=" * 70 + "\n\n")

        for idx, (i, row) in enumerate(df_errors.head(20).iterrows(), 1):
            f.write(f"Case #{idx}:\n")
            f.write(f"  Text: {row['text'][:200]}...\n")
            f.write(f"  True label: {row['actual_label']}\n")
            f.write(f"  Predicted label: {row['predicted_label']}\n")
            if 'predicted_probability' in row:
                f.write(f"  Confidence: {row['predicted_probability']:.4f}\n")
            if 'group' in row and pd.notna(row['group']):
                f.write(f"  Group: {row['group']}\n")
            f.write("\n")

    print(f"✓ Summary of failure cases sauvegardé: {summary_path}")


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ANALYSIS OF FAILURE CASES - HEARTS-FR")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    # 1. Load results
    df = load_results(RESULTS_FILE)

    # 2. Générer the confusion matrix
    cm_results = generate_confusion_matrix(df, OUTPUT_DIR, MODEL_NAME)

    # 3. Analyze variance du F1 between groups
    variance_results = analyze_group_f1_variance(df, OUTPUT_DIR, MODEL_NAME)

    # 4. Extraire failure cases
    extract_failure_cases(df, OUTPUT_DIR, top_n=100)

    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"\nAll results have been saved dans: {OUTPUT_DIR}/")
    print("\nFiles generated:")
    print("  • confusion_matrix.png - Visualization of the confusion matrix")
    print("  • confusion_matrix_stats.txt - Detailed statistics")
    print("  • group_f1_variance.png - F1 variance between groups")
    print("  • group_f1_variance.txt - Detailed analysis par groupe")
    print("  • failure_cases.csv - Top 100 cas d'échec")
    print("  • failure_cases_summary.txt - Résumé des erreurs")
    print("\n" + "=" * 70)
