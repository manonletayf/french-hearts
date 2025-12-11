"""
Visualization Script for ALBERT on MGSD - HEARTS-FR

Generates visualizations for the ALBERT baseline model on the MGSD dataset.
Adapted from the main script for CamemBERT.

Usage:
    python visualizations_albert_mgsd.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from collections import Counter
import os
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chart configuration for poster (high quality)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 24
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 30

# Professional color palette
COLOR_PALETTE = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Purple
    'success': '#06A77D',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'neutral': '#6C757D',      # Gray
    'light': '#E9ECEF',        # Light gray
    'dark': '#212529'          # Black
}

# Data files - ALBERT on MGSD
ALBERT_RESULTS = "../Model/result_output_albertv2/mgsd_trained/mgsd/full_results.csv"

# MGSD Dataset
DATASET_FILE = "../../MGSD.csv"

# Output directory
OUTPUT_DIR = "visualizations_albert_mgsd"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_results(filepath, model_name):
    """Load results from a model"""
    df = pd.read_csv(filepath)
    print(f"✓ {model_name}: {len(df)} examples")
    return df

def calculate_f1_macro(df):
    """Calculate F1-macro of a model"""
    y_true = df['actual_label'].values
    y_pred = df['predicted_label'].values
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def save_figure(fig, filename, tight=True):
    """Save a figure with optimal parameters"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if tight:
        fig.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
    else:
        fig.savefig(filepath, dpi=300, facecolor='white')
    print(f"  ✓ {filename}")
    plt.close(fig)

# ============================================================================
# VISUALISATION 1: CONFUSION MATRIX
# ============================================================================

def viz_1_confusion_matrix():
    """
    Confusion matrix with high-quality heatmap
    """
    print("\n1. Generating Confusion Matrix...")

    df = load_results(ALBERT_RESULTS, "ALBERT-baseline")
    y_true = df['actual_label'].values
    y_pred = df['predicted_label'].values

    labels = sorted(df['actual_label'].unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap with annotations
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Number of Predictions'},
                linewidths=0.5, linecolor='gray',
                annot_kws={'size': 24, 'weight': 'bold'},
                ax=ax)

    ax.set_title('Confusion Matrix - ALBERT on MGSD',
                 fontsize=30, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=28, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=28, fontweight='bold')

    save_figure(fig, '01_confusion_matrix_albert_mgsd.png')

# ============================================================================
# VISUALISATION 2: F1 PER CLASS
# ============================================================================

def viz_2_f1_per_class():
    """
    Bar plot comparing F1 between Stereotype and Non-Stereotype
    """
    print("\n2. Generating F1 per Class...")

    df = load_results(ALBERT_RESULTS, "ALBERT-baseline")
    y_true = df['actual_label'].values
    y_pred = df['predicted_label'].values

    # Calculate F1 per class
    from sklearn.metrics import f1_score
    labels = sorted(df['actual_label'].unique())
    f1_scores = []

    for label in labels:
        y_true_bin = (y_true == label).astype(int)
        y_pred_bin = (y_pred == label).astype(int)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        f1_scores.append(f1)

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Map numeric labels to readable names
    label_names = []
    colors = []
    for label in labels:
        if isinstance(label, (int, np.integer)):
            # Numeric labels: 0 = non-stereotype, 1 = stereotype
            if label == 1:
                label_names.append('Stereotype')
                colors.append(COLOR_PALETTE['danger'])
            else:
                label_names.append('Non-Stereotype')
                colors.append(COLOR_PALETTE['success'])
        else:
            # Text labels
            label_names.append(str(label))
            if 'stereotype' in str(label).lower() and 'non' not in str(label).lower():
                colors.append(COLOR_PALETTE['danger'])
            else:
                colors.append(COLOR_PALETTE['success'])

    bars = ax.bar(label_names, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add values on the bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom',
                fontsize=24, fontweight='bold')

    ax.set_ylabel('F1 Score', fontsize=28, fontweight='bold')
    ax.set_xlabel('Class', fontsize=28, fontweight='bold')
    ax.set_title('F1 Score per Class - ALBERT on MGSD',
                 fontsize=30, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    save_figure(fig, '02_f1_per_class_albert_mgsd.png')

# ============================================================================
# VISUALISATION 3: ERROR DISTRIBUTION PIE CHART
# ============================================================================

def viz_3_error_distribution():
    """
    Pie chart showing FN vs FP
    """
    print("\n3. Generating Error Distribution Pie Chart...")

    df = load_results(ALBERT_RESULTS, "ALBERT-baseline")

    # Identify errors
    errors = df[df['actual_label'] != df['predicted_label']].copy()

    if len(errors) == 0:
        print("  ⚠ No errors found, skipping...")
        return

    # Count FN and FP (1 = stereotype, 0 = non-stereotype)
    fn_count = len(errors[(errors['actual_label'] == 1) &
                          (errors['predicted_label'] == 0)])
    fp_count = len(errors[(errors['actual_label'] == 0) &
                          (errors['predicted_label'] == 1)])

    # If no FN/FP, maybe labels are strings
    if fn_count == 0 and fp_count == 0:
        fn_count = len(errors[(errors['actual_label'] == 'stereotype') &
                              (errors['predicted_label'] != 'stereotype')])
        fp_count = len(errors[(errors['actual_label'] != 'stereotype') &
                              (errors['predicted_label'] == 'stereotype')])

    other_errors = len(errors) - fn_count - fp_count

    # If still no valid data, skip
    if fn_count == 0 and fp_count == 0:
        print("  ⚠ No FN/FP errors found, skipping...")
        return

    # Data
    sizes = [fn_count, fp_count] if other_errors == 0 else [fn_count, fp_count, other_errors]
    labels = [f'False Negatives\n(Missed Stereotypes)\n{fn_count} ({fn_count/len(errors)*100:.1f}%)',
              f'False Positives\n(Over-detection)\n{fp_count} ({fp_count/len(errors)*100:.1f}%)']
    if other_errors > 0:
        labels.append(f'Other Errors\n{other_errors} ({other_errors/len(errors)*100:.1f}%)')

    colors = [COLOR_PALETTE['danger'], COLOR_PALETTE['warning']] if other_errors == 0 else [COLOR_PALETTE['danger'], COLOR_PALETTE['warning'], COLOR_PALETTE['neutral']]
    explode = [0.05] * len(sizes)

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        explode=explode, shadow=True,
                                        textprops={'fontsize': 24, 'weight': 'bold'})

    # Style the percentages
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(28)
        autotext.set_weight('bold')

    ax.set_title('Error Distribution: FN vs FP - ALBERT on MGSD',
                 fontsize=30, fontweight='bold', pad=20)

    save_figure(fig, '03_error_distribution_albert_mgsd.png')

# ============================================================================
# VISUALISATION 4: ERROR WORDCLOUDS
# ============================================================================

def viz_4_error_wordclouds():
    """
    Wordclouds for FN and FP
    """
    print("\n4. Generating Error Wordclouds...")

    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  ⚠ wordcloud not installed, skipping...")
        return

    df = load_results(ALBERT_RESULTS, "ALBERT-baseline")

    # FN: Missed stereotypes (actual=1, predicted=0)
    fn_texts = df[(df['actual_label'] == 1) &
                  (df['predicted_label'] == 0)]['text'].tolist()

    # FP: False positives (actual=0, predicted=1)
    fp_texts = df[(df['actual_label'] == 0) &
                  (df['predicted_label'] == 1)]['text'].tolist()

    # If no results, try with text labels
    if not fn_texts:
        fn_texts = df[(df['actual_label'] == 'stereotype') &
                      (df['predicted_label'] != 'stereotype')]['text'].tolist()
    if not fp_texts:
        fp_texts = df[(df['actual_label'] != 'stereotype') &
                      (df['predicted_label'] == 'stereotype')]['text'].tolist()

    if not fn_texts and not fp_texts:
        print("  ⚠ Not enough error examples, skipping...")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Wordcloud FN
    if fn_texts:
        fn_text = ' '.join(fn_texts)
        wordcloud_fn = WordCloud(width=800, height=600,
                                 background_color='white',
                                 colormap='Reds',
                                 max_words=50,
                                 relative_scaling=0.5).generate(fn_text)

        ax1.imshow(wordcloud_fn, interpolation='bilinear')
        ax1.axis('off')
        ax1.set_title(f'False Negatives (Missed Stereotypes)\n{len(fn_texts)} examples',
                      fontsize=28, fontweight='bold', pad=10)
    else:
        ax1.text(0.5, 0.5, 'No False Negatives', ha='center', va='center',
                fontsize=28, transform=ax1.transAxes)
        ax1.axis('off')

    # Wordcloud FP
    if fp_texts:
        fp_text = ' '.join(fp_texts)
        wordcloud_fp = WordCloud(width=800, height=600,
                                 background_color='white',
                                 colormap='Oranges',
                                 max_words=50,
                                 relative_scaling=0.5).generate(fp_text)

        ax2.imshow(wordcloud_fp, interpolation='bilinear')
        ax2.axis('off')
        ax2.set_title(f'False Positives (Over-detection)\n{len(fp_texts)} examples',
                      fontsize=28, fontweight='bold', pad=10)
    else:
        ax2.text(0.5, 0.5, 'No False Positives', ha='center', va='center',
                fontsize=28, transform=ax2.transAxes)
        ax2.axis('off')

    plt.suptitle('Error Analysis - Word Frequency (ALBERT on MGSD)',
                 fontsize=32, fontweight='bold', y=0.98)

    save_figure(fig, '04_error_wordclouds_albert_mgsd.png', tight=False)

# ============================================================================
# VISUALISATION 5: DATASET OVERVIEW
# ============================================================================

def viz_5_dataset_overview():
    """
    Donut charts showing MGSD dataset composition (labels + groups)
    """
    print("\n5. Generating Dataset Overview...")

    try:
        df_dataset = pd.read_csv(DATASET_FILE)
    except:
        print("  ⚠ Dataset file not found, skipping...")
        return

    # Create figure with 2 subplots (labels + groups)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Subplot 1: Label distribution
    label_counts = df_dataset['label'].value_counts()
    labels_raw = label_counts.index.tolist()
    sizes_1 = label_counts.values.tolist()

    # Map labels to readable names/colors
    labels_1 = []
    colors_1 = []
    for l in labels_raw:
        if isinstance(l, (int, np.integer)):
            if l == 1:
                labels_1.append('Stereotype')
                colors_1.append(COLOR_PALETTE['danger'])
            else:
                labels_1.append('Non-Stereotype')
                colors_1.append(COLOR_PALETTE['success'])
        else:
            labels_1.append(str(l))
            if 'stereotype' in str(l).lower() and 'non' not in str(l).lower():
                colors_1.append(COLOR_PALETTE['danger'])
            elif 'neutral' in str(l).lower():
                colors_1.append(COLOR_PALETTE['neutral'])
            elif 'unrelated' in str(l).lower():
                colors_1.append(COLOR_PALETTE['warning'])
            else:
                colors_1.append(COLOR_PALETTE['warning'])

    wedges1, texts1, autotexts1 = ax1.pie(sizes_1, labels=labels_1, colors=colors_1,
                                            autopct='%1.1f%%', startangle=90,
                                            pctdistance=0.75, labeldistance=1.1,
                                            explode=[0.05]*len(sizes_1),
                                            textprops={'fontsize': 24, 'weight': 'bold'})

    # Donut hole
    centre_circle1 = plt.Circle((0, 0), 0.60, fc='white')
    ax1.add_artist(centre_circle1)

    # Style
    for autotext in autotexts1:
        autotext.set_color('white')
        autotext.set_fontsize(20)
        autotext.set_weight('bold')

    ax1.set_title(f'Label Distribution\n(Total: {len(df_dataset):,} examples)',
                  fontsize=24, fontweight='bold', pad=15)

    # Subplot 2: Group distribution (MGSD specific)
    if 'group' in df_dataset.columns:
        group_counts = df_dataset['group'].value_counts()
        labels_2 = group_counts.index.tolist()
        sizes_2 = group_counts.values.tolist()

        # Varied color palette
        colors_2 = plt.cm.Set3(np.linspace(0, 1, len(labels_2)))

        wedges2, texts2, autotexts2 = ax2.pie(sizes_2, labels=None, colors=colors_2,
                                                autopct='%1.1f%%', startangle=90,
                                                pctdistance=0.75, explode=[0.02]*len(sizes_2),
                                                textprops={'fontsize': 20, 'weight': 'bold'})

        # Donut hole
        centre_circle2 = plt.Circle((0, 0), 0.60, fc='white')
        ax2.add_artist(centre_circle2)

        # Percentage style
        for autotext in autotexts2:
            autotext.set_color('black')
            autotext.set_fontsize(20)
            autotext.set_weight('bold')

        # Legend on the right
        ax2.legend(wedges2, labels_2,
                   loc="center left",
                   bbox_to_anchor=(1, 0, 0.5, 1),
                   fontsize=20,
                   frameon=True,
                   fancybox=True,
                   shadow=True)

        ax2.set_title(f"Group Distribution\n(Total: {df_dataset['group'].nunique()} groups)",
                      fontsize=28, fontweight='bold', pad=15)
    else:
        ax2.text(0.5, 0.5, 'Group information\nnot available',
                ha='center', va='center', fontsize=28,
                transform=ax2.transAxes)
        ax2.axis('off')

    plt.suptitle('MGSD Dataset Overview',
                 fontsize=32, fontweight='bold', y=1.00)

    # Adjust layout to avoid overlap
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_figure(fig, '05_dataset_overview_albert_mgsd.png', tight=True)

# ============================================================================
# VISUALISATION 6: GROUP F1 SCORES
# ============================================================================

def viz_6_group_f1_scores():
    """
    Horizontal bar plot of F1 scores by group (CamemBERT CrowS style)
    """
    print("\n6. Generating Group F1 Scores...")

    df = load_results(ALBERT_RESULTS, "ALBERT-baseline")

    if 'group' not in df.columns:
        print("  ⚠ 'group' column missing in results, skipping...")
        return

    # Calculate F1 per group
    groups = df['group'].dropna().unique()
    group_f1 = {}

    for group in groups:
        df_group = df[df['group'] == group]
        y_true = df_group['actual_label'].values
        y_pred = df_group['predicted_label'].values
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        group_f1[group] = f1

    # Sort by F1
    sorted_groups = sorted(group_f1.items(), key=lambda x: x[1], reverse=True)
    groups_names = [g[0] for g in sorted_groups]
    f1_values = [g[1] for g in sorted_groups]

    # Average
    f1_mean = np.mean(f1_values)

    # Colors: green if > average, red otherwise
    colors = [COLOR_PALETTE['success'] if f1 >= f1_mean
              else COLOR_PALETTE['danger'] for f1 in f1_values]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    bars = ax.barh(groups_names, f1_values, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1)

    # Mean line
    ax.axvline(f1_mean, color='navy', linestyle='--', linewidth=2.5,
               label=f'Mean F1: {f1_mean:.3f}', zorder=5)

    # Add values
    for bar, score in zip(bars, f1_values):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{score:.3f}', ha='left', va='center',
                fontsize=24, fontweight='bold')

    ax.set_xlabel('F1-macro Score', fontsize=28, fontweight='bold')
    ax.set_ylabel('Stereotype Group', fontsize=28, fontweight='bold')
    ax.set_title('F1 Performance by Stereotype Group',
                 fontsize=30, fontweight='bold', pad=20)
    ax.set_xlim(0, max(f1_values) + 0.1)
    ax.legend(loc='lower right', fontsize=24)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    save_figure(fig, '06_group_f1_scores_albert_mgsd.png')

# ============================================================================
# PROGRAMME PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GENERATING VISUALIZATIONS - ALBERT ON MGSD")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("\nGenerating 6 visualizations...\n")

    try:
        viz_1_confusion_matrix()
        viz_2_f1_per_class()
        viz_3_error_distribution()
        viz_4_error_wordclouds()
        viz_5_dataset_overview()
        viz_6_group_f1_scores()

        print("\n" + "=" * 80)
        print("✓ ALL VISUALIZATIONS HAVE BEEN GENERATED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nFiles saved in: {OUTPUT_DIR}/")
        print("\nVisualizations generated:")
        print("  01. Confusion Matrix (ALBERT on MGSD)")
        print("  02. F1 per Class")
        print("  03. Error Distribution Pie Chart")
        print("  04. Error Wordclouds")
        print("  05. MGSD Dataset Overview")
        print("  06. Group F1 Scores (ALBERT on MGSD)")
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
