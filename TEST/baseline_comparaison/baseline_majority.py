"""
Baseline with majority class for textual stereotype detection
Uses DummyClassifier from scikit-learn with 'most_frequent' strategy
to establish a simple baseline to compare with CamemBERT.
"""

import os
import sys
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, balanced_accuracy_score

# Add parent folder to path to import data_loader
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

# Import data_loader function from the CamemBERT script in the parent folder
import importlib.util
spec = importlib.util.spec_from_file_location("baseline_camembert_crows_fr",
                                               os.path.join(parent_dir, "baseline_camembert_crows_fr.py"))
baseline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_module)
data_loader = baseline_module.data_loader


def train_and_evaluate_majority_baseline(train_data, test_data, result_output_dir, dataset_name):
    """
    Train a majority class model (DummyClassifier) and evaluate its performance.

    Args:
        train_data: DataFrame with columns ['text', 'label', 'group', 'data_name']
        test_data: DataFrame with columns ['text', 'label', 'group', 'data_name']
        result_output_dir: Path of folder to save the results
        dataset_name: Name of dataset (used to name the subfolder)

    Returns:
        df_report: DataFrame containing the classification report
    """

    # Create output folder if it does not exist
    output_path = os.path.join(result_output_dir, dataset_name)
    os.makedirs(output_path, exist_ok=True)

    # Separate features and labels
    # Note: For DummyClassifier, the features (text) are not used,
    # but we keep them for consistency with the code structure
    X_train = train_data['text'].values
    y_train = train_data['label'].values

    X_test = test_data['text'].values
    y_test = test_data['label'].values

    print(f"\n{'='*60}")
    print(f"Majority Class Baseline - {dataset_name}")
    print(f"{'='*60}")
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    print(f"Distribution of labels in train set:")
    print(train_data['label'].value_counts())
    print(f"\nDistribution of labels in test set:")
    print(test_data['label'].value_counts())

    # Créer et entraîner le DummyClassifier with 'most_frequent' strategy
    print(f"\n{'='*60}")
    print("Training the model de classe majoritaire...")
    print(f"{'='*60}")

    dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy_clf.fit(X_train, y_train)

    print(f"Predicted majority class: {dummy_clf.classes_[dummy_clf.class_prior_.argmax()]}")
    print(f"Proportion of majority class: {dummy_clf.class_prior_.max():.2%}")

    # Faire des prédictions sur le test set
    print(f"\n{'='*60}")
    print("Evaluation on test set...")
    print(f"{'='*60}")

    y_pred = dummy_clf.predict(X_test)

    # Create a DataFrame with detailed results
    results_df = pd.DataFrame({
        'text': test_data['text'].values,
        'predicted_label': y_pred,
        'actual_label': y_test,
        'group': test_data['group'].values,
        'dataset_name': test_data['data_name'].values
    })

    # Save complete results
    full_results_path = os.path.join(output_path, "full_results.csv")
    results_df.to_csv(full_results_path, index=False)
    print(f"\nComplete results saved: {full_results_path}")

    # Generate the report global classification
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()

    # Save the report de classification
    report_path = os.path.join(output_path, "classification_report.csv")
    df_report.to_csv(report_path)
    print(f"Classification report saved: {report_path}")

    # Display the report de classification
    print(f"\n{'='*60}")
    print("Rapport global classification:")
    print(f"{'='*60}")
    print(df_report)

    # Calculate metrics macro
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print("Global metrics (macro):")
    print(f"{'='*60}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    # Calculate metrics par groupe
    print(f"\n{'='*60}")
    print("Calculating metrics by group...")
    print(f"{'='*60}")

    groups = results_df['group'].unique()
    group_metrics = []

    for group in groups:
        group_data = results_df[results_df['group'] == group]
        group_y_true = group_data['actual_label'].values
        group_y_pred = group_data['predicted_label'].values

        # Calculate metrics for this group
        group_precision, group_recall, group_f1, _ = precision_recall_fscore_support(
            group_y_true, group_y_pred, average='macro', zero_division=0
        )
        group_balanced_acc = balanced_accuracy_score(group_y_true, group_y_pred)

        group_metrics.append({
            'group': group,
            'precision': group_precision,
            'recall': group_recall,
            'f1': group_f1,
            'balanced_accuracy': group_balanced_acc,
            'support': len(group_data)
        })

    # Create a DataFrame avec les métriques par groupe
    df_group_metrics = pd.DataFrame(group_metrics)

    # Save metrics par groupe
    group_metrics_path = os.path.join(output_path, "metrics_by_group.csv")
    df_group_metrics.to_csv(group_metrics_path, index=False)
    print(f"Metrics by group saved: {group_metrics_path}")

    # Display metrics par groupe
    print(f"\n{'='*60}")
    print("Metrics by group:")
    print(f"{'='*60}")
    print(df_group_metrics.to_string(index=False))

    print(f"\n{'='*60}")
    print("Evaluation completed!")
    print(f"{'='*60}\n")

    return df_report


if __name__ == "__main__":
    # Load data French with the same parameters as for CamemBERT
    print("Loading CrowS-Pairs FR data...")
    train_data_fr, test_data_fr = data_loader(
        csv_file_path='../../data/crows_pairs_fr_final.csv',
        labelling_criteria='stereotype',
        dataset_name='CrowS_FR',
        sample_size=1000000,
        num_examples=5
    )

    # Entraîner et évaluer la baseline de classe majoritaire
    df_report = train_and_evaluate_majority_baseline(
        train_data=train_data_fr,
        test_data=test_data_fr,
        result_output_dir='../../result_output_baselines',
        dataset_name='majority_class'
    )
