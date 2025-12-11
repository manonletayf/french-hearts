"""
Baseline Logistic Regression + TF-IDF for textual stereotype detection
Version simplifiée pour CrowS-Pairs FR
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support, balanced_accuracy_score
import joblib


def data_loader(csv_file_path, labelling_criteria, dataset_name, sample_size, num_examples):
    """
    Charge et prépare les données depuis un fichier CSV.

    Args:
        csv_file_path: Path to file CSV
        labelling_criteria: Critère de labellisation (ex: 'stereotype')
        dataset_name: Name of dataset
        sample_size: Taille de l'échantillon à utiliser
        num_examples: Nombre d'examples à afficher

    Returns:
        train_data, test_data: DataFrames d'entraînement et de test
    """
    combined_data = pd.read_csv(csv_file_path, usecols=['text', 'label', 'group'])

    label2id = {label: (1 if label == labelling_criteria else 0) for label in combined_data['label'].unique()}
    combined_data['label'] = combined_data['label'].map(label2id)

    combined_data['data_name'] = dataset_name

    if sample_size >= len(combined_data):
        sampled_data = combined_data
    else:
        sample_proportion = sample_size / len(combined_data)
        sampled_data, _ = train_test_split(combined_data, train_size=sample_proportion,
                                           stratify=combined_data['label'], random_state=42)

    train_data, test_data = train_test_split(sampled_data, test_size=0.2, random_state=42,
                                             stratify=sampled_data['label'])

    print("First few examples from the training data:")
    print(train_data.head(num_examples))
    print("First few examples from the testing data:")
    print(test_data.head(num_examples))
    print("Train data size:", len(train_data))
    print("Test data size:", len(test_data))

    return train_data, test_data


def train_model(train_data, model_output_base_dir, dataset_name, seed):
    """
    Entraîne un modèle Logistic Regression avec features TF-IDF.

    Args:
        train_data: DataFrame d'entraînement
        model_output_base_dir: Dossier de base pour sauvegarder le modèle
        dataset_name: Name of dataset
        seed: Seed for reproducibility

    Returns:
        model_output_dir: Chemin du dossier contenant le modèle sauvegardé
    """
    np.random.seed(seed)
    num_labels = len(np.unique(train_data['label']))
    print(f"Number of unique labels: {num_labels}")

    # Créer le vectorizer TF-IDF avec les paramètres spécifiés
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    X = vectorizer.fit_transform(train_data['text'])
    y = train_data['label']

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # Entraîner le modèle Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed
    )
    model.fit(X_train, y_train)

    # Évaluer sur le validation set
    y_pred = model.predict(X_val)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro')

    print(f"Validation metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Save the model and the vectorizer
    model_output_dir = os.path.join(model_output_base_dir, dataset_name)
    os.makedirs(model_output_dir, exist_ok=True)

    model_path = os.path.join(model_output_dir, 'model.pkl')
    vectorizer_path = os.path.join(model_output_dir, 'vectorizer.pkl')

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Model and vectorizer saved to {model_output_dir}")

    return model_output_dir


def evaluate_model(test_data, model_output_dir, result_output_base_dir, dataset_name, seed):
    """
    Évalue le modèle sur le test set et sauvegarde les résultats.

    Args:
        test_data: DataFrame de test
        model_output_dir: Dossier contenant le modèle sauvegardé
        result_output_base_dir: Dossier de base pour sauvegarder les résultats
        dataset_name: Name of dataset
        seed: Seed for reproducibility

    Returns:
        df_report: DataFrame containing the report de classification
    """
    np.random.seed(seed)
    num_labels = len(test_data['label'].unique())
    print(f"Number of unique labels: {num_labels}")

    # Charger le modèle et le vectorizer
    model_path = os.path.join(model_output_dir, 'model.pkl')
    vectorizer_path = os.path.join(model_output_dir, 'vectorizer.pkl')

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Transformer les données de test
    X_test = vectorizer.transform(test_data['text'])

    # Faire des prédictions
    y_pred_probs = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    pred_labels = y_pred.tolist()
    pred_probs = y_pred_probs.max(axis=1).tolist()
    y_true = test_data['label'].tolist()

    # Créer le DataFrame with detailed results
    results_df = pd.DataFrame({
        'text': test_data['text'],
        'predicted_label': pred_labels,
        'predicted_probability': pred_probs,
        'actual_label': y_true,
        'group': test_data['group'],
        'dataset_name': test_data['data_name']
    })

    # Create output folder
    result_output_dir = os.path.join(result_output_base_dir, dataset_name)
    os.makedirs(result_output_dir, exist_ok=True)

    # Save complete results
    results_file_path = os.path.join(result_output_dir, "full_results.csv")
    results_df.to_csv(results_file_path, index=False)

    # Générer et sauvegarder le rapport global classification
    report = classification_report(y_true, pred_labels, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    result_file_path = os.path.join(result_output_dir, "classification_report.csv")
    df_report.to_csv(result_file_path)

    print(f"\nClassification report saved to {result_file_path}")
    print(df_report)

    # Calculate metrics par groupe
    groups = results_df['group'].unique()
    group_metrics = []

    for group in groups:
        group_data = results_df[results_df['group'] == group]
        group_y_true = group_data['actual_label'].tolist()
        group_pred_labels = group_data['predicted_label'].tolist()

        precision, recall, f1, _ = precision_recall_fscore_support(
            group_y_true, group_pred_labels, average='macro', zero_division=0
        )
        balanced_acc = balanced_accuracy_score(group_y_true, group_pred_labels)

        group_metrics.append({
            'group': group,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'balanced_accuracy': balanced_acc,
            'support': len(group_data)
        })

    # Save metrics par groupe
    df_group_metrics = pd.DataFrame(group_metrics)
    group_metrics_file_path = os.path.join(result_output_dir, "metrics_by_group.csv")
    df_group_metrics.to_csv(group_metrics_file_path, index=False)

    print(f"\nMetrics by group saved to {group_metrics_file_path}")
    print(df_group_metrics)

    return df_report


if __name__ == "__main__":
    # Load data CrowS-Pairs FR
    print("Loading CrowS-Pairs FR data...")
    train_data_fr, test_data_fr = data_loader(
        csv_file_path='../../data/crows_pairs_fr_final.csv',
        labelling_criteria='stereotype',
        dataset_name='CrowS_FR',
        sample_size=1000000,
        num_examples=5
    )

    # Entraîner le modèle Logistic Regression + TF-IDF
    print("\n" + "="*60)
    print("Training Logistic Regression + TF-IDF model...")
    print("="*60)
    model_output_dir = train_model(
        train_data_fr,
        model_output_base_dir='../../model_output_LR_tfidf',
        dataset_name='crows_fr_trained',
        seed=42
    )

    # Évaluer le modèle
    print("\n" + "="*60)
    print("Evaluating model on test set...")
    print("="*60)
    df_report = evaluate_model(
        test_data_fr,
        model_output_dir='../../model_output_LR_tfidf/crows_fr_trained',
        result_output_base_dir='../../result_output_LR_tfidf',
        dataset_name='crows_fr',
        seed=42
    )

    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
