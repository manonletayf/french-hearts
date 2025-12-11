# Define helper function for loading data with ablation options
import pandas as pd
from sklearn.model_selection import train_test_split

def data_loader(csv_file_path, labelling_criteria, dataset_name, sample_size, num_examples,
                drop_unrelated=False, drop_neutral=False, three_class=False):
    """
    Load data with ablation options

    Args:
        csv_file_path: Path to the CSV
        labelling_criteria: Labeling criteria
        dataset_name: Name of dataset
        sample_size: Size of sample
        num_examples: Number of examples to display
        drop_unrelated: If True, removes 'unrelated' rows
        drop_neutral: If True, removes 'neutral' rows
        three_class: If True, uses 3 classes instead of 2
    """
    combined_data = pd.read_csv(csv_file_path, usecols=['text', 'label', 'group'])

    # ABLATION: Data filtering
    if drop_unrelated:
        combined_data = combined_data[combined_data['label'] != 'unrelated']
        print(f"Dropped 'unrelated' labels. Remaining samples: {len(combined_data)}")

    if drop_neutral:
        combined_data = combined_data[combined_data['label'] != 'neutral']
        print(f"Dropped 'neutral' labels. Remaining samples: {len(combined_data)}")

    # ABLATION: Label mapping
    if three_class:
        # 3-class classification
        label2id = {
            'stereotype': 0,
            'neutral': 1,
            'unrelated': 2
        }
        print("Using 3-class classification: stereotype=0, neutral=1, unrelated=2")
    else:
        # Binary classification (default)
        label2id = {label: (1 if label == labelling_criteria else 0)
                    for label in combined_data['label'].unique()}
        print("Using binary classification: stereotype=1, others=0")

    combined_data['label'] = combined_data['label'].map(label2id)
    combined_data['data_name'] = dataset_name

    # Sampling
    if sample_size >= len(combined_data):
        sampled_data = combined_data
    else:
        sample_proportion = sample_size / len(combined_data)
        sampled_data, _ = train_test_split(combined_data, train_size=sample_proportion,
                                           stratify=combined_data['label'], random_state=42)

    # Split train/test
    train_data, test_data = train_test_split(sampled_data, test_size=0.2, random_state=42,
                                             stratify=sampled_data['label'])

    print("\nFirst few examples from the training data:")
    print(train_data.head(num_examples))
    print("\nFirst few examples from the testing data:")
    print(test_data.head(num_examples))
    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    print(f"Label distribution in train:\n{train_data['label'].value_counts()}")

    return train_data, test_data


# Define function for fine tuning language model
import os
import numpy as np
import logging
from sklearn.metrics import classification_report, precision_recall_fscore_support, balanced_accuracy_score
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, pipeline

# Enable progress bar and set up logging
os.environ["HUGGINGFACE_TRAINER_ENABLE_PROGRESS_BAR"] = "1"
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)

def train_model(train_data, model_path, batch_size, epoch, learning_rate, model_output_base_dir,
                dataset_name, seed, max_length=512):
    """
    Train the model with ablation option on max_length

    Args:
        max_length: Maximum length for tokenization (ABLATION)
    """
    np.random.seed(seed)
    num_labels = len(train_data['label'].unique())
    print(f"Number of unique labels: {num_labels}")
    print(f"Max sequence length: {max_length}")

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels,
                                                               ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ABLATION: max_length paramétrable
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=max_length)

    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    tokenized_train = Dataset.from_pandas(train_data).map(tokenize_function, batched=True).map(
        lambda examples: {'labels': examples['label']})
    print("Sample tokenized input from train:", tokenized_train[0])
    tokenized_val = Dataset.from_pandas(val_data).map(tokenize_function, batched=True).map(
        lambda examples: {'labels': examples['label']})
    print("Sample tokenized input from validation:", tokenized_val[0])

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        balanced_acc = balanced_accuracy_score(labels, predictions)
        return {"precision": precision, "recall": recall, "f1": f1, "balanced accuracy": balanced_acc}

    model_output_dir = os.path.join(model_output_base_dir, dataset_name)
    os.makedirs(model_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_output_dir, num_train_epochs=epoch, evaluation_strategy="epoch",
        learning_rate=learning_rate, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, weight_decay=0.01,
        save_strategy="epoch", load_best_model_at_end=True, save_total_limit=1)

    trainer = Trainer(
        model=model, args=training_args, tokenizer=tokenizer, train_dataset=tokenized_train,
        eval_dataset=tokenized_val, compute_metrics=compute_metrics)

    trainer.train()
    trainer.save_model(model_output_dir)

    return model_output_dir


# Define function for evaluating the model
def evaluate_model(test_data, model_output_dir, result_output_base_dir, dataset_name, seed, max_length=512):
    """
    Évalue le modèle avec option d'ablation sur max_length

    Args:
        max_length: Longueur maximale pour la tokenisation (ABLATION)
    """
    np.random.seed(seed)
    num_labels = len(test_data['label'].unique())
    print(f"Number of unique labels: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(model_output_dir, num_labels=num_labels,
                                                               ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(model_output_dir)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=max_length)

    tokenized_test = Dataset.from_pandas(test_data).map(tokenize_function, batched=True).map(
        lambda examples: {'labels': examples['label']})
    print("Sample tokenized input from test:", tokenized_test[0])

    result_output_dir = os.path.join(result_output_base_dir, dataset_name)
    os.makedirs(result_output_dir, exist_ok=True)

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

    predictions = pipe(test_data['text'].to_list(), return_all_scores=True)
    pred_labels = [int(max(pred, key=lambda x: x['score'])['label'].split('_')[-1]) for pred in predictions]
    pred_probs = [max(pred, key=lambda x: x['score'])['score'] for pred in predictions]
    y_true = test_data['label'].tolist()

    results_df = pd.DataFrame({
        'text': test_data['text'],
        'predicted_label': pred_labels,
        'predicted_probability': pred_probs,
        'actual_label': y_true,
        'group': test_data['group'],
        'dataset_name': test_data['data_name']
    })

    results_file_path = os.path.join(result_output_dir, "full_results.csv")
    results_df.to_csv(results_file_path, index=False)

    report = classification_report(y_true, pred_labels, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    result_file_path = os.path.join(result_output_dir, "classification_report.csv")
    df_report.to_csv(result_file_path)

    # Calculate metrics per group
    groups = results_df['group'].unique()
    group_metrics = []

    for group in groups:
        group_data = results_df[results_df['group'] == group]
        group_y_true = group_data['actual_label'].tolist()
        group_pred_labels = group_data['predicted_label'].tolist()

        group_report = classification_report(group_y_true, group_pred_labels, output_dict=True, zero_division=0)
        precision, recall, f1, _ = precision_recall_fscore_support(group_y_true, group_pred_labels,
                                                                   average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(group_y_true, group_pred_labels)

        group_metrics.append({
            'group': group,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'balanced_accuracy': balanced_acc,
            'support': len(group_data)
        })

    df_group_metrics = pd.DataFrame(group_metrics)
    group_metrics_file_path = os.path.join(result_output_dir, "metrics_by_group.csv")
    df_group_metrics.to_csv(group_metrics_file_path, index=False)

    print("\nMetrics by group:")
    print(df_group_metrics)

    return df_report


if __name__ == "__main__":
    print("="*80)
    print("EXAMPLES D'ABLATIONS")
    print("="*80)

    # BASELINE (sans ablation)
    print("\n### BASELINE (pas d'ablation) ###")
    train_data_baseline, test_data_baseline = data_loader(
        csv_file_path='../../data/crows_pairs_fr_final.csv',
        labelling_criteria='stereotype',
        dataset_name='CrowS_FR',
        sample_size=1000000,
        num_examples=3
    )
    # train_model(train_data_baseline, model_path='camembert-base', batch_size=32, epoch=4,
    #             learning_rate=2e-5, model_output_base_dir='../../model_output_ablation_baseline',
    #             dataset_name='baseline', seed=42, max_length=512)

    # ABLATION A: drop_unrelated=True
    print("\n" + "="*80)
    print("### ABLATION A: drop_unrelated=True ###")
    train_data_a, test_data_a = data_loader(
        csv_file_path='../../data/crows_pairs_fr_final.csv',
        labelling_criteria='stereotype',
        dataset_name='CrowS_FR',
        sample_size=1000000,
        num_examples=3,
        drop_unrelated=True
    )
    # train_model(train_data_a, model_path='camembert-base', batch_size=32, epoch=4,
    #             learning_rate=2e-5, model_output_base_dir='../../model_output_ablation_drop_unrelated',
    #             dataset_name='drop_unrelated', seed=42)

    # ABLATION B: drop_neutral=True
    print("\n" + "="*80)
    print("### ABLATION B: drop_neutral=True ###")
    train_data_b, test_data_b = data_loader(
        csv_file_path='../../data/crows_pairs_fr_final.csv',
        labelling_criteria='stereotype',
        dataset_name='CrowS_FR',
        sample_size=1000000,
        num_examples=3,
        drop_neutral=True
    )
    # train_model(train_data_b, model_path='camembert-base', batch_size=32, epoch=4,
    #             learning_rate=2e-5, model_output_base_dir='../../model_output_ablation_drop_neutral',
    #             dataset_name='drop_neutral', seed=42)

    # ABLATION C: three_class=True
    print("\n" + "="*80)
    print("### ABLATION C: three_class=True ###")
    train_data_c, test_data_c = data_loader(
        csv_file_path='../../data/crows_pairs_fr_final.csv',
        labelling_criteria='stereotype',
        dataset_name='CrowS_FR',
        sample_size=1000000,
        num_examples=3,
        three_class=True
    )
    # train_model(train_data_c, model_path='camembert-base', batch_size=32, epoch=4,
    #             learning_rate=2e-5, model_output_base_dir='../../model_output_ablation_3class',
    #             dataset_name='3class', seed=42)

    # ABLATION D: max_length=128
    print("\n" + "="*80)
    print("### ABLATION D: max_length=128 ###")
    # train_model(train_data_baseline, model_path='camembert-base', batch_size=32, epoch=4,
    #             learning_rate=2e-5, model_output_base_dir='../../model_output_ablation_maxlen128',
    #             dataset_name='maxlen128', seed=42, max_length=128)

    # ABLATION E: learning_rate=5e-5
    print("\n" + "="*80)
    print("### ABLATION E: learning_rate=5e-5 ###")
    # train_model(train_data_baseline, model_path='camembert-base', batch_size=32, epoch=4,
    #             learning_rate=5e-5, model_output_base_dir='../../model_output_ablation_lr5e5',
    #             dataset_name='lr5e5', seed=42)

    # ABLATION F: epoch=8
    print("\n" + "="*80)
    print("### ABLATION F: epoch=8 ###")
    # train_model(train_data_baseline, model_path='camembert-base', batch_size=32, epoch=8,
    #             learning_rate=2e-5, model_output_base_dir='../../model_output_ablation_epoch8',
    #             dataset_name='epoch8', seed=42)

    print("\n" + "="*80)
    print("Pour lancer une ablation, décommente les lignes correspondantes ci-dessus")
    print("="*80)
