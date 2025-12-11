"""
Interpretability Analysis with LIME - HEARTS-FR
Adapted for CamemBERT on CrowS-Pairs-FR

LIME (Local Interpretable Model-agnostic Explanations) explains predictions
by creating a local linear model around each example.

Usage:
    python explainability_lime.py
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and tokenizer
MODEL_PATH = "model_output_camembert_final/crows_fr_trained_final"
MODEL_NAME = "camembert-base"

# Dataset
DATASET_FILE = "data/crows_pairs_fr_final.csv"

# Output directory
OUTPUT_DIR = "explainability_results/lime"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LIME parameters
N_SAMPLES = 50  # Number of examples to analyze
NUM_FEATURES = 10  # Number of important words to display
NUM_SAMPLES_LIME = 1000  # Number of samples for LIME
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer():
    """
    Load the fine-tuned CamemBERT model and tokenizer
    """
    print("\nLoading model and tokenizer...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    print(f"✓ Model loaded: {MODEL_NAME}")
    print(f"✓ Classes: {model.config.num_labels}")

    # Create a pipeline to simplify predictions
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1,
        return_all_scores=True,
        max_length=MAX_LENGTH,
        truncation=True
    )

    return model, tokenizer, pipe


def load_dataset(filepath, n_samples=None):
    """
    Load the CrowS-Pairs-FR dataset
    """
    print(f"\nLoading dataset: {filepath}")
    df = pd.read_csv(filepath)

    if n_samples:
        # Stratified sampling by label
        df_sample = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), n_samples // df['label'].nunique()),
                              random_state=42)
        )
        df = df_sample.reset_index(drop=True)

    print(f"✓ Dataset loaded: {len(df)} examples")
    return df

# ============================================================================
# PREDICTION FUNCTION FOR LIME
# ============================================================================

class PredictorWrapper:
    """
    Wrapper for predictions compatible with LIME
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = MAX_LENGTH

    def predict_proba(self, texts):
        """
        Probability prediction for LIME

        Args:
            texts: List of texts

        Returns:
            Array of probabilities (n_samples, n_classes)
        """
        # Tokenization
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Prediction
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in encodings.items()}
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        return probs.cpu().numpy()


# ============================================================================
# LIME ANALYSIS
# ============================================================================

def create_lime_explainer(class_names):
    """
    Create a LIME explainer for text classification

    Args:
        class_names: Class names

    Returns:
        LimeTextExplainer
    """
    print("\nCreating LIME explainer...")

    explainer = LimeTextExplainer(
        class_names=class_names,
        split_expression=' ',  # Split by spaces (tokens)
        random_state=42
    )

    print("✓ LIME explainer created")
    return explainer


def analyze_with_lime(explainer, predictor, texts, labels, output_dir,
                      top_k=10, num_features=10, num_samples=1000):
    """
    Analyze examples with LIME

    Args:
        explainer: LimeTextExplainer
        predictor: PredictorWrapper
        texts: List of texts
        labels: Real labels
        output_dir: Output directory
        top_k: Number of examples to analyze in detail
        num_features: Number of features to display
        num_samples: Number of samples for LIME

    Returns:
        List of explanations
    """
    print(f"\nLIME analysis of {min(top_k, len(texts))} examples...")

    results = []

    for idx in tqdm(range(min(top_k, len(texts))), desc="LIME analysis"):
        text = texts[idx]
        label = labels[idx]

        try:
            # Get LIME explanation
            exp = explainer.explain_instance(
                text,
                predictor.predict_proba,
                num_features=num_features,
                num_samples=num_samples,
                labels=(1,)  # Explain the "stereotype" class
            )

            # Save explanation
            results.append({
                'text': text,
                'label': label,
                'explanation': exp,
                'idx': idx
            })

            # HTML visualization
            html_file = os.path.join(output_dir, f"lime_example_{idx}.html")
            exp.save_to_file(html_file)

            # Matplotlib visualization
            fig = exp.as_pyplot_figure(label=1)
            fig.set_size_inches(12, 6)
            plt.title(f'LIME Explanation - Example {idx}\nLabel: {label}',
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"lime_example_{idx}.png"),
                       dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"  ⚠ Error on example {idx}: {str(e)}")
            continue

    print(f"✓ {len(results)} examples analyzed")
    return results


def extract_important_words(results, output_dir):
    """
    Extract the most important words according to LIME

    Args:
        results: List of LIME results
        output_dir: Output directory
    """
    print("\nExtracting important words...")

    # Dictionaries to store weights
    positive_words = {}  # Words contributing to "stereotype"
    negative_words = {}  # Words contributing to "non-stereotype"

    for result in results:
        exp = result['explanation']

        # Get weights for class 1 (stereotype)
        word_weights = exp.as_list(label=1)

        for word, weight in word_weights:
            word = word.strip().lower()

            if weight > 0:
                positive_words[word] = positive_words.get(word, 0) + weight
            else:
                negative_words[word] = negative_words.get(word, 0) + abs(weight)

    # Sort by importance
    top_positive = sorted(positive_words.items(), key=lambda x: x[1], reverse=True)[:20]
    top_negative = sorted(negative_words.items(), key=lambda x: x[1], reverse=True)[:20]

    # Save
    summary_file = os.path.join(output_dir, "important_words.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("IMPORTANT WORDS - LIME ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("TOP 20 WORDS CONTRIBUTING TO 'STEREOTYPE':\n")
        f.write("-" * 80 + "\n")
        for word, weight in top_positive:
            f.write(f"  {word:30} | Weight: {weight:>8.4f}\n")

        f.write("\n\n")
        f.write("TOP 20 WORDS CONTRIBUTING TO 'NON-STEREOTYPE':\n")
        f.write("-" * 80 + "\n")
        for word, weight in top_negative:
            f.write(f"  {word:30} | Weight: {weight:>8.4f}\n")

    print(f"✓ Important words saved: {summary_file}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Positive words (stereotype)
    words_pos = [w for w, _ in top_positive[:10]]
    weights_pos = [wt for _, wt in top_positive[:10]]

    ax1.barh(words_pos, weights_pos, color='#C73E1D', alpha=0.8)
    ax1.set_xlabel('LIME Weight', fontsize=12, fontweight='bold')
    ax1.set_title('Words Contributing to "Stereotype"',
                  fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Negative words (non-stereotype)
    words_neg = [w for w, _ in top_negative[:10]]
    weights_neg = [wt for _, wt in top_negative[:10]]

    ax2.barh(words_neg, weights_neg, color='#06A77D', alpha=0.8)
    ax2.set_xlabel('LIME Weight', fontsize=12, fontweight='bold')
    ax2.set_title('Words Contributing to "Non-Stereotype"',
                  fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.suptitle('Most Important Words - LIME Analysis',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "important_words.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Visualization saved: important_words.png")


def generate_lime_summary(results, output_dir):
    """
    Generate a summary of LIME analyses

    Args:
        results: List of LIME results
        output_dir: Output directory
    """
    print("\nGenerating LIME summary...")

    summary_file = os.path.join(output_dir, "lime_summary.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LIME ANALYSIS SUMMARY - HEARTS-FR\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Number of examples analyzed: {len(results)}\n\n")

        f.write("EXPLANATION DETAILS:\n")
        f.write("-" * 80 + "\n\n")

        for result in results:
            idx = result['idx']
            text = result['text']
            label = result['label']
            exp = result['explanation']

            f.write(f"Example {idx + 1}:\n")
            f.write(f"  Text: {text[:100]}...\n")
            f.write(f"  Real label: {label}\n")

            # Top 5 most important words
            word_weights = exp.as_list(label=1)[:5]
            f.write(f"  Top 5 important words:\n")
            for word, weight in word_weights:
                direction = "→ Stereotype" if weight > 0 else "→ Non-Stereotype"
                f.write(f"    • {word}: {weight:+.4f} {direction}\n")

            f.write(f"  Visualizations:\n")
            f.write(f"    - HTML: lime_example_{idx}.html\n")
            f.write(f"    - PNG: lime_example_{idx}.png\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")
        f.write("LIME creates a local linear model to explain each prediction:\n")
        f.write("- POSITIVE weight (+): The word contributes to predicting 'stereotype'\n")
        f.write("- NEGATIVE weight (-): The word contributes to predicting 'non-stereotype'\n")
        f.write("- Weight magnitude: Importance of the contribution\n\n")

    print(f"✓ Summary saved: {summary_file}")


def analyze_by_group(df, explainer, predictor, output_dir):
    """
    LIME analysis by stereotype group

    Args:
        df: DataFrame with texts, labels, groups
        explainer: LimeTextExplainer
        predictor: PredictorWrapper
        output_dir: Output directory
    """
    print("\nLIME analysis by stereotype group...")

    if 'group' not in df.columns:
        print("  ⚠ 'group' column not found, group analysis skipped")
        return

    # Most frequent groups
    top_groups = df['group'].value_counts().head(5).index.tolist()

    group_results = {}

    for group in top_groups:
        print(f"\n  Analyzing group: {group}")

        df_group = df[df['group'] == group]

        # Take some examples from the group
        n_examples = min(3, len(df_group))
        sample = df_group.sample(n_examples, random_state=42)

        texts = sample['text'].tolist()
        labels = sample['label'].tolist()

        # Create a subdirectory for this group
        group_dir = os.path.join(output_dir, f"group_{group.replace('/', '_')}")
        os.makedirs(group_dir, exist_ok=True)

        # Analyze with LIME
        results = analyze_with_lime(
            explainer,
            predictor,
            texts,
            labels,
            group_dir,
            top_k=n_examples,
            num_features=NUM_FEATURES,
            num_samples=NUM_SAMPLES_LIME
        )

        group_results[group] = results

    # Save a summary by group
    summary_file = os.path.join(output_dir, "lime_by_group_summary.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LIME SUMMARY BY GROUP - HEARTS-FR\n")
        f.write("=" * 80 + "\n\n")

        for group, results in group_results.items():
            f.write(f"\nGroup: {group}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Examples analyzed: {len(results)}\n\n")

            for result in results:
                idx = result['idx']
                text = result['text']
                f.write(f"  {idx + 1}. {text[:80]}...\n")

            f.write("\n")

    print(f"✓ Summary by group saved: {summary_file}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main LIME analysis function
    """
    print("=" * 80)
    print("INTERPRETABILITY ANALYSIS WITH LIME - HEARTS-FR")
    print("Model: CamemBERT on CrowS-Pairs-FR")
    print("=" * 80)

    # 1. Load model and tokenizer
    model, tokenizer, pipe = load_model_and_tokenizer()

    # 2. Load dataset
    df = load_dataset(DATASET_FILE, n_samples=N_SAMPLES)

    # 3. Prepare data
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # 4. Create predictor wrapper
    predictor = PredictorWrapper(model, tokenizer, DEVICE)

    # 5. Create LIME explainer
    class_names = ['Non-Stereotype', 'Stereotype']
    explainer = create_lime_explainer(class_names)

    # 6. Analyze examples
    top_k = min(20, len(texts))
    results = analyze_with_lime(
        explainer,
        predictor,
        texts,
        labels,
        OUTPUT_DIR,
        top_k=top_k,
        num_features=NUM_FEATURES,
        num_samples=NUM_SAMPLES_LIME
    )

    # 7. Extract important words
    extract_important_words(results, OUTPUT_DIR)

    # 8. Generate summary
    generate_lime_summary(results, OUTPUT_DIR)

    # 9. Group analysis (optional)
    analyze_by_group(df, explainer, predictor, OUTPUT_DIR)

    # Final summary
    print("\n" + "=" * 80)
    print("LIME ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  • lime_example_*.html - Interactive LIME explanations")
    print("  • lime_example_*.png - LIME visualizations")
    print("  • important_words.txt - Most important words")
    print("  • important_words.png - Important words visualization")
    print("  • lime_summary.txt - Analysis summary")
    print("  • lime_by_group_summary.txt - Summary by group")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
