"""
Interpretability Analysis with SHAP - HEARTS-FR
Adapted for CamemBERT on CrowS-Pairs-FR

SHAP (SHapley Additive exPlanations) explains model predictions
by identifying the importance of each token in the decision.

Usage:
    python explainability_shap.py
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
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
OUTPUT_DIR = "explainability_results/shap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SHAP parameters
N_SAMPLES = 50  # Number of examples to analyze (SHAP is computationally expensive)
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

    return model, tokenizer


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
# WRAPPER FOR SHAP
# ============================================================================

class CamemBERTWrapper:
    """
    Wrapper to make the model compatible with SHAP
    """
    def __init__(self, model, tokenizer, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = next(model.parameters()).device

    def predict_proba(self, texts):
        """
        Probability prediction for SHAP

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
# SHAP ANALYSIS
# ============================================================================

def create_shap_explainer(model, tokenizer, background_texts):
    """
    Create a SHAP explainer

    Args:
        model: CamemBERT model
        tokenizer: Tokenizer
        background_texts: Reference texts for SHAP

    Returns:
        SHAP explainer
    """
    print("\nCreating SHAP explainer...")

    # Model wrapper
    wrapper = CamemBERTWrapper(model, tokenizer)

    # Create an explainer with a text masker
    # shap.maskers.Text expects a complete Hugging Face tokenizer (which returns
    # input_ids), not just the .tokenize method which returns a list.
    masker = shap.maskers.Text(tokenizer, mask_token=tokenizer.mask_token)

    # Partition explainer (faster than Kernel SHAP)
    explainer = shap.Explainer(
        wrapper.predict_proba,
        masker,
        algorithm="partition"
    )

    print("✓ SHAP explainer created")
    return explainer


def analyze_with_shap(explainer, texts, labels, output_dir, top_k=10):
    """
    Analyze examples with SHAP

    Args:
        explainer: SHAP explainer
        texts: List of texts to analyze
        labels: Real labels
        output_dir: Output directory
        top_k: Number of top examples to visualize
    """
    print(f"\nSHAP analysis of {len(texts)} examples...")

    results = []

    for idx, text in enumerate(tqdm(texts[:top_k], desc="SHAP analysis")):
        try:
            # Calculate SHAP values
            shap_values = explainer([text])

            # Save results
            result = {
                'text': text,
                'label': labels[idx],
                'shap_values': shap_values
            }
            results.append(result)

            # Individual visualization
            plt.figure(figsize=(12, 6))

            # Text plot for class 1 (stereotype)
            shap.plots.text(shap_values[:, :, 1], display=False)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_example_{idx}.png"),
                       dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"  ⚠ Error on example {idx}: {str(e)}")
            continue

    print(f"✓ {len(results)} examples analyzed")
    return results


def generate_shap_summary(results, output_dir):
    """
    Generate a summary of SHAP analyses

    Args:
        results: List of SHAP results
        output_dir: Output directory
    """
    print("\nGenerating SHAP summary...")

    summary_file = os.path.join(output_dir, "shap_summary.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SHAP ANALYSIS SUMMARY - HEARTS-FR\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Number of examples analyzed: {len(results)}\n\n")

        for idx, result in enumerate(results):
            f.write(f"\nExample {idx + 1}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Text: {result['text']}\n")
            f.write(f"Label: {result['label']}\n")
            f.write(f"Visualization: shap_example_{idx}.png\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")
        f.write("The SHAP visualizations show:\n")
        f.write("- Words in RED: Contribute to predicting 'stereotype'\n")
        f.write("- Words in BLUE: Contribute to predicting 'non-stereotype'\n")
        f.write("- Color intensity: Importance of the contribution\n\n")

    print(f"✓ Summary saved: {summary_file}")


# ============================================================================
# ANALYSIS BY GROUP
# ============================================================================

def analyze_by_group(df, explainer, output_dir):
    """
    SHAP analysis by stereotype group

    Args:
        df: DataFrame with texts, labels, groups
        explainer: SHAP explainer
        output_dir: Output directory
    """
    print("\nSHAP analysis by stereotype group...")

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

        # Analyze with SHAP
        results = analyze_with_shap(
            explainer,
            texts,
            labels,
            os.path.join(output_dir, f"group_{group.replace('/', '_')}"),
            top_k=n_examples
        )

        group_results[group] = results

    # Save a summary by group
    summary_file = os.path.join(output_dir, "shap_by_group_summary.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SHAP SUMMARY BY GROUP - HEARTS-FR\n")
        f.write("=" * 80 + "\n\n")

        for group, results in group_results.items():
            f.write(f"\nGroup: {group}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Examples analyzed: {len(results)}\n\n")

            for idx, result in enumerate(results):
                f.write(f"  {idx + 1}. {result['text'][:100]}...\n")

            f.write("\n")

    print(f"✓ Summary by group saved: {summary_file}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main SHAP analysis function
    """
    print("=" * 80)
    print("INTERPRETABILITY ANALYSIS WITH SHAP - HEARTS-FR")
    print("Model: CamemBERT on CrowS-Pairs-FR")
    print("=" * 80)

    # 1. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 2. Load dataset
    df = load_dataset(DATASET_FILE, n_samples=N_SAMPLES)

    # 3. Prepare data
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    # Background texts for SHAP (representative sample)
    background_texts = texts[:min(10, len(texts))]

    # 4. Create SHAP explainer
    explainer = create_shap_explainer(model, tokenizer, background_texts)

    # 5. Analyze examples
    top_k = min(20, len(texts))
    results = analyze_with_shap(explainer, texts, labels, OUTPUT_DIR, top_k=top_k)

    # 6. Generate summary
    generate_shap_summary(results, OUTPUT_DIR)

    # 7. Group analysis (optional)
    analyze_by_group(df, explainer, OUTPUT_DIR)

    # Final summary
    print("\n" + "=" * 80)
    print("SHAP ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  • shap_example_*.png - Individual SHAP visualizations")
    print("  • shap_summary.txt - Analysis summary")
    print("  • shap_by_group_summary.txt - Summary by group")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
