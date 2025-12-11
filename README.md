# HEARTS French Stereotype Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This project is a reproduction of the **HEARTS framework** adapted for **French language stereotype detection** on short sentences. The implementation follows the methodology from the [original HEARTS research paper](https://arxiv.org/abs/2409.11579), applying explainable, low-carbon models to detect stereotypes in French text.

---

## Project Structure

All experimental code and results are contained in the `experience` folder, which includes:

### Baseline Models

#### CamemBERT Models (French)
- **`baseline_camembert_crows_fr.py`**: Initial CamemBERT baseline on French CrowS-Pairs dataset
- **`baseline_camembert_crows_fr_v2.py`**: Second iteration with improvements
- **`baseline_camembert_crows_fr_v3.py`**: Third iteration with optimizations
- **`final_baseline_camembert_crows_fr.py`**: Final optimized CamemBERT implementation

#### ALBERT Models (Cross-lingual Comparison)
- **`baseline_albert_mgsd.py`**: ALBERT baseline on MGSD dataset
- **`baseline_albert_crows_simple.py`**: ALBERT baseline on CrowS-Pairs dataset

### Model Analysis

#### Explainability
- **`explainability_shap.py`**: SHAP (SHapley Additive exPlanations) analysis for interpreting model predictions
- **`explainability_lime.py`**: LIME (Local Interpretable Model-agnostic Explanations) for model transparency

#### Metrics and Error Analysis
- **`compute_group_metrics_albert_mgsd.py`**: Compute performance metrics per demographic group
- **`extract_camembert_errors.py`**: Detailed error analysis for CamemBERT predictions

#### Visualizations
- **`visualizations.py`**: General visualization scripts for results and analysis
- **`visualizations_albert_mgsd.py`**: Specific visualizations for ALBERT model performance on MGSD

### Experimental Results

The `experience` folder contains multiple subdirectories with experimental outputs:

- **Model Outputs**: Saved predictions and model checkpoints from various configurations
  - `model_output_camembert/`, `model_output_camembert_v2/`, `model_output_camembert_v3/`, `model_output_camembert_final/`
  - `model_output_albertv2/`, `model_output_albert_crows_simple/`
  - `model_output_distilbert/`
  - `model_output_LR_tfidf/` (Logistic Regression with TF-IDF)

- **Result Outputs**: Performance metrics and evaluation results
  - `result_output_camembert/`, `result_output_camembert_final/`
  - `result_output_albertv2/`, `result_output_baselines/`
  - Various ablation study results

- **Ablation Studies**: Results from hyperparameter tuning experiments
  - `result_output_ablation_3class/`: Three-class classification experiments
  - `result_output_ablation_drop_neutral/`: Experiments without neutral class
  - `result_output_ablation_drop_unrelated/`: Experiments without unrelated class
  - `result_output_ablation_epoch8/`: 8-epoch training experiments
  - `result_output_ablation_lr5e5/`: Learning rate 5e-5 experiments
  - `result_output_ablation_maxlen128/`: Max sequence length 128 experiments

- **Analysis Outputs**:
  - `failure_analysis_results/`: Detailed error case studies
  - `explainability_results/`: SHAP and LIME interpretation outputs
  - `visualizations_albert_mgsd/`: Charts and plots for ALBERT performance
  - `visualizations_poster/`: Publication-ready visualizations
  - `results_significance/`: Statistical significance test results

- **Data**: Preprocessed French datasets and intermediate files

---

## Quickstart

1. Navigate to the experience folder:
   ```bash
   cd experience
   ```

2. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

3. Run a baseline model (example with CamemBERT):
   ```bash
   python final_baseline_camembert_crows_fr.py
   ```

4. Generate explainability analysis:
   ```bash
   python explainability_shap.py
   python explainability_lime.py
   ```

5. Create visualizations:
   ```bash
   python visualizations.py
   ```

---

## Key Features

- **French Language Adaptation**: CamemBERT-based models fine-tuned for French stereotype detection
- **Cross-lingual Evaluation**: Comparative analysis with English models (ALBERT, DistilBERT)
- **Explainable AI**: SHAP and LIME implementations for model interpretability
- **Comprehensive Ablation Studies**: Systematic evaluation of hyperparameters and dataset configurations
- **Per-Group Metrics**: Fairness analysis across different demographic groups
- **Low-Carbon Focus**: Following HEARTS methodology for efficient, explainable models

---

## Methodology

This reproduction follows the HEARTS framework:

1. **Data Preparation**: French adaptation of stereotype detection datasets (CrowS-Pairs)
2. **Model Training**: Fine-tuning CamemBERT and other transformer models on French stereotyped sentences
3. **Evaluation**: Performance assessment with special attention to per-group fairness metrics
4. **Explainability**: SHAP and LIME analysis to understand model decision-making
5. **Ablation Studies**: Systematic evaluation of model configurations and hyperparameters

---

## Original HEARTS Framework

This project builds upon the HEARTS methodology described in:

```
@article{hearts2024,
  title={HEARTS: Enhancing Stereotype Detection with Explainable, Low-Carbon Models},
  author={Author Names},
  journal={arXiv preprint arXiv:2409.11579},
  year={2024}
}
```

For the original English implementation and EMGSD dataset, see the [HEARTS paper](https://arxiv.org/abs/2409.11579) and [Hugging Face dataset](https://huggingface.co/datasets/holistic-ai/EMGSD).

---

## License

This repository is licensed under the [MIT License](LICENSE).
