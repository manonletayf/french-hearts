# HEARTS French Stereotype Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This project is a reproduction of the **HEARTS framework** adapted for **French language stereotype detection** on short sentences. The implementation follows the methodology from the [original HEARTS research paper](https://arxiv.org/abs/2409.11579), applying explainable, low-carbon models to detect stereotypes in French text.

---

## Project Structure

All experimental code and results are organized in the `project/` folder:

```
project/
├── Model/                              # Model training scripts and evaluation results
│   ├── baseline_albert_crows_simple.py
│   ├── baseline_albert_mgsd.py
│   ├── baseline_camembert_crows_fr.py
│   ├── final_baseline_camembert_crows_fr.py
│   ├── extract_camembert_errors.py
│   ├── result_output_albertv2/         # ALBERT-v2 evaluation results on MGSD
│   ├── result_output_albert_crows_simple/  # ALBERT results on CrowS-Pairs
│   ├── result_output_camembert/        # CamemBERT baseline results
│   └── result_output_camembert_final/  # Final CamemBERT results
│
├── visualisation/                      # Visualization scripts and outputs
│   ├── visualizations.py               # General visualization script
│   ├── visualizations_albert_mgsd.py   # ALBERT-specific visualizations
│   ├── visualizations_albert_mgsd/     # ALBERT visualization outputs
│   └── visualizations_poster/          # Publication-ready figures
│
├── TEST/                               # Statistical analysis and ablation studies
│   ├── ABLATION/                       # Ablation study scripts
│   ├── baseline_comparaison/           # Baseline comparison scripts
│   ├── failure_analysis/               # Error analysis scripts
│   └── significance/                   # Statistical significance tests
│
├── data/                               # Datasets and data processing
│   ├── crows_pairs_fr_final.csv        # French CrowS-Pairs dataset
│   ├── convert_crows_pairs_for_baseline.py
│   └── datageneration_commented.py
│
├── explainability_results/             # SHAP and LIME outputs
│   ├── lime/                           # LIME explanations
│   └── shap/                           # SHAP explanations
│
├── explainability_lime.py              # LIME explainability script
└── explainability_shap.py              # SHAP explainability script
```

---

## Key Directories

### Model/ - Training and Evaluation
Contains baseline training scripts and their evaluation results:

**Training Scripts:**
- **`baseline_albert_mgsd.py`**: Train ALBERT on MGSD dataset
- **`baseline_albert_crows_simple.py`**: Train ALBERT on French CrowS-Pairs
- **`baseline_camembert_crows_fr.py`**: Initial CamemBERT training
- **`final_baseline_camembert_crows_fr.py`**: Final optimized CamemBERT training
- **`extract_camembert_errors.py`**: Extract and analyze CamemBERT prediction errors

**Evaluation Results (`result_output_*/`):**
Each result folder contains:
- `full_results.csv`: Complete predictions with actual/predicted labels and probabilities
- `metrics_by_group.csv`: Per-group performance metrics (F1, precision, recall, balanced accuracy)
- `errors_by_group.csv`: Detailed error analysis by demographic group

### visualisation/ - Visual Analysis
Visualization scripts and their outputs:

**Scripts:**
- **`visualizations.py`**: Generate comprehensive visualizations (confusion matrices, F1 comparisons, error distributions, word clouds)
- **`visualizations_albert_mgsd.py`**: ALBERT-specific visualizations on MGSD dataset

**Outputs:**
- **`visualizations_poster/`**: Publication-ready, high-resolution figures
- **`visualizations_albert_mgsd/`**: ALBERT performance charts and analysis plots

### TEST/ - Advanced Analysis
Testing, ablation studies, and statistical comparisons:

- **`ABLATION/`**: Hyperparameter ablation studies (dropout rates, learning rates, max sequence length, epochs)
- **`baseline_comparaison/`**: Scripts comparing different baseline models (DistilBERT, Logistic Regression, majority baseline)
- **`failure_analysis/`**: Detailed error case studies and failure mode analysis
- **`significance/`**: Statistical significance tests between models (McNemar's test, paired t-tests)

### data/ - Datasets
- **`crows_pairs_fr_final.csv`**: Main French stereotype detection dataset (CrowS-Pairs adaptation)
- Data preprocessing and generation scripts

### Explainability
- **Root scripts** (`explainability_lime.py`, `explainability_shap.py`): Generate model explanations
- **`explainability_results/`**: Saved SHAP and LIME outputs for analysis

---




## Key Features

- **French Language Adaptation**: CamemBERT-based models fine-tuned for French stereotype detection
- **Organized Structure**: Clear separation of training, visualization, testing, and analysis
- **Cross-lingual Evaluation**: Comparative analysis with English models (ALBERT, DistilBERT)
- **Explainable AI**: SHAP and LIME implementations for model interpretability
- **Comprehensive Testing**: Ablation studies, statistical significance tests, and failure analysis
- **Per-Group Metrics**: Fairness analysis across different demographic groups
- **Publication-Ready Visualizations**: High-resolution figures optimized for research papers
- **Low-Carbon Focus**: Following HEARTS methodology for efficient, explainable models

---

# Technical Implementation

## 1. Replicate the baseline AI methodology using the open dataset

### 1.a) Cloning the Official HEARTS Repository
The official HEARTS repository was cloned from GitHub, providing scripts, model configs, explainability tools, and evaluation modules needed to reproduce the baseline.  

### 1.b) Document all dependencies and environment setup
A dedicated Python 3.10.19 virtual environment was created. All dependencies from the repository’s requirements (PyTorch, Transformers, datasets, SHAP/LIME, etc.) were installed, ensuring reproducibility.  

### 1.c) Reproduce baseline results within ±5%
The ALBERT-v2 + MGSD baseline was reproduced by isolating the corresponding training script.  
Obtained macro-F1: **0.7852**, within 1.5% of the original (0.797).  

### 1.d) Provide reproducible notebook or Python scripts
A simplified script (`baseline_albert_mgsd.py`) was created to reproduce the baseline cleanly, using the same hyperparameters and exporting evaluation reports.  

---

## 2. Identify a contextually relevant challenge

### 2.a) Problem and SDG alignment
Stereotype detection for French-language text is essential because most existing tools and benchmark datasets are developed in English and reflect US-centric social categories, norms, and linguistic patterns. This creates a blind spot for francophone contexts, where stereotypes may target different groups, use different expressions, and operate under different legal and historical frameworks. Developing a French-specific, explainable stereotype detection model directly supports several Sustainable Development Goals (SDGs) while also addressing local challenges in France.
  
This contributes to SDGs: **10 (Inequalities), 5 (Gender), 8 (Decent Work), 16 (Institutions)**. Automated systems deployed in hiring, housing allocation, social services, and credit scoring increasingly rely on textual data (CVs, motivation letters, case notes, free-text comments). If these systems are trained on biased data or use unchecked language models, they can silently reproduce stereotypes about nationality, gender, ethnicity, religion, disability, age, or socioeconomic background. 

However, reducing inequalities is not automatic: detection must be embedded within governance frameworks (e.g. internal audits, corrective procedures, rights to appeal) to have concrete redistributive effects.

### 2.b) Limitations and ethical considerations
CrowS-Pairs-FR contains offensive stereotypes; handling requires care.  
Some US-specific stereotypes do not translate culturally.  
The dataset is anonymous, but representational imbalance across categories persists.  
Explainability and controlled data use mitigate ethical risk.  

France’s legal and cultural reluctance to collect ethnic statistics complicates the operationalisation of “protected groups”. A stereotype model can highlight biased language but cannot directly be linked to granular demographic impact without careful legal and ethical justification.

### 2.c) Scalability and sustainability
ALBERT and CamemBERT are lightweight, low-carbon models.  
The pipeline is reproducible and extensible to other francophone regions.  
Explainability runs only at inference, keeping compute low.  

---

## 3. Curate or identify an alternative dataset

### 3.a) Identify contextually appropriate dataset
The **French CrowS-Pairs** dataset was chosen due to its explicit design for evaluating stereotypes across demographic groups, including culturally specific French samples.  
The French CrowS-Pairs dataset is composed of 1,467 sentence pairs translated and culturally adapted from the original English CrowS-Pairs corpus, and 210 additional stereotype pairs newly collected from French-speaking volunteers through the LanguageARC platform. This ensures that the dataset reflects both the structure of the original benchmark and the socio-cultural specificities of stereotypes expressed in French.  

### 3.b) Document data collection/access process
The dataset is public, anonymised, and ethically sourced via INRIA and LanguageARC.  
Despite containing harmful text, it is appropriate for stereotype-mitigation research under controlled use.  

### 3.c) Provide data preprocessing pipeline
The preprocessing included three main steps:

1. **Convert CrowS-Pairs pairs → single-sentence stereotype dataset.**  
2. **Generate neutral and unrelated variants** via GPT-4o-mini using strict prompts translated from HEARTS SeeGULL prompt.  
3. **Combine all sentences** into a balanced dataset (≈5,029 samples).  

Group labels from CrowS-Pairs are preserved for fairness analysis.  
Note that in this project, as well as in the original HEARTS framework, the “neutral” versions are not fully neutral in an absolute sense; instead, they aim to neutralise the stereotypical content by replacing the stereotype with a positive or genuinely neutral word or expression, while preserving the original identity terms. The aim is to detect negative stereotypes, not generalization.

---

## 4. Adapt the model architecture and training pipeline

### 4.a) Justify architectural modifications
The ALBERT baseline was replaced by **CamemBERT-base**, more suitable for French.  
The training structure remains identical to HEARTS, with added group-level fairness evaluation.  
The dataset was replaced by the generated **CrowS-Pairs-FR final dataset**.  

### 4.b) Document hyper-parameter tuning
Several ablations were performed:

- Removing unrelated examples reduces performance.  
- Removing neutral creates misleading high scores (model exploits absurdity).   
- Reducing max length (max token per sentence) to 128 improves efficiency.  
- Learning rate 5e-5 yields best macro-F1.  

**Final configuration:**  
Binary classification, CamemBERT, LR = 5e-5, epochs = 6, batch = 64, max_len = 128.  

---

## 5. Evaluate the adapted model

### 5.a) Compare original vs. adapted model performance
- **Original HEARTS baseline (ALBERT + MGSD):** macro-F1 ≈ 0.785  
- **Adapted model (CamemBERT + FR CrowS-Pairs):** macro-F1 ≈ 0.825  

→ The French adaptation performs better despite a smaller dataset and language change.  

### 5.b) Use appropriate metrics
Primary metric: **Macro-F1**, as it balances class imbalance and penalises poor performance on “stereotype”.  
Additional metrics include accuracy, macro precision/recall, weighted F1, and per-group fairness metrics (balanced accuracy).  

### 5.c) Statistical significance testing
McNemar and bootstrap tests confirm that **CamemBERT-FR-Final** significantly outperforms baseline and multilingual models.   

### 5.d) Failure case analysis
Key weaknesses:

- **False negatives dominate** → implicit stereotypes remain hard to detect.  
- **False positives** triggered by identity-related words.  
- **Group-level variability:** gender, age, and race-color show lower F1 than socioeconomic, disability, etc.  

→ Performance is uneven across bias categories, echoing fairness concerns. 

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
