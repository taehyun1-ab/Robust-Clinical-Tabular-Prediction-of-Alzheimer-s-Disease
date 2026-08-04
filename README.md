# Robust Clinical Tabular Prediction of Alzheimer's Disease Using Feature-Token Masking

Code for evaluating the robustness of Transformer-based clinical tabular prediction under missing-feature conditions.

## Overview

This repository contains the experimental code for the study:

**Robust Clinical Tabular Prediction of Alzheimer's Disease Using Feature-Token Masking Under Missing Feature Conditions**

The study investigates whether random feature-token masking during training improves the robustness of Transformer-based clinical tabular models when one or more clinical variables are unavailable during inference.

The proposed robust FT-Transformer represents each clinical variable as an individual feature token. During training, randomly selected feature tokens are replaced with a learnable mask token. This masking-aware training strategy is intended to reduce excessive dependence on a small number of dominant clinical variables and improve prediction stability under incomplete-input conditions.

The experiments include:

- complete-input classification
- random one-feature masking
- random two-feature masking
- masking-probability analysis
- single-feature ablation
- SHAP-based interpretation
- CLS-to-feature attention analysis

## Prediction Tasks

The models were evaluated on three binary classification tasks:

- AD vs. CN
- AD vs. MCI
- MCI vs. CN

The positive class was defined as:

- AD for AD vs. CN
- AD for AD vs. MCI
- MCI for MCI vs. CN

## Clinical Variables

Eight baseline clinical variables were used:

- AGE
- CDRSB
- FAQTOTAL
- MMSCORE
- BMI
- PULSE
- GENDER
- APOE4

The diagnostic label column is:

- DIAGNOSIS

Only baseline records were used. Early mild cognitive impairment and late mild cognitive impairment were combined into a single MCI group.

## Compared Models

Five models were evaluated:

- Random Forest
- XGBoost
- Multilayer Perceptron
- Standard FT-Transformer
- Robust FT-Transformer

The robust FT-Transformer applies random feature-token masking during training. The final masking probability used in the main experiments was `p = 0.2`.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   ├── README_data.md
│   └── sample/
│       └── ADNI_clinical_sample_data.csv
├── code/
│   ├── models.py
│   ├── main_experiment.py
│   ├── random_masking.py
│   ├── feature_ablation.py
│   └── interpretability.py
└── results/
    └── README_results.md
```

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- NumPy
- pandas
- scikit-learn
- PyTorch
- XGBoost
- SHAP
- matplotlib

## Data Availability

The clinical data used in this study were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI).

Due to the ADNI data-use agreement, raw participant-level data are not distributed through this repository. Qualified investigators must request access through the official ADNI data-access procedure.

No real participant-level ADNI data are included in this repository.

This repository provides:

- model implementation
- preprocessing procedures
- model training and evaluation code
- missing-feature robustness analysis
- masking-probability analysis
- single-feature ablation analysis
- model interpretation analysis

## Synthetic Sample Data

The `data/sample/` directory contains a synthetic CSV file illustrating the expected input format:

```text
data/sample/ADNI_clinical_sample_data.csv
```

The sample file does not contain real ADNI participant records. All values are artificially generated for format inspection only.

The synthetic sample data are not intended for:

- model training
- performance evaluation
- reproduction of the reported results

To reproduce the study results, users must obtain access to the original ADNI data and prepare a baseline clinical dataframe following the same column structure.

## Input Data Format

The expected input is a CSV file containing the following columns:

```text
AGE
CDRSB
FAQTOTAL
MMSCORE
BMI
PULSE
GENDER
APOE4
DIAGNOSIS
```

The expected diagnostic labels are:

```text
CN
MCI
AD
```

An example input structure is shown below:

```csv
AGE,CDRSB,FAQTOTAL,MMSCORE,BMI,PULSE,GENDER,APOE4,DIAGNOSIS
72.4,0.0,1.0,29.0,24.8,68,Male,0,CN
76.1,4.5,12.0,22.0,23.1,74,Female,1,AD
70.8,1.5,4.0,27.0,25.3,71,Male,1,MCI
```

## Data Preprocessing

Subjects with missing values in any of the eight selected clinical variables were excluded before model training and evaluation.

All preprocessing procedures were performed separately within each cross-validation fold.

Continuous variables were standardized using statistics estimated only from the training data within each fold. Categorical encoders were also fitted only on the training data and were subsequently applied to the corresponding validation and test sets.

No information from the held-out test set was used to estimate preprocessing parameters.

## Experimental Workflow

### 1. Main Experiment

```text
code/main_experiment.py
```

This script trains and evaluates the five comparison models under complete-input conditions.

Main steps:

- load the baseline clinical dataframe
- construct the three binary classification tasks
- perform stratified five-fold cross-validation
- fit preprocessing procedures within each fold
- train Random Forest
- train XGBoost
- train the multilayer perceptron
- train the standard FT-Transformer
- train the robust FT-Transformer
- evaluate performance on the held-out test folds
- save fold-wise and summary results
- save trained model checkpoints

Expected outputs:

```text
results/full_feature_results.csv
results/full_feature_fold_results.csv
results/checkpoints/
```

### 2. Random Missing-Feature Evaluation

```text
code/random_masking.py
```

This script evaluates trained models when one or two clinical variables are randomly unavailable for each test subject.

Two missing-feature conditions are evaluated:

- random masking of one feature per subject
- random masking of two features per subject

Each random masking condition is repeated 100 times to reduce dependence on a single randomly generated masking pattern.

For each subject, the masked feature or features are selected independently from the eight input variables.

Expected outputs:

```text
results/random_one_feature_results.csv
results/random_two_feature_results.csv
results/random_masking_summary.csv
```

### 3. Masking-Probability Analysis

The robust FT-Transformer is evaluated using the following training masking probabilities:

```text
p = 0.1
p = 0.2
p = 0.3
```

A masking probability of `p = 0.2` was selected for the main experiments because it provided the most balanced overall performance across the three classification tasks.

Expected output:

```text
results/masking_probability_results.csv
```

### 4. Single-Feature Ablation

```text
code/feature_ablation.py
```

This script evaluates the effect of removing one clinical variable at a time.

For the robust FT-Transformer, the selected feature is replaced with the learnable mask token during inference.

The reduction in discrimination performance is calculated as:

```text
Delta AUROC = Full-feature AUROC - Ablated-feature AUROC
```

A positive value indicates that AUROC decreased after feature ablation. A negative value indicates a slight increase relative to complete-input evaluation.

Expected output:

```text
results/single_feature_ablation_results.csv
```

### 5. Interpretability Analysis

```text
code/interpretability.py
```

This script performs interpretation analyses for the robust FT-Transformer.

The main analyses include:

- fold-wise SHAP value calculation
- pooling of out-of-fold SHAP values across five folds
- SHAP beeswarm visualization
- CLS-to-feature attention extraction
- averaging of attention weights across attention heads, test subjects, and cross-validation folds
- task-specific attention heatmap visualization

Expected outputs:

```text
results/shap_values/
results/attention_weights/
results/figures/shap_AD_vs_CN.png
results/figures/shap_AD_vs_MCI.png
results/figures/shap_MCI_vs_CN.png
results/figures/attention_heatmaps.png
```

## Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Sensitivity
- F1-score
- AUROC

Precision, sensitivity, and F1-score are macro-averaged across the two classes.

Missing-feature robustness is additionally evaluated using the reduction in AUROC relative to complete-input performance.

## Experimental Settings

The main experimental settings are:

- random seed: `17`
- cross-validation: stratified five-fold cross-validation
- number of input variables: `8`
- robust FT-Transformer masking probability: `0.2`
- random masking repetitions: `100`
- random masking conditions: one or two features per subject

The final model-specific hyperparameters are described in the manuscript and Supplementary Material.

## Reproducibility

All experiments used fixed random seeds.

The held-out test folds were not used for:

- hyperparameter selection
- early stopping
- model selection
- preprocessing parameter estimation

Continuous variables were standardized using training-derived statistics. Categorical encoders were fitted only on the corresponding training subsets.

The fitted preprocessing objects were then applied to the validation and held-out test sets.

For random missing-feature evaluation, the same trained model checkpoint was evaluated repeatedly using different subject-specific masking patterns.

## Notes on Data and Results

Raw ADNI data are excluded from this repository.

Participant-level prediction files, subject identifiers, processed participant-level datasets, and other restricted data-derived files are also excluded from version control.

The following directories are expected to be generated locally after running the code:

```text
results/checkpoints/
results/fold_results/
results/shap_values/
results/attention_weights/
results/figures/
```

Summary-level result files that do not contain participant identifiers may be included in the repository.

## Citation

Citation information will be added after publication.

## License

This repository is released under the MIT License. See the `LICENSE` file for details.

The license applies only to the source code and does not apply to the ADNI dataset.

## Acknowledgment

Data collection and sharing for this project were supported by the Alzheimer's Disease Neuroimaging Initiative.

## Contact

Taehyeon Yun  
Department of Data Engineering  
Pukyong National University  
Busan, Republic of Korea  
Email: ytae1014@pukyong.ac.kr
