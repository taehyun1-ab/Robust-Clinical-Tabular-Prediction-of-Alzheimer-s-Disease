# Robust Clinical Tabular Prediction of Alzheimer's Disease Using Feature-Token Masking

Code for evaluating the robustness of Transformer-based clinical tabular prediction under missing-feature conditions.

## Overview

This repository contains the experimental code and aggregate results for the study:

**Robust Clinical Tabular Prediction of Alzheimer's Disease Using Feature-Token Masking Under Missing Feature Conditions**

The study investigates whether random feature-token masking during training improves the robustness of Transformer-based clinical tabular models when one or more clinical variables are unavailable during inference.

The proposed robust Feature Tokenizer Transformer (FT-Transformer) represents each clinical variable as an individual feature token. During training, randomly selected feature tokens are replaced with a learnable mask token. This masking-aware training strategy is intended to improve prediction stability under incomplete-input conditions.

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

- `AGE`
- `CDRSB`
- `FAQTOTAL`
- `MMSCORE`
- `BMI`
- `PULSE`
- `GENDER`
- `APOE4`

The diagnostic label column is:

- `DIAGNOSIS`

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
│       └── synthetic_clinical_sample.csv
├── figures/
│   ├── Fig1_model_architecture.png
│   ├── Fig2a_shap_AD_vs_CN.png
│   ├── Fig2b_shap_AD_vs_MCI.png
│   ├── Fig2c_shap_MCI_vs_CN.png
│   └── Fig3_attention_heatmaps.png
├── results/
│   ├── README_results.md
│   ├── Table1_subject_counts.csv
│   ├── Table2_complete_input_results.csv
│   ├── Table3_random_masking_results.csv
│   ├── Table4_single_feature_ablation.csv
│   └── supplementary/
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_utils.py
    ├── models.py
    ├── train_utils.py
    ├── main_experiment.py
    ├── random_masking.py
    ├── masking_probability_analysis.py
    ├── feature_ablation.py
    ├── interpretability.py
    └── test_sample_data.py
```

The unused `src/models/` directory should be removed if it is empty, because the model definitions are stored in `src/models.py`.

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

**No real participant-level ADNI data are included in this repository.**

This repository provides:

- model implementations
- fold-specific preprocessing procedures
- model training and evaluation code
- missing-feature robustness analysis
- masking-probability analysis
- single-feature ablation analysis
- model interpretation analysis
- synthetic sample data for code verification
- aggregate manuscript-level results

## Synthetic Sample Data

The repository contains a synthetic CSV file illustrating the expected input format:

```text
data/sample/synthetic_clinical_sample.csv
```

The file contains 200 artificially generated records and does not contain real ADNI participant data.

The synthetic data are provided only for:

- input-format inspection
- preprocessing verification
- model forward-pass testing
- code execution and workflow testing

The synthetic data must not be used for:

- reproducing the performance reported in the manuscript
- clinical interpretation
- clinical decision-making
- comparison with the reported ADNI results

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

Example:

```csv
AGE,CDRSB,FAQTOTAL,MMSCORE,BMI,PULSE,GENDER,APOE4,DIAGNOSIS
72.4,0.0,1,29,24.8,68,Male,0,CN
76.1,4.5,12,22,23.1,74,Female,1,AD
70.8,1.5,4,27,25.3,71,Male,1,MCI
```

## Data Preprocessing

All preprocessing procedures were performed separately within each cross-validation fold.

For Random Forest and XGBoost:

- continuous missing values were replaced using the median estimated from the corresponding training fold
- categorical missing values were replaced using the most frequent category estimated from the training fold
- categorical variables were ordinal-encoded using an encoder fitted only on the training fold

For the MLP:

- continuous missing values were replaced using the training-fold median
- categorical missing values were replaced using the training-fold mode
- continuous variables were standardized using a `StandardScaler` fitted only on the training subset
- categorical variables were ordinal-encoded using an encoder fitted only on the training subset

For the standard and robust FT-Transformer:

- continuous missing values were replaced using the training-fold median
- categorical missing values were represented using a dedicated `"Missing"` category
- continuous variables were standardized using a `StandardScaler` fitted only on the training subset
- categorical variables were ordinal-encoded using an encoder fitted only on the training subset

No information from the held-out test folds was used to estimate imputation, scaling, or encoding parameters.

## Cross-Validation and Validation Strategy

All models were evaluated using stratified five-fold cross-validation.

For Random Forest and XGBoost:

- four folds were used for training
- one fold was used as the held-out test set
- no additional validation subset was created during the final evaluation because the selected hyperparameters were fixed before the reported evaluation

For the MLP, standard FT-Transformer, and robust FT-Transformer:

- one fold was used as the held-out test set
- the remaining four folds formed the outer training portion
- 10% of the outer training portion was further separated as an internal validation set
- the validation set was used for early stopping
- the held-out test fold was used only for final performance evaluation

The held-out test folds were not used for:

- hyperparameter selection
- early stopping
- model selection
- imputation parameter estimation
- scaling parameter estimation
- categorical encoding

## Testing with Synthetic Sample Data

Run the following command from the repository root:

```bash
python -m src.test_sample_data
```

The script automatically loads:

```text
data/sample/synthetic_clinical_sample.csv
```

This lightweight smoke test verifies:

- CSV loading
- required-column validation
- construction of the three classification tasks
- tree-based preprocessing
- neural-network preprocessing
- MLP forward pass
- FT-Transformer forward pass
- robust FT-Transformer forward pass
- training-time feature-token masking

The smoke test does not perform the complete manuscript experiment and does not reproduce the reported results.

## Running the Complete Training Pipeline on Synthetic Data

The complete training pipeline can also be executed using the synthetic sample:

```bash
python -m src.main_experiment \
  --data_path data/sample/synthetic_clinical_sample.csv \
  --output_dir results/sample_demo
```

This command runs the full five-fold training workflow using the synthetic data. The generated metrics are only for workflow verification and must not be interpreted as manuscript results.

## Experimental Workflow

### 1. Complete-Input Experiment

Script:

```text
src/main_experiment.py
```

Run:

```bash
python -m src.main_experiment \
  --data_path /path/to/prepared_clinical_dataframe.csv \
  --output_dir results/main_experiment
```

The script:

- loads the prepared clinical dataframe
- constructs the three binary classification tasks
- performs stratified five-fold cross-validation
- applies fold-specific preprocessing
- trains Random Forest
- trains XGBoost
- trains the MLP
- trains the standard FT-Transformer
- trains the robust FT-Transformer
- evaluates complete-input performance on held-out test folds
- saves fold-wise and summary results
- saves Transformer checkpoints locally

Locally generated outputs include:

```text
results/main_experiment/
├── complete_input_fold_results.csv
├── complete_input_summary.csv
└── checkpoints/
```

Checkpoints and participant-level outputs should not be committed to the public repository.

### 2. Random Missing-Feature Evaluation

Script:

```text
src/random_masking.py
```

Run:

```bash
python -m src.random_masking \
  --data_path /path/to/prepared_clinical_dataframe.csv \
  --checkpoint_dir results/main_experiment/checkpoints \
  --output_dir results/random_masking
```

The script evaluates trained FT-Transformer and robust FT-Transformer checkpoints under:

- complete-input evaluation
- random masking of one feature per subject
- random masking of two features per subject

For each subject, masked features are selected independently from the eight input variables.

The one-feature and two-feature masking conditions are repeated 100 times.

Locally generated outputs include:

```text
results/random_masking/
├── random_masking_repeat_results.csv
└── random_masking_summary.csv
```

### 3. Masking-Probability Analysis

Script:

```text
src/masking_probability_analysis.py
```

Run:

```bash
python -m src.masking_probability_analysis \
  --data_path /path/to/prepared_clinical_dataframe.csv \
  --output_dir results/masking_probability
```

The robust FT-Transformer is trained using:

- `p = 0.1`
- `p = 0.2`
- `p = 0.3`

The main experiments used `p = 0.2`.

Locally generated outputs include:

```text
results/masking_probability/
├── masking_probability_fold_results.csv
└── masking_probability_summary.csv
```

### 4. Single-Feature Ablation

Script:

```text
src/feature_ablation.py
```

Run:

```bash
python -m src.feature_ablation \
  --data_path /path/to/prepared_clinical_dataframe.csv \
  --checkpoint_dir results/main_experiment/checkpoints \
  --output_dir results/feature_ablation
```

For the robust FT-Transformer, each selected feature is replaced with the learned mask token during inference.

The AUROC degradation is defined as:

```text
Delta AUROC = Complete-input AUROC - Ablated-input AUROC
```

A positive value indicates a decrease in AUROC after masking the feature.

Locally generated outputs include:

```text
results/feature_ablation/
├── single_feature_ablation_fold_results.csv
└── single_feature_ablation_summary.csv
```

### 5. Interpretability Analysis

Script:

```text
src/interpretability.py
```

The interpretation analysis includes:

- fold-wise SHAP value calculation
- pooling of out-of-fold SHAP values across five folds
- SHAP beeswarm visualization
- CLS-to-feature attention extraction
- averaging of attention weights across attention heads, test subjects, and cross-validation folds
- task-specific attention heatmap visualization

The public repository should contain only aggregate interpretation results and final figures. Participant-level SHAP values, subject identifiers, and raw attention values should not be committed.

> Important: `src/interpretability.py` should contain the complete executable SHAP and attention implementation before public release. A placeholder-only version should not be published.

## Evaluation Metrics

Model performance was evaluated using:

- Accuracy
- Precision
- Sensitivity
- F1-score
- AUROC

Precision, sensitivity, and F1-score were macro-averaged across the two classes.

Missing-feature robustness was additionally evaluated using the reduction in AUROC relative to complete-input performance.

## Experimental Settings

The main experimental settings were:

- random seed: `17`
- outer cross-validation: stratified five-fold cross-validation
- internal validation ratio for neural models: `10%` of the outer training portion
- number of input variables: `8`
- robust FT-Transformer masking probability: `0.2`
- random masking repetitions: `100`
- random masking conditions: zero, one, or two features per subject

The final model-specific hyperparameters are defined in `src/config.py` and reported in the Supplementary Material.

## Manuscript-Level Results

The `results/` directory contains aggregate results reported in the manuscript:

```text
results/
├── Table1_subject_counts.csv
├── Table2_complete_input_results.csv
├── Table3_random_masking_results.csv
├── Table4_single_feature_ablation.csv
└── supplementary/
```

These files contain only summary-level results and do not contain participant identifiers.

## Figures

The `figures/` directory contains the final manuscript figures:

```text
figures/
├── Fig1_model_architecture.png
├── Fig2a_shap_AD_vs_CN.png
├── Fig2b_shap_AD_vs_MCI.png
├── Fig2c_shap_MCI_vs_CN.png
└── Fig3_attention_heatmaps.png
```

## Reproducibility

All experiments used fixed random seeds.

Preprocessing was fitted independently within each fold. The held-out test data were not used for fitting imputers, scalers, encoders, or early-stopping criteria.

For random missing-feature evaluation, the same trained model checkpoint was repeatedly evaluated using different subject-specific masking patterns.

Because the original ADNI participant-level data cannot be redistributed, the synthetic dataset supports code and workflow verification only. Exact manuscript results require authorized access to the original ADNI data and the same data-preparation procedure.

## Notes on Restricted Data and Outputs

The following items are excluded from the public repository:

- raw ADNI data
- processed participant-level ADNI data
- participant identifiers
- participant-level predictions
- participant-level probabilities
- participant-level SHAP values
- raw subject-level attention values
- model checkpoints trained on restricted ADNI data

The following directories may be generated locally:

```text
results/main_experiment/checkpoints/
results/random_masking/
results/masking_probability/
results/feature_ablation/
results/interpretability/
```

Only aggregate, non-identifiable result files should be committed.

## Citation

Citation information will be added after publication.

## License

This repository is released under the MIT License. See the `LICENSE` file for details.

The license applies only to the source code and repository documentation. It does not apply to the ADNI dataset or other restricted third-party data.

## Acknowledgment

Data collection and sharing for this project were supported by the Alzheimer's Disease Neuroimaging Initiative.

## Contact

Taehyeon Yun  
Department of Data Engineering  
Pukyong National University  
Busan, Republic of Korea  
Email: ytae1014@pukyong.ac.kr
