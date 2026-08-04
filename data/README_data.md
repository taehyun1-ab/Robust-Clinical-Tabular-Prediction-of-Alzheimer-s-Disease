# Data Directory

This directory describes the expected input data format for the experiments in this repository.

## Data Source

The clinical data used in this study were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI).

Due to the ADNI data-use agreement, raw participant-level ADNI data are not included in this repository. Users who wish to reproduce the study must request access through the official ADNI data-access procedure.

No real participant-level ADNI data are distributed with this repository.

## Directory Structure

```text
data/
├── README_data.md
└── sample/
    └── ADNI_clinical_sample_data.csv
```

The `sample/` directory contains a synthetic CSV file illustrating the expected input structure.

## Synthetic Sample Data

The following file is provided for format inspection:

```text
data/sample/ADNI_clinical_sample_data.csv
```

This file contains artificially generated values and does not include any real ADNI participant records.

The synthetic sample data are intended only to illustrate:

- required column names
- expected data types
- diagnostic label format
- input file structure

The synthetic sample data are not intended for:

- model training
- performance evaluation
- statistical analysis
- reproduction of the reported results

To reproduce the study results, users must obtain authorized access to the original ADNI data and prepare a baseline clinical dataframe following the format described below.

## Expected Input Format

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

## Variable Description

| Variable | Description |
|---|---|
| `AGE` | Age at baseline |
| `CDRSB` | Clinical Dementia Rating–Sum of Boxes |
| `FAQTOTAL` | Functional Activities Questionnaire total score |
| `MMSCORE` | Mini-Mental State Examination score |
| `BMI` | Body mass index |
| `PULSE` | Pulse measurement |
| `GENDER` | Participant gender |
| `APOE4` | APOE ε4 status |
| `DIAGNOSIS` | Baseline diagnostic label |

## Diagnostic Labels

The expected values in the `DIAGNOSIS` column are:

```text
CN
MCI
AD
```

The labels represent:

- `CN`: cognitively normal
- `MCI`: mild cognitive impairment
- `AD`: Alzheimer's disease

Early mild cognitive impairment and late mild cognitive impairment are combined into a single MCI group.

## Example Data Format

```csv
AGE,CDRSB,FAQTOTAL,MMSCORE,BMI,PULSE,GENDER,APOE4,DIAGNOSIS
72.4,0.0,1.0,29.0,24.8,68,Male,0,CN
76.1,4.5,12.0,22.0,23.1,74,Female,1,AD
70.8,1.5,4.0,27.0,25.3,71,Male,1,MCI
```

## Inclusion Criteria

The experiments use baseline clinical records only.

Participants are included when:

- a valid baseline diagnosis is available
- the diagnosis belongs to CN, MCI, or AD
- all eight selected clinical variables are available

Participants are excluded when:

- the baseline diagnosis is missing
- the diagnosis is outside the target groups
- one or more selected clinical variables are missing

No imputation is performed in the main experiments because participants with missing values in the selected variables are excluded before model training and evaluation.

## Preprocessing

All preprocessing procedures are performed independently within each cross-validation fold.

Continuous variables are standardized using statistics estimated only from the training data within each fold.

Categorical encoders are also fitted only on the training data and then applied to the corresponding validation and test sets.

The held-out test data are not used to estimate preprocessing parameters.

## Data Preparation Notes

The repository does not provide a script for downloading restricted ADNI data.

Users are responsible for:

- obtaining authorized access to ADNI
- downloading the required clinical data
- selecting baseline records
- harmonizing diagnostic labels
- selecting the eight clinical variables
- excluding incomplete records
- saving the prepared dataframe in CSV format

The prepared input file should follow the column names and label format described in this document.

## Data-Use Restrictions

The source code in this repository is released under the MIT License.

The MIT License does not apply to ADNI data.

Access to, use of, and redistribution of ADNI data remain subject to the applicable ADNI data-use agreements and policies.
