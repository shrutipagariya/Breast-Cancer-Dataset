
# Breast Cancer MLOps Data Pipeline

## 1. Introduction

This project implements an automated data pipeline for the Breast Cancer Classification dataset using Python, Pandas, Scikit-learn and DVC.

The pipeline performs data collection, preprocessing, feature engineering and data validation.

## 2. Dataset Description

The project uses the Wisconsin Breast Cancer dataset from Scikit-learn.

- Total records: 569
- Original columns: 30 input features and 1 target column
- Target 0: Malignant
- Target 1: Benign
- Training records: 455
- Testing records: 114

## 3. Pipeline Stages

### Stage 1: Data Collection

File: `src/pipeline/collect.py`

The dataset is loaded using Scikit-learn and saved as:

`data/raw/breast_cancer.csv`

### Stage 2: Data Preprocessing

File: `src/pipeline/preprocess.py`

The preprocessing stage:
- Removes duplicate records.
- Removes missing values.
- Separates input features and target.
- Splits the dataset into 80% training and 20% testing.
- Applies StandardScaler to standardize the features.

Output:
- `data/processed/train.csv`
- `data/processed/test.csv`

### Stage 3: Feature Engineering

File: `src/pipeline/features.py`

Two additional squared features are created:

1. `mean_radius_squared`
2. `mean_perimeter_squared`

These features are calculated by squaring the corresponding standardized input columns.

Output:
- `data/features/train_features.csv`
- `data/features/test_features.csv`

### Stage 4: Data Validation

File: `src/pipeline/validate.py`

The validation stage checks:
- Required columns.
- Missing values.
- Valid target values (0 and 1).
- Existence of training and testing files.

The validation script successfully detects invalid target values.

## 4. DVC Pipeline Automation

DVC is used to automate and manage the data pipeline.

The pipeline is configured in `dvc.yaml`.

The execution order is:

Collect → Preprocess → Features → Validate

Commands used:

```bash
dvc init
dvc dag
dvc repro
dvc status
dvc push
```

## 5. DVC Remote Storage

A local DVC remote named `localstorage` is configured.

Storage path:

`D:\DVC_Remote_Breast_Cancer`

The command `dvc push` successfully transferred five tracked files to the remote storage.

## 6. Validation Testing

An invalid target value (5) was temporarily inserted into the training dataset.

The validation script correctly raised:

`ValueError: Invalid target values in training data!`

The original dataset was restored, and validation was executed successfully again.

## 7. Final Result

The automated Breast Cancer data pipeline was successfully implemented using Python and DVC.

All four pipeline stages executed successfully, the data was validated, and the tracked data was pushed to local DVC remote storage.
