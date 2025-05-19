# Breast Cancer Prediction with LightGBM

This project implements a complete pipeline for predicting breast cancer likelihood using a LightGBM classifier. It covers data loading, preprocessing, handling class imbalance, model training with hyperparameter tuning, evaluation, and generating a submission file.

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Data](#data)
* [Usage](#usage)
* [Pipeline Steps](#pipeline-steps)

  * [1. Load Data](#1-load-data)
  * [2. Preprocessing](#2-preprocessing)
  * [3. Train/Validation Split](#3-trainvalidation-split)
  * [4. Handle Imbalance (SMOTE)](#4-handle-imbalance-smote)
  * [5. Model Definition & Hyperparameter Tuning](#5-model-definition--hyperparameter-tuning)
  * [6. Evaluation](#6-evaluation)
  * [7. Test Prediction & Submission](#7-test-prediction--submission)
* [Logging](#logging)
* [Results](#results)
* [License](#license)
* [Author](#author)

## Overview

This script reads training and test datasets, preprocesses features (imputing missing values, scaling numeric columns, encoding categorical ones), balances the training data using SMOTE, trains a LightGBM classifier with randomized hyperparameter search, evaluates the model on a validation set using ROC AUC, and generates a submission CSV.

## Features

* Data loading from CSV files
* Missing value imputation for numeric and categorical data
* Feature scaling and one-hot encoding
* Train/validation split with stratification
* Handling class imbalance with SMOTE
* Hyperparameter tuning using `RandomizedSearchCV`
* Model evaluation (ROC AUC)
* Generation of a submission file
* Detailed logging of each step

## Prerequisites

* Python 3.7 or higher
* `pandas`
* `scikit-learn`
* `imbalanced-learn`
* `lightgbm`

## Installation

```bash
# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows

# Install dependencies
pip install pandas scikit-learn imbalanced-learn lightgbm
```

## Project Structure

```text
project-root/
├── train.csv           # Training data with `id` and `breast_cancer` target
├── test.csv            # Test data with `id` column
├── train_and_predict.py  # Main Python script (the code snippet)
└── submission.csv      # Generated submission file (output)
```

## Data

* **train.csv**: Contains features, an `id` column, and a `breast_cancer` target column (0/1).
* **test.csv**: Contains features and an `id` column for which predictions will be made.

Place both CSV files in the project root before running the script.

## Usage

Run the Python script from the project root:

```bash
python train_and_predict.py
```

This will:

1. Load `train.csv` and `test.csv`.
2. Preprocess features and split the training data.
3. Apply SMOTE to address class imbalance.
4. Perform randomized search for LightGBM hyperparameters.
5. Evaluate the best model and print validation ROC AUC.
6. Generate `submission.csv` with columns `id` and `breast_cancer` probabilities.

## Pipeline Steps

### 1. Load Data

* Reads `train.csv` and `test.csv` using `pandas.read_csv`.

### 2. Preprocessing

* **Numeric columns**: Impute missing values (mean) and scale with `StandardScaler`.
* **Categorical columns**: Impute missing values (most frequent) and one-hot encode.

### 3. Train/Validation Split

* Splits the training set into train/validation (80/20) with stratification on the target.

### 4. Handle Imbalance (SMOTE)

* Uses `SMOTE` to oversample the minority class in the training data.

### 5. Model Definition & Hyperparameter Tuning

* Defines `LGBMClassifier` with `class_weight='balanced'`.
* Uses `RandomizedSearchCV` over a grid of hyperparameters:

  * `n_estimators`: \[100, 200]
  * `max_depth`: \[3, 5]
  * `learning_rate`: \[0.05, 0.1]
  * `subsample`: \[0.8, 1.0]
  * `colsample_bytree`: \[0.8, 1.0]
* Optimizes for ROC AUC with 3-fold cross-validation.

### 6. Evaluation

* Prints best hyperparameters.
* Computes ROC AUC on the validation set.

### 7. Test Prediction & Submission

* Predicts probabilities on the test set.
* Writes `submission.csv` with `id` and `breast_cancer` probability scores.

## Logging

The script uses Python's `logging` module to output informative messages at each major step (data loading, preprocessing, SMOTE, tuning, evaluation, and submission creation).

## Results

* **Validation AUC**: Displayed in the console after tuning.
* **submission.csv**: Contains probabilistic predictions for the test set.

## License

This project is released under the MIT License.

## Author

Created by Qudsi Aljabiri.
