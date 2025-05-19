import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Step 1: Load Data
logging.info("Loading data...")
train_file = "train.csv"
test_file = "test.csv"
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Separate features and target
X = train_data.drop(columns=["id", "breast_cancer"], errors="ignore")
y = train_data["breast_cancer"]

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=["number"]).columns
categorical_cols = X.select_dtypes(exclude=["number"]).columns

# Define transformers for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine transformers in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Step 2: Split Train/Validation Data
logging.info("Splitting train/validation data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test = test_data.drop(columns=["id"], errors="ignore")

# Preprocess training and validation data
logging.info("Preprocessing data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Step 3: Handle Class Imbalance with SMOTE
logging.info("Applying SMOTE...")
smote = SMOTE(random_state=42, sampling_strategy="auto")  # Automatically determine strategy
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

# Step 4: Define and Tune LightGBM Model
logging.info("Defining and tuning the model...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

lgbm = LGBMClassifier(random_state=42, class_weight="balanced")

random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_grid,
    scoring="roc_auc",
    n_iter=10,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

logging.info("Starting hyperparameter tuning...")
random_search.fit(X_train_balanced, y_train_balanced)

# Step 5: Evaluate the Best Model
logging.info("Evaluating the model...")
best_model = random_search.best_estimator_
print(f"Best Parameters: {random_search.best_params_}")

y_val_pred = best_model.predict_proba(X_val_processed)[:, 1]
auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation AUC: {auc:.4f}")

# Step 6: Predict on Test Set
logging.info("Predicting on test set...")
y_test_pred = best_model.predict_proba(X_test_processed)[:, 1]

# Step 7: Create Submission File
logging.info("Creating submission file...")
submission = pd.DataFrame({
    "id": test_data["id"],
    "breast_cancer": y_test_pred
})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

