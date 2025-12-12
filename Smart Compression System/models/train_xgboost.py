"""
Compression Algorithm Optimizer Training Pipeline.

This script trains an XGBoost classifier to predict the optimal compression algorithm
(7z, ZIP, RAR) based on file characteristics such as size and entropy.
It handles feature engineering, class balancing, and hyperparameter tuning.
"""

import os
import sys
import pickle
import pandas as pd
import xgboost as xgb
import scipy.stats as stats
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

# Configuration
DATASET_PATH = "data/compression_metrics.csv"  # Ensure this path is correct relative to execution
MODEL_OUTPUT_PATH = "models/xgb.pkl"

def load_data(path):
    """Loads and validates the training dataset."""
    if not os.path.exists(path):
        # Fallback for flat directory structures during dev
        if os.path.exists("xgboost_dataset.csv"):
            return pd.read_csv("xgboost_dataset.csv")
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)

def engineer_features(df):
    """Generates features for the model."""
    epsilon = 1e-6
    # Entropy-to-Size Ratio: Helps identify highly compressible files
    df['size_entropy_ratio'] = df['original_size'] / (df['entropy'] + epsilon)

    # Binning file sizes for categorical context
    bins = [0, 1_048_576, 10_485_760, 104_857_600, float('inf')]
    labels = ['small', 'medium', 'large', 'huge']
    df['size_bin'] = pd.cut(df['original_size'], bins=bins, labels=labels, right=False)
    
    return df, bins, labels

def train_optimizer_model():
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        df = load_data(DATASET_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print(f"Loaded {len(df)} records. Class distribution:\n{df['best_algo'].value_counts()}")

    # Feature Engineering
    df, bins, bin_labels = engineer_features(df)

    # Encoding
    le = LabelEncoder()
    df["best_algo_encoded"] = le.fit_transform(df["best_algo"])

    # One-hot encode categorical features
    df_enc = pd.get_dummies(df, columns=["extension", "size_bin"])

    # Define Feature Matrix (X) and Target (y)
    # removing outcome-dependent columns (compression stats)
    drop_cols = [
        "filename", "best_algo", "best_algo_encoded",
        "7z_size_bytes", "7z_time_sec", "7z_ratio",
        "zip_size_bytes", "zip_time_sec", "zip_ratio",
        "rar_size_bytes", "rar_time_sec", "rar_ratio"
    ]
    
    # Drop only columns that actually exist in the dataframe
    existing_drop = [c for c in drop_cols if c in df_enc.columns]
    X = df_enc.drop(existing_drop, axis=1)
    y = df_enc["best_algo_encoded"]
    
    feature_columns = X.columns.tolist()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    numerical_cols = ['original_size', 'entropy', 'size_entropy_ratio']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Class Balancing
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # Hyperparameter Tuning
    print("Starting RandomizedSearchCV...")
    param_dist = {
        "n_estimators": stats.randint(100, 1000),
        "max_depth": stats.randint(3, 10),
        "learning_rate": stats.uniform(0.01, 0.3),
        "subsample": stats.uniform(0.7, 0.3),
        "colsample_bytree": stats.uniform(0.7, 0.3)
    }

    xgb_model = xgb.XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
    
    search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=25,
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train, sample_weight=sample_weights)
    best_model = search.best_estimator_
    
    print(f"Best Parameters: {search.best_params_}")

    # Evaluation
    print("\nModel Evaluation:")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Serialization
    # We save all artifacts needed for the inference pipeline
    artifacts = (
        best_model, 
        le, 
        feature_columns, 
        scaler, 
        bins, 
        bin_labels, 
        numerical_cols
    )
    
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(artifacts, f)
        
    print(f"Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train_optimizer_model()