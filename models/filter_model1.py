"""
Baseline Decision Filter Training.

Trains a Random Forest classifier on synthetic data to enforce basic archival policies
(e.g., ignoring system files, skipping recently accessed files) before the adaptive
system takes over.
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

MODEL_OUTPUT_PATH = "models/1st_filter.pkl"

def generate_synthetic_data(n_samples=1000):
    """Generates a synthetic dataset representing typical file system metadata."""
    print(f"Generating {n_samples} synthetic file records...")
    data = []
    extensions = [".docx", ".jpg", ".zip", ".mp3", ".sql", ".pdf", ".iso", ".txt", ".mp4", ".pptx", ".csv", ".json", ".log"]
    
    for _ in range(n_samples):
        # Log-normal size distribution
        size_mb = max(0.1, np.random.lognormal(mean=np.log(50), sigma=1.5))
        
        # Skewed dates (older files are more common)
        days_since_access = np.random.triangular(left=0.1, mode=90, right=730) 
        days_since_mod = np.random.triangular(left=0.1, mode=60, right=730)
        ext = random.choice(extensions)
        
        # Policy Logic (Ground Truth Generation)
        label = "compress"
        
        # Rule: Ignore very new files
        if days_since_access < 1 or days_since_mod < 1:
            label = "ignore"
        # Rule: Often ignore already compressed formats unless very old
        elif ext in [".jpg", ".zip", ".mp3", ".mp4", ".iso"]:
            label = "ignore" if random.random() > 0.2 else "compress"
            
        data.append([days_since_access, days_since_mod, size_mb, ext, label])

    return pd.DataFrame(data, columns=["days_since_access", "days_since_mod", "size_mb", "ext", "label"])

def train_baseline_model():
    df = generate_synthetic_data()

    X = df[["days_since_access", "days_since_mod", "size_mb", "ext"]]
    y = df["label"]

    # Preprocessing Pipeline
    # Scale numbers, One-Hot Encode extensions
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["days_since_access", "days_since_mod", "size_mb"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["ext"])
        ],
        remainder="passthrough"
    )

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    accuracy = model_pipeline.score(X_test, y_test)
    print(f"Baseline Model Accuracy: {accuracy*100:.1f}%")
    print(classification_report(y_test, model_pipeline.predict(X_test)))
    
    # Save
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(model_pipeline, f)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train_baseline_model()