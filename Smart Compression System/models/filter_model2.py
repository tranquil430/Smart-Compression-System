"""
Adaptive Learning & User Feedback Module.

This module manages the secondary filtering model (Model 3) which learns from
user behavior. It handles logging of runtime actions and conditional retraining
of the RandomForest classifier based on new data.
"""

import os
import joblib
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# Constants
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_PATH, "user_interaction.log.csv")
MODEL_FILE = os.path.join(BASE_PATH, "2nd_filter.pkl")

# Encoders
LE_TYPE_FILE = os.path.join(BASE_PATH, "le_filetype.pkl")
LE_TIME_FILE = os.path.join(BASE_PATH, "le_time.pkl")
LE_ACTION_FILE = os.path.join(BASE_PATH, "le_action.pkl")

MIN_LOGS_FOR_RETRAIN = 20

MODEL_PARAMS = {
    'n_estimators': 50,
    'max_depth': 8,
    'min_samples_split': 5,
    'min_samples_leaf': 3,
    'max_features': 'sqrt',
    'random_state': 42
}

def setup_log_file():
    """Initializes the CSV log file if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["file_type", "file_size", "similar_history", "time_of_day", "action"])
        df.to_csv(LOG_FILE, index=False)

def log_user_action(file_type, file_size, similar_history, time_of_day, action):
    """Appends a new interaction event to the log."""
    setup_log_file()
    new_entry = pd.DataFrame([[file_type, file_size, similar_history, time_of_day, action]],
                             columns=["file_type", "file_size", "similar_history", "time_of_day", "action"])
    new_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)

def retrain_model_if_needed():
    """Checks log volume and retrains the model if sufficient data exists."""
    if not os.path.exists(LOG_FILE):
        return

    try:
        data = pd.read_csv(LOG_FILE)
    except pd.errors.EmptyDataError:
        return

    if len(data) < MIN_LOGS_FOR_RETRAIN:
        return

    print(f"[ADAPTIVE] Retraining model with {len(data)} interaction logs...")

    # Load or Create Encoders
    try:
        le_type = joblib.load(LE_TYPE_FILE)
        le_time = joblib.load(LE_TIME_FILE)
        le_action = joblib.load(LE_ACTION_FILE)
        
        # Transform (Handling unknown labels by retraining encoders if necessary, 
        # but for stability we typically just extend or ignore. Here we refit for simplicity.)
        # In a strict production env, you'd handle unseen labels more gracefully.
        data["file_type"] = le_type.fit_transform(data["file_type"])
        data["time_of_day"] = le_time.fit_transform(data["time_of_day"])
        data["action"] = le_action.fit_transform(data["action"])
        
    except (FileNotFoundError, ValueError):
        # Initial fit
        le_type = LabelEncoder()
        le_time = LabelEncoder()
        le_action = LabelEncoder()
        
        data["file_type"] = le_type.fit_transform(data["file_type"])
        data["time_of_day"] = le_time.fit_transform(data["time_of_day"])
        data["action"] = le_action.fit_transform(data["action"])

    # Prepare Data
    X = data[["file_type", "file_size", "similar_history", "time_of_day"]]
    y = data["action"]

    # Balance Classes
    classes = np.unique(y)
    if len(classes) > 1:
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
    else:
        class_weight_dict = None

    # Train
    model = RandomForestClassifier(**MODEL_PARAMS, class_weight=class_weight_dict)
    model.fit(X, y)

    # Save Artifacts
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le_type, LE_TYPE_FILE)
    joblib.dump(le_time, LE_TIME_FILE)
    joblib.dump(le_action, LE_ACTION_FILE)
    
    print(f"[ADAPTIVE] Model updated successfully.")

if __name__ == "__main__":
    # If run directly, perform a force retrain
    print("Running manual adaptive model update...")
    setup_log_file()
    retrain_model_if_needed()