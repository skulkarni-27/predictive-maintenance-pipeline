#!/usr/bin/env python3
"""Train Decision Tree model with MLflow tracking"""
import os
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import hf_hub_download
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "SharleyK")
DATASET_NAME = os.getenv("DATASET_NAME", "PredictiveMaintenance")
repo_id = f"{HF_USERNAME}/{DATASET_NAME}"

# Set up MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Predictive_Maintenance")

logger.info("Loading data from Hugging Face...")

# Download train and test data
train_file = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                              filename="train_scaled.csv", token=HF_TOKEN)
test_file = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                             filename="test_scaled.csv", token=HF_TOKEN)

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

X_train = train_df.drop('engine_condition', axis=1)
y_train = train_df['engine_condition']
X_test = test_df.drop('engine_condition', axis=1)
y_test = test_df['engine_condition']

logger.info("Training Decision Tree...")

param_grid = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

with mlflow.start_run(run_name="Decision_Tree"):
    mlflow.set_tag("model_type", "Decision Tree")
    
    dt_model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Log parameters
    mlflow.log_params(grid_search.best_params_)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log model
    mlflow.sklearn.log_model(best_model, "model")
    
    # Save model locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/decision_tree.pkl")
    
    # Save metrics
    metrics = {
        "model": "Decision Tree",
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }
    
    os.makedirs("outputs/models", exist_ok=True)
    with open("outputs/models/decision_tree_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"✓ Decision Tree trained! F1-Score: {f1:.4f}")
