#!/usr/bin/env python3
"""Train XGBoost model with MLflow tracking"""
import os, logging, pandas as pd, mlflow, mlflow.sklearn, joblib, json
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import hf_hub_download
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "SharleyK")
DATASET_NAME = os.getenv("DATASET_NAME", "PredictiveMaintenance")
repo_id = f"{HF_USERNAME}/{DATASET_NAME}"

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Predictive_Maintenance")

# Load data
train_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="train_scaled.csv", token=HF_TOKEN)
test_file = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="test_scaled.csv", token=HF_TOKEN)

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

X_train = train_df.drop('engine_condition', axis=1)
y_train = train_df['engine_condition']
X_test = test_df.drop('engine_condition', axis=1)
y_test = test_df['engine_condition']

logger.info("Training XGBoost...")

param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]}

with mlflow.start_run(run_name="XGBoost"):
    mlflow.set_tag("model_type", "XGBoost")
    
    model = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    mlflow.sklearn.log_model(best_model, "model")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/xgboost.pkl")
    
    logger.info(f"✓ XGBoost trained! F1-Score: {f1:.4f}")
